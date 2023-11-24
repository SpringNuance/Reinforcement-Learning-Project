
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import numpy as np
import torch

# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


# Critic class. The critic is represented by a neural network.
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, state_shape[0]), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, state_shape[0]), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.extra = {}
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra:dict=None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device='cpu'):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[ind].to(device),
            action = self.action[ind].to(device), 
            next_state = self.next_state[ind].to(device), 
            reward = self.reward[ind].to(device), 
            not_done = self.not_done[ind].to(device), 
            extra = extra
        )
        return batch
    
    def get_all(self, device='cpu'):
        if self.extra:
            extra = {key: value[:self.size].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state = self.state[:self.size].to(device),
            action = self.action[:self.size].to(device), 
            next_state = self.next_state[:self.size].to(device), 
            reward = self.reward[:self.size].to(device), 
            not_done = self.not_done[:self.size].to(device), 
            extra = extra
        )
        return batch

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state