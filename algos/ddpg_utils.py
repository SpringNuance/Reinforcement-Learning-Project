
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
            nn.Linear(32, action_dim), nn.Tanh()
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

class CriticTD3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, 1]

# Critic class. The critic is represented by a neural network.
class CriticQR(nn.Module):
    def __init__(self, state_dim, action_dim, N=100):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, N))

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x) # output shape [batch, N]

class PotentialFunction(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_sanding = env.n_sanding
        self.n_no_sanding = env.n_no_sanding
        self.MAX_SIZE = env.MAX_SIZE
        self.MIN_SIZE = env.MIN_SIZE
        self.diagonal = np.sqrt(2) * (self.MAX_SIZE - self.MIN_SIZE)

    def forward(self, state):
        """
            - state: agent state [batch_size, state_dim]
            A state \( s \) is defined as:

        s = [(xRobot, yRobot), (xSand, ySand)1, (xSand, ySand)2, ..., (xSand, ySand)N, (xNoSand, yNoSand)1, (xNoSand, yNoSand)2, ..., (xNoSand, yNoSand)M)]

        where (xRobot, yRobot) is the robot position and (xSand, ySand) is the sand position and (xNoSand, yNoSand) is the no sand position.
            return potential_value [batch_size, 1]
        """
        # print(self.n_sanding, self.n_no_sanding)
        # print('state shape', state.shape)   # shape [batch_size, state_dim]
        # print('state[:, :2].unsqueeze(1)  shape', state[:, :2].unsqueeze(1).shape) # shape [batch_size, 1, 2]
        # print('state[:, 2:2+self.n_sanding*2].view(-1, self.n_sanding, 2) shape', state[:, 2:2+self.n_sanding*2].view(-1, self.n_sanding, 2).shape)     # shape [batch_size, n_sanding, 2]
        # print('state[:, 2+self.n_sanding*2:].view(-1, self.n_no_sanding, 2) shape', state[:, 2+self.n_sanding*2:].view(-1, self.n_no_sanding, 2).shape) # shape [batch_size, n_no_sanding, 2]
        distance_to_sand = torch.norm(state[:, :2].unsqueeze(1) - state[:, 2:2+self.n_sanding*2].view(-1, self.n_sanding, 2), dim=2)                # shape [batch_size, n_sanding]
        
        distance_to_nosand = torch.norm(state[:, :2].unsqueeze(1) - state[:, 2+self.n_sanding*2:].view(-1, self.n_no_sanding, 2), dim=2)            # shape [batch_size, n_no_sanding]
        
        avg_distance_to_sand = torch.mean(distance_to_sand, dim=1)                     # shape [batch_size, 1]
        
        avg_distance_to_nosand = torch.mean(distance_to_nosand, dim=1)                 # shape [batch_size, 1]
        
        potential_value = avg_distance_to_nosand/avg_distance_to_sand                # shape [batch_size, 1]
        return potential_value

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

class PrioritizedReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, alpha, beta, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        dtype = torch.uint8 if len(state_shape) == 3 else torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, state_shape[0]), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros((max_size, state_shape[0]), dtype=dtype)
        self.reward = torch.zeros((max_size, 1), dtype=dtype)
        self.not_done = torch.zeros((max_size, 1), dtype=dtype)
        self.priority = torch.zeros((max_size, 1), dtype=dtype)
        self.count = 0
        self.update_ab_interval = 10000
        self.extra = {}
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, timestep, extra:dict=None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1. - done)
        if self.count == 0:
            self.priority[self.ptr] = 1
        elif self.ptr < timestep:
            self.priority[self.ptr] = self._to_tensor(torch.max(torch.concat((self.priority[:self.ptr+1], self.priority[-(timestep-self.ptr):]))))
        else:
            self.priority[self.ptr] = self._to_tensor(torch.max(self.priority[self.ptr-timestep:self.ptr+1]))

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra: # init buffer
                    self.extra[key] = torch.zeros((self.max_size, *value.shape), dtype=torch.float32)
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.count += 1
        # if self.count > self.update_ab_interval:
        #     print('Update alpha and beta')
        #     self.alpha = max(self.alpha + 0.05, 0.7)
        #     self.beta = max(self.beta + 0.05, 1)
        #     self.count = 0


    def sample(self, batch_size, device='cpu'):
        self.probs = torch.pow(self.priority[:self.size+1], self.alpha)/torch.sum(torch.pow(self.priority[:self.size+1], self.alpha))
        ind = torch.multinomial(self.probs.view(-1), batch_size, replacement=True)
        weights = (self.size*self.probs[ind])**(-self.beta)
        weights = weights/torch.max(weights)

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
        return ind, batch, weights
    
    def update_priorities(self, ind, priorities):
        self.priority[ind] = self._to_tensor(priorities)

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
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state