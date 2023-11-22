
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
        

    def forward(self, state):
        return 


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        True

    def forward(self, state, action):
        return

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        True
