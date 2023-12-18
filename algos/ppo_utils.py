import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from torch.distributions import Categorical
import  torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, max_action, hidden_size=64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.max_action = max_action

        self.fc1_actor = layer_init(torch.nn.Linear(state_space, hidden_size))
        self.fc2_actor = layer_init(torch.nn.Linear(hidden_size, hidden_size))
        self.fc3_actor = layer_init(torch.nn.Linear(hidden_size, action_space))

        # Fixed log standard deviation, not a learnable parameter
        self.actor_logstd = torch.full((1, action_space), fill_value=0)

        self.fc1_critic = layer_init(torch.nn.Linear(state_space, hidden_size))
        self.fc2_critic = layer_init(torch.nn.Linear(hidden_size, hidden_size))
        self.fc3_critic = layer_init(torch.nn.Linear(hidden_size, 1))

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.fc1_actor(x)
        x_actor = F.relu(x_actor)
        x_actor = self.fc2_actor(x_actor)
        x_actor = F.relu(x_actor)
        x_actor = self.fc3_actor(x_actor)

        action_mean = self.max_action * F.tanh(x_actor)
        action_std = torch.exp(self.actor_logstd)
        action_dist = Normal(action_mean, action_std)
        action_dist = Independent(action_dist, 1)

        x_critic = self.fc1_critic(x)
        x_critic = F.relu(x_critic)
        x_critic = self.fc2_critic(x_critic)
        x_critic = F.relu(x_critic)
        x_critic = self.fc3_critic(x_critic)

        # x_critic = self.critic(x)

        return action_dist, x_critic

    def set_logstd_ratio(self, ratio_of_episodes):
        """
        Adjusts the log standard deviation of the policy's action distribution.
        A higher ratio means more exploration, while a lower ratio means more exploitation.
        """
        self.actor_logstd = torch.full((1, self.action_space), fill_value=ratio_of_episodes)


