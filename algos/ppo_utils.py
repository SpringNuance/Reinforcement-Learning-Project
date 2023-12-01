import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from torch.distributions import Categorical
import  torch

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1_actor = torch.nn.Linear(state_space, hidden_size)
        self.fc2_actor = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_actor = torch.nn.Linear(hidden_size, action_space)

        self.fc1_critic = torch.nn.Linear(state_space, hidden_size)
        self.fc2_critic = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_critic = torch.nn.Linear(hidden_size, 1)

        self.init_weights()

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

        x_critic = self.fc1_critic(x)
        x_critic = F.relu(x_critic)
        x_critic = self.fc2_critic(x_critic)
        x_critic = F.relu(x_critic)
        x_critic = self.fc3_critic(x_critic)

        action_probs = F.softmax(x_actor, dim=-1)
        action_dist = Categorical(action_probs)

        return action_dist, x_critic
