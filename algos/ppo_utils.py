import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from torch.distributions import Categorical
import  torch



# # Actor-critic agent
# class Policy(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super().__init__()
#         self.max_action = max_action
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 32), nn.ReLU(),
#             nn.Linear(32, 32), nn.ReLU(),
#             nn.Linear(32, action_dim), nn.Tanh()
#         )

#     def forward(self, state):
#         return self.max_action * torch.tanh(self.actor(state))


# # Critic class. The critic is represented by a neural network.
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.value = nn.Sequential(
#             nn.Linear(state_dim+action_dim, 32), nn.ReLU(),
#             nn.Linear(32, 32), nn.ReLU(),
#             nn.Linear(32, 1))

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         return self.value(x) # output shape [batch, 1]
    

# class Policy(torch.nn.Module):
#     def __init__(self, state_space, action_space, hidden_size=32):
#         super(Policy, self).__init__()
#         self.fc1_actor = torch.nn.Linear(state_space, hidden_size)
#         self.fc2_actor = torch.nn.Linear(hidden_size, hidden_size)
#         self.fc3_actor = torch.nn.Linear(hidden_size, action_space)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if type(m) is torch.nn.Linear:
#                 torch.nn.init.normal_(m.weight, 0, 1e-1)
#                 torch.nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x_actor = self.fc1_actor(x)
#         x_actor = F.relu(x_actor)
#         x_actor = self.fc2_actor(x_actor)
#         x_actor = F.relu(x_actor)
#         x_actor = self.fc3_actor(x_actor)

#         # mean = self.fc3(x)
#         # std_dev = torch.ones_like(mean)  # Fixed standard deviation
#         # return Normal(mean, std_dev)
#         return 


# class Critic(torch.nn.Module):
#     def __init__(self, state_space, hidden_size=32):
#         super(Critic, self).__init__()
#         self.fc1_critic = torch.nn.Linear(state_space, hidden_size)
#         self.fc2_critic = torch.nn.Linear(hidden_size, hidden_size)
#         self.fc3_critic = torch.nn.Linear(hidden_size, 1)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if type(m) is torch.nn.Linear:
#                 torch.nn.init.normal_(m.weight, 0, 1e-1)
#                 torch.nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x_critic = self.fc1_critic(x)
#         x_critic = F.relu(x_critic)
#         x_critic = self.fc2_critic(x_critic)
#         x_critic = F.relu(x_critic)
#         x_critic = self.fc3_critic(x_critic)
#         return x_critic




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

        return action_dist, x_critic
