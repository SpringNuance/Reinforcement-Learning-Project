import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
from torch.distributions import Categorical
import  torch


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, max_action, hidden_size=32):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.max_action = max_action

        self.fc1_actor = torch.nn.Linear(state_space, hidden_size)
        self.fc2_actor = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3_actor = torch.nn.Linear(hidden_size, action_space)

        # Learnable log standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))

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

        action_mean = self.max_action * torch.tanh(x_actor)
        #print(action_mean.shape)
        # Standard deviation from logstd parameter
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(self.actor_logstd)

        # Parameterize the normal distribution
        action_dist = Normal(action_mean, action_std)
        # Treat each action dimension as independent
        action_dist = Independent(action_dist, 1)

        x_critic = self.fc1_critic(x)
        x_critic = F.relu(x_critic)
        x_critic = self.fc2_critic(x_critic)
        x_critic = F.relu(x_critic)
        x_critic = self.fc3_critic(x_critic)

        return action_dist, x_critic

    def set_logstd_ratio(self, ratio_of_episodes):
        """
        Adjusts the log standard deviation of the policy's action distribution.
        A higher ratio means more exploration, while a lower ratio means more exploitation.
        """
        new_logstd = torch.full_like(self.actor_logstd, np.log(ratio_of_episodes))
        self.actor_logstd.data.copy_(new_logstd)

# # Layer initialization utility
# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

# class Policy(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size=32):
#         super(Policy, self).__init__()
#         # Actor network for mean of actions
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
#         )
#         # Learnable log standard deviation
#         self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

#         # Critic network for value function
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(state_dim, hidden_size)), nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, 1), std=1)
#         )

#     def forward(self, state):
#         # Get mean from actor network
#         action_mean = self.actor_mean(state)
        
#         # Standard deviation from logstd parameter
#         action_logstd = self.actor_logstd.expand_as(action_mean)
#         action_std = torch.exp(action_logstd)

#         # Parameterize the normal distribution
#         action_dist = Normal(action_mean, action_std)
#         # Treat each action dimension as independent
#         action_dist = Independent(action_dist, 1)

#         # Get state value from critic network
#         state_value = self.critic(state)

#         return action_dist, state_value


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




