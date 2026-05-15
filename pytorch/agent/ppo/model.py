import torch 
import torch.nn as nn 
import torch.nn.functional as F
import gymnasium as gym 
from gymnasium import spaces 
from torch.distributions import Normal 


class Actor(nn.Module):

    def __init__(self, obs_dim,
                 action_dim,
                 hidden = 256,
                 log_std_init = -0.5,
                 ):
        super().__init__()

        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), 
                                 nn.Tanh(),
                                 nn.Linear(hidden,hidden),
                                 nn.Tanh())
        
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim)* log_std_init)

    def get_dist(self, obs_features):
        mean , std = self.forward(obs_features)
        return Normal(mean, std)

    def forward(self, obs_features):

        feat = self.net(obs_features)
        mean = self.mean(feat)
        log_std = torch.clamp(self.log_std.expand_as(mean), -5,2)
        std = torch.exp(log_std)

        return mean , std 

    def sample_action(self, obs_features,
                      deterministic_bool = False ):
        dist = self.get_dist(obs_features)

        if deterministic_bool:
            raw_action = dist.mean 
        else:
            raw_action = dist.rsample()

        # bounded action 
        action = torch.clamp(raw_action, -1.0, 1.0)

        log_prob = dist.log_prob(raw_action).sum(dim = -1)

        return action, log_prob

    def get_log_prob(self,obs_features, action):
        dist = self.get_dist(obs_features)
        return dist.log_prob(action).sum(dim = -1)

    def get_entropy(self, obs_features):
        dist = self.get_dist(obs_features)
        return dist.entropy().sum(dim = -1)

class Critic(nn.Module):

    def __init__(self, obs_dim , 
                 action_dim, 
                 hidden = 256):
        super().__init__()
        
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), 
                                 nn.Tanh(),
                                 nn.Linear(hidden, hidden),
                                 nn.Tanh(), 
                                 nn.Linear(hidden,1))
    
        self.action_dim = action_dim

    def forward(self, obs_features):
        return self.net(obs_features)


