import torch  
import torch.nn as nn  
import numpy as np 
from torch.distributions import Normal 

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden = 256,log_std_init= -0.5, max_action = 1):
        '''
        Action clipped and using change of variable to compute accurately log_prob

                                                z ~ N(μ,σ)
                                                a = tanh(z) 
                                            p(a) = p(z) |dz/da|
                                        log(p(a)) = log(p(z)) - log(|da/dz|)
                                                = log(p(z)) - log(1 - tanh^{2}(z)) 
                                                = log(p(z)) - log(1 - a^{2})
        
        '''
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
        )
 
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim)* log_std_init)
    
    def forward(self, obs: torch.tensor):
        
        obs_feature = self.net(obs)

        mean = self.mean(obs_feature)
        log_std = torch.clamp(self.log_std.expand_as(mean),-5,2)
        std = torch.exp(log_std)

        return mean , std 
    
    def get_dist(self, obs: torch.tensor):
        mean , std = self.forward(obs)
        return Normal(mean, std)
    
    def sample(self, obs: torch.tensor,
                deterministic_bool : bool = False):
        
        dist = self.get_dist(obs)
        
        if deterministic_bool: 
            z = dist.mean 
        else:
            z = dist.rsample()

        action = torch.tanh(z) 

        # compute log_prob 
        log_prob = dist.log_prob(z)
     
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6))

        log_prob = log_prob.sum(dim = -1)

        return action, log_prob.unsqueeze(-1)

    def get_log_prob(self, obs: torch.tensor , action: torch.tensor):

        dist = self.get_dist(obs)

        z = dist.rsample()

        action = torch.tanh(z) 

        # compute log_prob 
        log_prob = dist.log_prob(z)
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6))

        log_prob = log_prob.sum(dim = -1)

        return log_prob.unsqueeze(-1)

class Critic(nn.Module):

    def __init__(self, obs_dim , action_dim , hidden = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim , hidden), 
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )
    
    def forward(self, obs, action):

        return self.net(torch.cat([obs,action], dim = 1))
