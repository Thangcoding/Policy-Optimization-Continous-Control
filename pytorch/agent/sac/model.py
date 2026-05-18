import torch  
import torch.nn as nn  
import numpy as np 
from torch.distributions import Normal 

class Actor(nn.Module):

    def __init__(self, obs_dim,
                 action_dim,
                 max_action,
                 min_action,  
                 hidden = 256,
                 ):
        '''
        Action clipped and using change of variable to compute accurately log_prob

                                                z ~ N(μ,σ)
                                                a = scale*tanh(z) + bias  
                                            p(a) = p(z) |dz/da|
                                        log(p(a)) = log(p(z)) - log(|da/dz|)
                                                = log(p(z)) - log(1 - tanh^{2}(z)) - log(scale) 
                                            
        
        '''

        super().__init__()  

        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (max_action - min_action) / 2,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "action_bias", 
            torch.as_tensor(
                (max_action + min_action) / 2, 
                dtype = torch.float32
            )
        )

        self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), 
                                 nn.ReLU(),
                                 nn.Linear(hidden,hidden),
                                 nn.ReLU())
        
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def get_dist(self, obs_features):
        mean , std = self.forward(obs_features)
        return Normal(mean, std)

    def forward(self, obs_features):

        feat = self.net(obs_features)
        mean = self.mean(feat)
        
        log_std = torch.clamp(self.log_std(feat), -20,2)
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
        tanh_action = torch.tanh(raw_action)

        # scale action 
        action = self.action_scale*tanh_action + self.action_bias

        # change of variable 
        log_prob = dist.log_prob(raw_action).sum(dim = -1)

        log_prob = log_prob - torch.sum(torch.log(self.action_scale*(1 - tanh_action.pow(2)) + 1e-6), dim = -1)

        return action, log_prob.unsqueeze(-1)

    def get_log_prob(self,obs_features, action):
        dist = self.get_dist(obs_features)

        # raw action 
        tanh_action = (action - self.action_bias) /self.action_scale
        tanh_action = torch.clamp(tanh_action , -1 + 1e-6, 1 - 1e-6)
        raw_action = torch.atanh(tanh_action)

        # inverse change of variable 
        log_prob = dist.log_prob(raw_action).sum(dim = -1)

        correction = torch.sum(torch.log(self.action_scale*(1 - tanh_action.pow(2)) + 1e-6),dim = -1)

        log_prob = log_prob - correction

        return log_prob.unsqueeze(-1)

    def get_entropy(self, obs_features):
        dist = self.get_dist(obs_features)
        return dist.entropy().sum(dim = -1)

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
