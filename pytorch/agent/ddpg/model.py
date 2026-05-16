import torch 
import torch.nn as nn 

class Actor(nn.Module): 

    def __init__(self, obs_dim , action_dim, hidden = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), 
            nn.Tanh(), 
            nn.Linear(hidden,hidden), 
            nn.Tanh(),
            nn.Linear(hidden, action_dim), 
            nn.Tanh(),
        )

    def forward(self, obs: torch.tensor):   
        return self.net(obs)

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden = 256):
        super().__init__() 
        
        self.net = nn.Sequential( 
            nn.Linear(obs_dim + action_dim,hidden), 
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(), 
            nn.Linear(hidden, 1), 
            nn.Tanh(), 
        )

    def forward(self, obs: torch.tensor, action: torch.tensor):
        return self.net(torch.concat([obs, action], dim = 1))

