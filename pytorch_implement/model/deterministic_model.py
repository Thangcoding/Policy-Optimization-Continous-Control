import torch 
import torch.nn as nn 

class Actor(nn.Module): 

    def __init__(self, obs_dim , action_dim, max_action = None ):
        super().__init__()
        self.max_action = max_action 

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32,64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
    def forward(self, obs: torch.tensor):   
        if self.max_action is not None:
            return self.max_action*self.net(obs)
        return self.net(obs)

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__() 
        
        self.net = nn.Sequential( 
            nn.Linear(obs_dim + action_dim,32), 
            nn.ReLU(), 
            nn.Linear(32, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64,1), 
        )

    def forward(self, obs: torch.tensor, action: torch.tensor):
        return self.net(torch.concat([obs, action], dim = 1))

