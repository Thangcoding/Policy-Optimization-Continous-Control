import torch 
import torch.nn as nn 

class Actor(nn.Module): 

    def __init__(self, obs_dim , action_dim,max_action,min_action, hidden = 256):
        super().__init__()

        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (max_action - min_action)/2,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "action_bias", 
            torch.as_tensor(
                (max_action + min_action)/2,
                dtype = torch.float32
            )
        )

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden,hidden), 
            nn.ReLU(),
            nn.Linear(hidden, action_dim), 
            nn.Tanh(),
        )

    def forward(self, obs: torch.tensor):   
        return self.action_scale*self.net(obs) + self.action_bias

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden = 256):
        super().__init__() 
        
        self.net = nn.Sequential( 
            nn.Linear(obs_dim + action_dim,hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, 1), 
        )

    def forward(self, obs: torch.tensor, action: torch.tensor):
        return self.net(torch.concat([obs, action], dim = 1))

