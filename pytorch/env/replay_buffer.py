import numpy as np 
import torch 
import gymnasium as gym 
from gymnasium import spaces 


class ReplayBuffer:

    def __init__(self, buffer_size: int, 
                    num_envs: int, 
                    observation_space: spaces.Space, 
                    action_space: spaces.Space, 
                    device: torch.device 
                    ): 
        
        self.buffer_size = max(buffer_size // num_envs , 1) 
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device 

        # compute action and observation dim 
        self.observation_dim = self.observation_space.shape 

        if isinstance(self.action_space,spaces.Box):
            self.action_dim = self.action_space.shape[0]
        elif isinstance(self.action_space,spaces.Discrete):
            self.action_dim = 1
        elif isinstance(self.action_space,spaces.MultiDiscrete):
            self.action_dim = len(self.action_space.nvec)
        else:
            raise NotImplementedError("Unsupported action space")
        
        
        self.observation_buffer = np.zeros((buffer_size, *self.observation_dim),dtype = self.observation_space.dtype)
        self.next_observation_buffer = np.zeros((buffer_size, *self.observation_dim), dtype = self.observation_space.dtype)
        self.action_buffer = np.zeros((buffer_size, self.action_dim),dtype = np.float32)
        self.reward_buffer = np.zeros((buffer_size,1),dtype = np.float32)
        self.done_buffer = np.zeros((buffer_size, 1),dtype = np.float32)

        self.size = 0 
        self.pos = 0 

    def add(self, obs: np.ndarray, 
            next_obs: np.ndarray, 
            action: np.ndarray, 
            reward: np.ndarray, 
            done: np.ndarray):
        
        for idx in range(self.num_envs):

            self.observation_buffer[self.pos] = obs[idx]
            self.next_observation_buffer[self.pos] = next_obs[idx]
            self.action_buffer[self.pos] = action[idx]
            self.reward_buffer[self.pos] = reward[idx] 
            self.done_buffer[self.pos] = done[idx] 

            self.pos = (self.pos + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size : int):
        idxs = np.random.randint(0, self.size, size = batch_size)

        batch = dict(
            obs = torch.tensor(self.observation_buffer[idxs], device = self.device, dtype = torch.float32), 
            action = torch.tensor(self.action_buffer[idxs], device = self.device, dtype = torch.float32), 
            reward = torch.tensor(self.reward_buffer[idxs], device = self.device, dtype = torch.float32),
            next_obs = torch.tensor(self.next_observation_buffer[idxs], device = self.device, dtype = torch.float32), 
            done = torch.tensor(self.done_buffer[idxs], device = self.device, dtype = torch.float32)
        )

        return batch 

    def __len__(self):
        return self.size 



    