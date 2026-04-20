import numpy as np
import torch 
import gymnasium as gym  
from gymnasium import spaces 
from .vectorize_env import VectorEnv

class RolloutBuffer: 

    def __init__(self, buffer_size: int,
                num_envs: int, 
                observation_space: gym.Space, 
                action_space: gym.Space, 
                device: torch.device
                ):
        '''
        RolloutBuffer is a data structure to store and compute value for gradient update of policy and critic
            observation : state information 
            action : action value 
            reward : r(a_t,s_t)
            done : terminate indicator 
            value: value of the state 
            return : return value of the state
            log_prob : log probability of the action  
            advantgae : age advantage value of the state   
        '''
        
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device 

        self.observation_space = observation_space
        self.action_space = action_space

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

        # buffer store
        self.observation_buffer = np.zeros((buffer_size, self.num_envs, *self.observation_dim),self.observation_space.dtype)
        self.action_buffer = np.zeros((buffer_size,self.num_envs,self.action_dim), dtype = np.float32)
        self.reward_buffer = np.zeros((buffer_size,self.num_envs),dtype = np.float32)
        self.done_buffer = np.zeros((buffer_size, self.num_envs),dtype = np.float32)
        self.value_buffer = np.zeros((buffer_size, self.num_envs), dtype = np.float32)
        self.return_buffer = np.zeros((buffer_size, self.num_envs),dtype = np.float32)
        self.log_prob_buffer = np.zeros((buffer_size, self.num_envs), dtype = np.float32)
        self.advantage_buffer = np.zeros((buffer_size,self.num_envs),dtype = np.float32)

        # index buffer 
        self.pos = 0
        
    def reset(self) -> None:
        '''reset the buffer store'''
        self.observation_buffer = np.zeros((self.buffer_size, self.num_envs, *self.observation_dim),self.observation_space.dtype)
        self.action_buffer = np.zeros((self.buffer_size,self.num_envs,self.action_dim), dtype = np.float32)
        self.reward_buffer = np.zeros((self.buffer_size,self.num_envs),dtype = np.float32)
        self.done_buffer = np.zeros((self.buffer_size, self.num_envs),dtype = np.float32)
        self.value_buffer = np.zeros((self.buffer_size, self.num_envs),dtype = np.float32)
        self.return_buffer = np.zeros((self.buffer_size, self.num_envs),dtype = np.float32)
        self.log_prob_buffer = np.zeros((self.buffer_size, self.num_envs), dtype = np.float32)
        self.advantage_buffer = np.zeros((self.buffer_size, self.num_envs),dtype = np.float32)

        self.pos = 0
    def add(self,
            obs: np.ndarray, 
            action: np.ndarray, 
            reward: np.ndarray,
            value: np.ndarray,
            log_prob: np.ndarray, 
            done: np.ndarray) -> None:
        ''' 
        adding sample in rollout buffer
        '''

        if isinstance(self.action_space, spaces.Discrete):
            action = np.expand_dims(action,axis = -1).astype(np.int64)

        elif isinstance(self.action_space, spaces.Box):
            action = action.astype(np.float32)

        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action = action.astype(np.int64)

        else:
            raise NotImplementedError
    

        self.observation_buffer[self.pos] = obs
        self.action_buffer[self.pos] = action
        self.reward_buffer[self.pos] = reward
        self.value_buffer[self.pos] = value
        self.log_prob_buffer[self.pos] = log_prob
        self.done_buffer[self.pos] = done 

        self.pos += 1

    def gae_and_return_value(self,last_value: np.ndarray,
                    gamma: float = 0.99, 
                    gae_lambda: float = 0.95) -> np.ndarray[np.float32]:
        '''

        Args:
            last_value : the value V(s_{T}) used to compute temporal different of state s_{T-1}
            values (buffer_size x num_envs): the list of  V(s_{t}) for all state in rollout 
            gamma : discouted factor 
            lambda : control variance and bias 
                TD(lamba = 1) ~ monte carlo estimate 
                TD(gae_lambda = 0) ~ temporal different 
        
        Recursive form GAE:
            GAE_{t}(γ,λ) = δ_t + (λγ)GAE_{t+1}(γ,λ)
                
            δ_t = r_t + gamma*V(s_{t+1}) - V(s_{t})
        
        '''

        length = self.buffer_size
        gae = np.zeros(self.num_envs, dtype = np.float32)

        for step in reversed(range(length)):

            if step == length - 1:
                next_value = last_value 
            else:
                next_value = self.value_buffer[step+1]*(1 - self.done_buffer[step])

            curr_value = self.value_buffer[step]

            delta = self.reward_buffer[step] + gamma*next_value - curr_value
            
            gae = delta + gae_lambda*gamma*gae*(1 - self.done_buffer[step])

            self.advantage_buffer[step] = gae

        self.return_buffer = self.advantage_buffer + self.value_buffer

        return self.advantage_buffer
    
    def td_value(self):
        pass 

    def batch_data(self,batch_size: int = 64):

        # reshape to batch size 
        observation_batch = torch.tensor(self.observation_buffer, dtype = torch.float32).reshape(-1,*self.observation_dim)
        action_batch = torch.tensor(self.action_buffer, dtype = torch.float32).reshape(-1,self.action_dim)
        return_batch = torch.tensor(self.return_buffer,dtype = torch.float32).reshape(-1)
        advantage_batch = torch.tensor(self.advantage_buffer,dtype = torch.float32).reshape(-1)
        log_prob_batch = torch.tensor(self.log_prob_buffer,dtype = torch.float32).reshape(-1) 

        total_sample = self.buffer_size * self.num_envs
        idx = np.random.permutation(total_sample)

        for start in range(0, total_sample, batch_size):
            
            end = min(start + batch_size,total_sample)
            batch_idx = idx[start:end]

            # mini batch 
            yield{"obs":observation_batch[batch_idx].to(self.device),
                "action": action_batch[batch_idx].to(self.device),
                "advantage": advantage_batch[batch_idx].to(self.device), 
                "log_prob": log_prob_batch[batch_idx].to(self.device),
                "return": return_batch[batch_idx].to(self.device),
            }

if __name__ == '__main__':
    # test 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vec_env = VectorEnv(name_env= 'CartPole-v1')
    buffer = RolloutBuffer(buffer_size= 10, 
                        num_envs= 4,
                        gamma= 0.99,
                        observation_space= vec_env.observation_space,
                        action_space= vec_env.action_space,
                        device= device) 
    
    for batch in buffer.batch_data(batch_size=5):
        print(batch)
    
