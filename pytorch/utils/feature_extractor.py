import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import gymnasium as gym

class BaseFeatureExtractor(nn.Module):
    
    def __init__(self, observation_space: gym.Space | None = None, feature_dim: int | None = None , **kwargs):
        super().__init__() 
        ''' 
        Base class for feature extraction network 
        Args: 
            observation_space: dimension of observation 
            feature_dim : output dim of feature extractor 
        '''
        self.observation_space = observation_space
        self.feature_dim = feature_dim 

    def forward(self, obs: torch.tensor):
        raise NotImplementedError
    
class FeatureExtractorMLP(BaseFeatureExtractor):
    def __init__(self, observation_space: gym.Space,
                feature_dim : int = 512):

        ''' 
        MLP used to  extract feature with vector observation 
            Vector observation: D x 1
        '''
        super().__init__(observation_space, feature_dim)

        self.input_dim = observation_space.shape[0]

        self.network = nn.Sequential(nn.Linear(observation_space.shape[0],32), 
                                     nn.ReLU(), 
                                     nn.Linear(32, 64), 
                                     nn.ReLU(),
                                     nn.Linear(64,256), 
                                     nn.ReLU(), 
                                     nn.Linear(256, feature_dim))
        
    def forward(self, obs: torch.Tensor):
        # obs (batch x obs_dim)

        return self.network(obs)
    

class FeatureExtractorCNN(BaseFeatureExtractor):
    def __init__(self, observation_space: gym.Space,
                feature_dim: int = 512):

        ''' 
        CNN used to extract feature with the image observation 
            
            image observation: C x W X L 
        '''

        super().__init__(observation_space, feature_dim)
        self.num_channel = observation_space.shape[0]
    
        self.image_size = observation_space.shape[1:]
    
        self.cnn = nn.Sequential(nn.Conv2d(self.num_channel, 32, 8, stride = 4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride = 2), 
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride = 1), 
                                 nn.ReLU(), 
                                 nn.Flatten())

        # compute flatten size output 
        with torch.no_grad(): 
            sample = torch.zeros(1, self.num_channel, self.image_size[0], self.image_size[1])
            flatten_size = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flatten_size, self.feature_dim), 
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs/ 255.0 # image normalize 
        obs_feature = self.linear(self.cnn(obs))

        return obs_feature 
    



