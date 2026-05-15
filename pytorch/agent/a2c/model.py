import torch 
import torch.nn as nn 
import torch.nn.functional as F
import gymnasium as gym 
from gymnasium import spaces 
from ...utils.feature_extractor import BaseFeatureExtractor, FeatureExtractorMLP,FeatureExtractorCNN
from torch.distributions import Normal, Categorical 


class ContinuousPolicyHead(nn.Module):
    """
    action with hard clip [-1,1]
    """

    def __init__(
        self,                       
        feature_dim: int,           
        action_dim: int,            
        log_std_init: float = -0.5, 
    ):
        super().__init__()

        self.action_dim = action_dim

        # Mean network
        self.mean = nn.Linear(feature_dim, action_dim)

        # Global std parameter (best for PPO stability)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)

    def forward(self, features):
        mean = self.mean(features)

        log_std = torch.clamp(self.log_std.expand_as(mean),-5,2)
        std = torch.exp(log_std)

        return mean, std

    def get_dist(self, features):
        mean, std = self.forward(features)
        return Normal(mean, std)

    def sample_action(
        self,
        obs_features,
        deterministic_bool=False
    ):
        dist = self.get_dist(obs_features)

        if deterministic_bool:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()

        # bounded action
        action = torch.clamp(raw_action, -1.0, 1.0)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        return action, log_prob

    def get_log_prob(self, obs_features, action):
        dist = self.get_dist(obs_features)
        return dist.log_prob(action).sum(dim=-1)

    def get_entropy(self, obs_features):
        dist = self.get_dist(obs_features)
        return dist.entropy().sum(dim=-1)

class DiscretePolicyHead(nn.Module):
    
    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()

        self.action_dim = action_dim
        self.logits = nn.Linear(feature_dim, self.action_dim)

    def forward(self, obs_features : torch.Tensor):
        return self.logits(obs_features)

    def sample_action(self, obs_features: torch.Tensor, 
                      deterministic_bool: bool = False):
        
        logits = self.forward(obs_features)

        dist = Categorical(logits= logits)

        if deterministic_bool:
            action = dist.mode
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim = -1)

        return action, log_prob

    def get_log_prob(self,obs_features: torch.Tensor ,action: torch.Tensor):
        
        logits = self.forward(obs_features)
        dist = Categorical(logits= logits)

        return dist.log_prob(action)

    def get_entropy(self, obs_features: torch.Tensor):
        
        logits = self.forward(obs_features)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if len(entropy.shape) > 1 :
            entropy = entropy.sum(dim = -1)
        return entropy

class ValueNetwork(nn.Module):

    def __init__(self, feature_dim: int = 512):
        super().__init__()

        self.value_net = nn.Linear(feature_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, obs_features: torch.Tensor) -> torch.Tensor:

        return self.value_net(obs_features)

class ActorCritic(nn.Module):

    def __init__(self,feature_network: str | type[BaseFeatureExtractor], 
                    observation_space : gym.Space,  
                    action_space : gym.Space, 
                    feature_dim: int): 
        super().__init__() 

        self.action_space = action_space 

        if isinstance(action_space, spaces.Discrete):
            # Discrete action
            action_dim = action_space.n 
            self.policy = DiscretePolicyHead(action_dim= action_dim, feature_dim=feature_dim)
        elif isinstance(action_space, spaces.Box):
            # Box action
            action_dim = action_space.shape[0]
            self.policy = ContinuousPolicyHead(action_dim = action_dim, feature_dim= feature_dim)
        else:
            raise NotImplementedError("Unsupported action space")

        self.critic = ValueNetwork(feature_dim)

        if isinstance(feature_network, str):
            if feature_network == 'MLP':
                self.network = FeatureExtractorMLP(observation_space = observation_space, feature_dim= feature_dim)
            elif feature_network == 'CNN':
                self.network = FeatureExtractorCNN(observation_space = observation_space, feature_dim=feature_dim)
            else:
                raise ValueError("Unknown feature network")
        else:
            self.network = feature_network
    
    def evaluate_action(self, obs : torch.Tensor, action: torch.Tensor) -> tuple:
        # evaluation action 
        obs_features = self.network(obs)
        log_prob  = self.policy.get_log_prob(obs_features,action)
        entropy = self.policy.get_entropy(obs_features)

        value = self.critic(obs_features).squeeze(-1)

        return log_prob, value, entropy
    
    def predict(self, obs: torch.Tensor,deterministic_bool = False) -> tuple:
        
        obs_features = self.network(obs)
        
        action, log_prob = self.policy.sample_action(obs_features= obs_features,
                                                     deterministic_bool= deterministic_bool)

        value = self.critic(obs_features).squeeze(-1)

        return action, log_prob, value 

if __name__ == '__main__':
    pass 