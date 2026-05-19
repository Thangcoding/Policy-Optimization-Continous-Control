import torch 
import torch.nn as nn 
import torch.nn.functional as F
import gymnasium as gym 
from gymnasium import spaces 
from ...utils.feature_extractor import BaseFeatureExtractor, FeatureExtractorMLP,FeatureExtractorCNN
from torch.distributions import Normal, Categorical 


class ContinuousPolicyHead(nn.Module):
    def __init__(
        self,                       
        feature_dim: int,           
        action_dim: int, 
        max_action: int, 
        min_action: int,            
        log_std_init: float = -0.5, 
    ):
        '''
                
        - Scale action with action range [l, h]:

                            scale = (h - l) / 2 ,  bias = (h + l) / 2  

        - Action clipped and using change of variable to compute accurately log_prob:
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

    def get_dist(self, obs_features):
        mean, std = self.forward(obs_features)
        return Normal(mean, std)

    def sample_action(self, obs_features,
                        deterministic_bool = False ):
        dist = self.get_dist(obs_features)

        if deterministic_bool:
            raw_action = dist.mean 
        else:
            raw_action = dist.sample()

        # bounded action 
        tanh_action = torch.tanh(raw_action)

        # scale action
        action = self.action_scale*tanh_action + self.action_bias

        # change of variable 
        log_prob = dist.log_prob(raw_action).sum(dim = -1)

        log_prob = log_prob - torch.sum(torch.log(self.action_scale*(1 - tanh_action.pow(2)) + 1e-6), dim = -1)

        return action, log_prob

    def get_log_prob(self,obs_features, action):
        dist = self.get_dist(obs_features)

        # raw action 
        tanh_action = (action - self.action_bias) / self.action_scale 
        tanh_action = torch.clamp(tanh_action, -1 + 1e-6, 1 - 1e-6)
        raw_action = torch.atanh(tanh_action)

        # inverse change of variable 
        log_prob = dist.log_prob(raw_action).sum(dim = -1)

        correction = torch.sum(torch.log(self.action_scale*(1 - tanh_action.pow(2)) + 1e-6),dim = -1)

        log_prob = log_prob - correction

        return log_prob

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

class CriticHead(nn.Module):

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
            max_action = action_space.high 
            min_action = action_space.low 
            action_dim = action_space.shape[0]
            self.policy = ContinuousPolicyHead(action_dim = action_dim,
                                                feature_dim= feature_dim,
                                                max_action= max_action,
                                                min_action= min_action)
        else:
            raise NotImplementedError("Unsupported action space")

        self.critic = CriticHead(feature_dim)

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