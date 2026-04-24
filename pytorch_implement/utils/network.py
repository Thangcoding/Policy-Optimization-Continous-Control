import torch 
import torch.nn as nn 
import torch.nn.functional as F
import gymnasium as gym 
from gymnasium import spaces 
from ..utils.distributions import DiagGaussianAction, CategoricalAction,BernoulliAction,MultiCategoricalAction
from .feature_extractor import BaseFeatureExtractor, FeatureExtractorMLP,FeatureExtractorCNN

     
class ContinuousTanhPolicyHead(nn.Module):
                                   
    def __init__(self,action_dim: int,   
                    feature_dim: int,  
                    log_std_init: float = 0.0,
        ):
        
        '''
        The policy network for continuous action with parameter θ represented by πθ(a∣s) =N(μθ(s),σ(θ))

        Detail in implement: 
            - the mean network is dependent on the state: μ=fθ(s)
            - the standard diviation is independent on the state: σ=exp(log_std)
            - the log-std is a learnable parameter vector, initialized to log_std_init = 0.0 (σ = 1.0)

        Args: 
            action_dim : dimension of the action space 
            feature_extractor: network of feature extraction
            feature_dim : dimension of final output feature extraction
            log_std_init (float): initial valur for log standard deviation. 
        '''

        super().__init__()

        self.action_dim = action_dim

        # mean output with action_space values 
        self.mean = nn.Linear(feature_dim, self.action_dim)

        self.log_std_layer = nn.Linear(feature_dim, action_dim)

    def forward(self, obs_features: torch.Tensor) -> tuple:

        # bounded mean 
        mean = self.mean(obs_features) 

        #standard deviation 
        log_std = torch.clamp(self.log_std_layer(obs_features), -5,2)

        std = torch.exp(log_std)

        return mean, std 

    def sample_action(self, obs_features: torch.Tensor,
                    reparam_trick_bool: bool = False,
                    deterministic_bool : bool = False) -> tuple:

        ''' 
        Args:
            obs_features : vector features from feature-extract of the observation
            repram_trick : identiy sample action with reparameterization trick or not
            deterministic : identity sample action or use mean action

        Note:    
            * The sample of action can do in two way:
                - sample in a deterministic distribution without gradient respect to action, such as REINFORCE algorithm
                - sample in a parameteric distribution (also called reparameterization trick sample) used when the objective take gradient respect to action , 
                  such as SAC, MPO.

            * The output action must be bounded in the range [-1,1]

                - method 1: clip(action, -1, 1) 
                    => compute log_prob become not accurate  

                - method 2: using tanh function for the output of action  (also called tanh-squashed Gaussian sample)

                    => can use change of variable to compute accurately log_prob

                                            z ~ N(μ,σ)
                                            a = tanh(z) 
                                        p(a) = p(z) |dz/da|
                                    log(p(a)) = log(p(z)) - log(|da/dz|)
                                              = log(p(z)) - log(1 - tanh^{2}(z)) 
                                              = log(p(z)) - log(1 - a^{2})
        '''

        mean, std = self.forward(obs_features)

        dist = DiagGaussianAction(mean, std)
                                
        if deterministic_bool:
            # use mean without sampling 
            z = mean 
        else:
            # sample from gaussian distribution using reparameterization trick 
            z = dist.sample(reparam_trick_bool)

        # bounded action with tanh function 
        # clip action if we use method 1:  action = torch.clamp(action)  
        action = torch.tanh(z)

        #compute log_prob of action with change of variable 
        log_prob = dist.log_prob(z)
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim = -1)

        return action, log_prob  

    def get_log_prob(self, obs_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        ''' 
        Compute the log probability of a given action under the current policy. 

        '''
        mean , std = self.forward(obs_features)
        dist = DiagGaussianAction(mean, std)

        z = torch.atanh(action.clamp(-0.999,0.999))

        # change of variable to compute log_prob
        log_prob = dist.log_prob(z)
        log_prob = log_prob - torch.sum(torch.log(1 - action.pow(2) + 1e-6),dim = -1)

        return log_prob
        
    def get_entropy(self, obs_features: torch.Tensor) -> torch.Tensor:
        ''' 
        Compute the entropy of the policy distribution

        '''
        mean , std = self.forward(obs_features)
        
        dist = DiagGaussianAction(mean, std)
        entropy = dist.entropy()
            
        return entropy 

class ContinuousPolicyHead(nn.Module):
    def __init__(self, action_dim : int, 
                    feature_dim : int):
        super().__init__()
        
        self.action_dim = action_dim

        # mean output with action_space values 
        self.mean = nn.Linear(feature_dim, self.action_dim)

        self.log_std_layer = nn.Linear(feature_dim, action_dim)

    def forward(self, obs_features: torch.Tensor) -> tuple:

        # bounded mean 
        mean = self.mean(obs_features) 

        #standard deviation     
        log_std = torch.clamp(self.log_std_layer(obs_features),-5,2)

        # print('-------------------')
        # print(log_std)
        # print(mean)
        # print('--------------------')

        std = torch.exp(log_std)

        return mean, std 

    def sample_action(self, obs_features: torch.Tensor,
                    reparam_trick_bool: bool = False,
                    deterministic_bool : bool = False) -> tuple:
        mean, std = self.forward(obs_features)
        dist = DiagGaussianAction(mean, std)

        if deterministic_bool:
            action = mean
        else:
            action = dist.sample(reparam_trick_bool)

        log_prob = dist.log_prob(action)

        # clamp action vào [-1,1]
        action = torch.clamp(action, -1.0, 1.0)

        return action, log_prob
    
    def get_log_prob(self, obs_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        
        mean, std = self.forward(obs_features)
        dist = DiagGaussianAction(mean, std)

        return dist.log_prob(action)

    def get_entropy(self, obs_features: torch.Tensor) -> torch.Tensor:
        
        mean, std = self.forward(obs_features)

        dist = DiagGaussianAction(mean, std)

        return dist.entropy()

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

        dist = CategoricalAction(logits= logits)

        if deterministic_bool:
            action = dist.mode()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def get_log_prob(self,obs_features: torch.Tensor ,action: torch.Tensor):
        
        logits = self.forward(obs_features)
        dist = CategoricalAction(logits= logits)

        return dist.log_prob(action)

    def get_entropy(self, obs_features: torch.Tensor):
        
        logits = self.forward(obs_features)
        dist = CategoricalAction(logits=logits)
        entropy = dist.entropy()

        return entropy

class MultiDiscretePolicyHead(nn.Module):
    ''' 
    Args:
        action_dim (list) a collection of numbers of action value for each variable action 
            - example: action_dim = [a_1, a_2, a_3] , with a_i is the number of possible value for the first variable action  
    '''
    def __init__(self, action_dim: list, feature_dim : int):
        super().__init__()

        self.action_dim = action_dim 
        self.logits = nn.Linear(feature_dim, sum(action_dim))

    def forward(self, obs_features : torch.Tensor):
        return self.logits(obs_features)

    def sample_action(self,obs_features: torch.Tensor, 
                    deterministic_bool: bool = False):
        
        logits = self.forward(obs_features)
        dist = MultiCategoricalAction(logits=logits, nvec = self.action_dim)
        if deterministic_bool:
            action = dist.mode()
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)

        return action, log_prob
        
    def get_log_prob(self, obs_features : torch.Tensor, action: torch.Tensor):

        logits = self.forward(obs_features)
        dist = MultiCategoricalAction(logits=logits, nvec = self.action_dim)

        return dist.log_prob(action)
    
    def get_entropy(self, obs_features : torch.Tensor):
        
        logits = self.forward(obs_features)
        dist = MultiCategoricalAction(logits=logits, nvec = self.action_dim)
        return dist.entropy()

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

class ActorCriticPolicy(nn.Module):

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
            self.policy = ContinuousTanhPolicyHead(action_dim = action_dim, feature_dim= feature_dim)
        elif isinstance(action_space, spaces.MultiDiscrete):
            # MultiDiscrete action
            action_dim = action_space.nvec
            self.policy = MultiDiscretePolicyHead(action_dim = action_dim, 
                                            feature_dim = feature_dim)
        else:
            # MultiBinary action 
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