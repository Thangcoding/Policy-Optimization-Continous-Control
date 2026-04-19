import torch 
from torch.distributions import Normal , Categorical, Bernoulli

class ActionDistribution:

    def sample(self):
        raise NotImplementedError
    
    def log_prob(self):
        raise NotImplementedError 
    
    def entropy(self):
        raise NotImplementedError
    
class CategoricalAction(ActionDistribution):
    '''
    used for discrete action (spaces.Discrete type)
    '''

    def __init__(self,logits: torch.tensor):
        self.logits = logits
        self.dist =  Categorical(logits = logits)

    def sample(self):
        return self.dist.sample()

    def log_prob(self,action: torch.tensor):
        log_prob = self.dist.log_prob(action)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim = -1) 
        return log_prob
    
    def entropy(self):
        entropy = self.dist.entropy()

        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim = -1) 
        return entropy


class DiagGaussianAction(ActionDistribution):

    ''' 
    used for continuous action (spaces.box type)

    '''
    
    def __init__(self, mean : torch.tensor , std : torch.tensor):
        
        self.mean = mean 
        self.std = std 

        self.dist = Normal(mean, std)

    def sample(self, reparam_trick_bool : bool = False):

        return self.dist.rsample() if reparam_trick_bool else self.dist.sample()

    def log_prob(self,action: torch.tensor):
        log_prob = self.dist.log_prob(action)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim = -1) 
        return log_prob

    def entropy(self):
        entropy = self.dist.entropy()

        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim = -1) 
        return entropy

class BernoulliAction(ActionDistribution):
    ''' 
    used for discrete action (spaces.MultiBinary type)
    '''
    def __init__(self, logits: torch.Tensor):
        self.dist = Bernoulli(logits = logits)

    def sample(self):
        return self.dist.sample()

    def log_prob(self,action: torch.Tensor):
        log_prob = self.dist.log_prob(action)
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim = -1) 
        return log_prob

    def entropy(self):
        entropy = self.dist.entropy()

        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim = -1) 
        return entropy

class MultiCategoricalAction(ActionDistribution):
    ''' 
    used for discrete action (spaces.MultiDiscrete)

    Args:
        logits : 
        nvec : 
    '''
    def __init__(self, logits : torch.tensor , nvec : int):
        self.logits = logits 
        self.nvec = nvec 
        
        split_logits = torch.split(logits, nvec.tolit(), dim = -1)
        self.dist = [Categorical(logits = l) for l in split_logits]

    def sample(self):
        return torch.stack([d.sample() for d in self.dist], dim = -1)

    def log_prob(self, action: torch.tensor):
        return torch.stack([d.log_prob(a) for d, a in zip(self.dist,action.T)], dim = -1).sum(-1)

    def entropy(self):
        return torch.stack([d.entropy() for d in self.dist], dim = -1).sum(-1)

