import numpy as np 
import torch 
import torch.nn.functional as F 
import gymnasium as gym 
from ..utils.seed import set_seed
from ..utils.feature_extractor import BaseFeatureExtractor
from ..model.deterministic_model import Actor , Critic
from .agent import OffPolicyAlgorithm

class SAC(OffPolicyAlgorithm):

    def __init__(self):
        pass 

    def set_model(self):
        pass 

    def train(self):
        pass 

    def eval(self):
        pass 
