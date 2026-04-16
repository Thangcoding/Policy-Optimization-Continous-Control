import torch
import numpy as np
import gymnasium as gym 
from ..env.vectorize_env import VectorEnv
from ..env.rollout_buffer import RolloutBuffer 
from ..utils.network import ActorCriticPolicy 
from ..utils.feature_extractor import BaseFeatureExtractor , FeatureExtractorMLP, FeatureExtractorCNN


class OnPolicyAlgorithm:
    def __init__(self,env : gym.Env,
                num_envs: int,
                feature_network: str | type[BaseFeatureExtractor],
                feature_dim: int,  
                n_rollout_steps: int, 
                type_vector: str,  
                gamma: float, 
                gae_lambda: float, 
                device: torch.device, 
                ):
        
        self.n_rollout_steps = n_rollout_steps

        self.n_steps = n_rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device
    
        # setup vector env 
        self.vec_env = VectorEnv(env,num_envs,type_vector)

        # setup_model 
        self.agent = ActorCriticPolicy(feature_network, self.vec_env.observation_space, self.vec_env.action_space, feature_dim).to(device)

        # setup rollout 
        self.rollout_buffer = RolloutBuffer(buffer_size = n_rollout_steps, 
                                            num_envs = num_envs, 
                                            observation_space = self.vec_env.observation_space, 
                                            action_space = self.vec_env.action_space, 
                                            device = device)

    def collect_rollouts(self):
        self.rollout_buffer.reset()
        
        obs , _ = self.vec_env.reset()

        for _ in range(self.n_rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                action, log_prob , value = self.agent.predict(obs_tensor)

            action_np = action.cpu().numpy()

            next_obs, reward, terminated, truncated, _ = self.vec_env.step(action_np)

            done = (terminated | truncated).astype(np.float32) 

            self.rollout_buffer.add(
                obs,
                action_np,
                reward,
                done,
                value.cpu().numpy(),
                log_prob.cpu().numpy(),
            )

            obs = next_obs

        # bootstrap value  
        obs_tensor = torch.tensor(obs, dtype = torch.float32).to(self.device)
        with torch.no_grad(): 
            _, _, last_value = self.agent.predict(obs_tensor)
            last_value = last_value.cpu().numpy()  

        self.rollout_buffer.gae_and_return_value(
            last_value, self.gamma, self.gae_lambda
        )

    def train(self):
        raise NotImplementedError

    def learn(self, total_timesteps):
        timesteps = 0

        while timesteps < total_timesteps:
            self.collect_rollouts()

            self.train()

            timesteps += self.n_steps
    
    def save(self,path):
        torch.save({"feature_network": self.network.state_dict(),
                   "policy": self.policy.state_dict(), 
                   "critic": self.critic.state_dict(),}, path)

    def load(self, path):
        checkpoint = torch.load(path)

class OffPolicyAlgorithm:

    def __init__(self):
        pass 




