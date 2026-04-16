import torch
import torch.nn as nn
import torch.nn.functional as F 
import gymnasium as gym  
from .agent import OnPolicyAlgorithm  
from ..utils.feature_extractor import BaseFeatureExtractor

class A2C(OnPolicyAlgorithm): 

    def __init__(self,env: gym.Env, 
                num_envs: int,
                feature_network: str | type[BaseFeatureExtractor],
                feature_dim: int, 
                device: torch.device, 
                n_rollout_steps: int = 100, 
                type_vector: str = "Async", 
                learning_rate: float = 1e-5, 
                gamma: float = 0.99, 
                gae_lambda:float = 0.95,
                ent_coef: float = 0.5,
                vf_coef: float = 0.5,
                advantage_normalize: bool = False, 
                ):
        super().__init__(env, 
                        num_envs, 
                        feature_network, 
                        feature_dim, 
                        n_rollout_steps,
                        type_vector,
                        gamma,
                        gae_lambda,
                        device)
        
        self.learning_rate = learning_rate
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef 
        self.advantage_normalize = advantage_normalize

        self.optimizer = torch.optim.Adam(self.agent.parameters(),
                                        lr = learning_rate)

    def train(self) -> None:
        
        self.agent.train()

        for batch in self.rollout_buffer.batch_data(batch_size = 64): 
            obs = batch["obs"]
            action = batch["action"]
            advantage_value = batch["advantage"]
            return_value = batch["return"]

            if self.advantage_normalize:
                # normalize advantage value 
                advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.td() + 1e-8)

            # evaluation action
            log_prob , value, entropy = self.agent.evaluate_action(obs, action)

            policy_loss = -(advantage_value*log_prob).mean()

            # value loss 
            value_loss = F.mse_loss(value, return_value)

            # entropy loss 
            entropy_loss = -entropy.mean()

            # total loss 
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

            # optimization step 
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()

    def practive(self):
        pass 


if __name__ == '__main__':
    # test 
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = A2C(env = env,
                num_envs=4,
                feature_network='MLP',
                feature_dim=512,
                device= device,
                n_rollout_steps=50,
                type_vector='Sync',
                learning_rate= 1e-5,
                gamma = 0.99,
                gae_lambda = 0.95,
                )
    
    model.learn(total_timesteps = 100)