import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import gymnasium as gym 
from .agent import OnPolicyAlgorithm
from ..utils.feature_extractor import BaseFeatureExtractor
from ..utils.seed import set_seed 



class PPO(OnPolicyAlgorithm):

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
                epsilon: float = 0.001, 
                batch_size: int = 64,
                seed: int = 64, 
                use_wandb: bool = False,     
                advantage_normalize: bool = False, 
                ):

            super().__init__(env, 
                num_envs, 
                feature_network, 
                feature_dim,
                learning_rate,  
                n_rollout_steps,
                type_vector,
                gamma,
                gae_lambda, 
                use_wandb,
                seed, 
                device)
            
            self.ent_coef = ent_coef
            self.vf_coef = vf_coef 
            self.epsilon = epsilon
            self.advantage_normalize = advantage_normalize
            self.batch_size = batch_size
            
            set_seed(seed)
        
        def train(self): 
            
            self.agent.train()
            total_loss = 0 
            total_policy_loss = 0 
            total_value_loss = 0 
            total_entropy = 0 
            total_return = 0 
            n_batches = 0 
            mean_advantage , std_advantage = 0, 0 

            for batch in self.rollout_buffer.batch_data(batch_size= self.batch_size):
                obs = batch['obs']                    
                action = batch["action"]              
                advantage_value = batch['advantage']   
                return_value = batch['return']        
                log_prob_old = batch['log_prob']

                if self.advantage_normalize:
                    # normalize advantage value 
                    advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e-8)
                
                # evaluation action 
                log_prob_new , value, entropy = self.agent.evaluate_action(obs, action)
                
                # surrogate objective 
                ratio = torch.exp(log_prob_new - log_prob_old)
                surr1 = ratio * advantage_value
                surr2 = torch.clamp(ratio, 1 - self.epsilon , 1 + self.epsilon)* advantage_value

                # policy loss
                policy_loss =  torch.mean(torch.min(surr1, surr2))

                # critic loss 
                value_loss = F.mse_loss(value, return_value)

                # entropy loss 
                entropy_mean = entropy.mean()
            
                # total loss 
                loss =  self.vf_coef * value_loss - self.ent_coef * entropy_mean - policy_loss

                # optimization step     
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()

                # accumulate 
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_return += return_value.mean().item()
                mean_advantage += advantage_value.mean().item()
                std_advantage += advantage_value.std().item()
                n_batches += 1

            # logger store 
            logs = {
                    "loss": total_loss / n_batches,
                    "policy_loss": total_policy_loss / n_batches,
                    "value_loss": total_value_loss / n_batches,
                    "entropy": total_entropy / n_batches,
                    "avg_return": total_return / n_batches,
                    "adv_mean": mean_advantage / n_batches,
                    "adv_std": std_advantage / n_batches,
                }
            return logs             


if __name__ == '__main__':
    # test 
    env = gym.make("Hopper-v4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(env = env,
                num_envs=4,
                feature_network= 'MLP',
                feature_dim=128,
                device= device,
                n_rollout_steps=50,
                type_vector='Sync',
                learning_rate= 1e-5,
                gamma = 0.99,
                gae_lambda = 0.95,
                use_wandb=False
                )
    
    model.learn(total_timesteps = 300, n_epochs= 2)
                
        