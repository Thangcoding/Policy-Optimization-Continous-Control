import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import gymnasium as gym 
from .model import Actor, Critic
from ...env.vectorize_env import get_vec_env
from ..policy import OnPolicyAlgorithm
from ...utils.feature_extractor import BaseFeatureExtractor
from ...utils.seed import set_seed 

    
class PPO(OnPolicyAlgorithm):

    def __init__(self,env: gym.Env, 
                num_envs: int,
                feature_dim: int, 
                device: torch.device, 
                n_rollout_steps: int = 100, 
                type_vector: str = "Async", 
                max_step_eval: int = 1000,
                actor_lr: float = 3e-4, 
                critic_lr: float = 1e-5,
                gamma: float = 0.99,       
                gae_lambda:float = 0.95,   
                ent_coef: float = 0.5,    
                vf_coef: float = 0.5,     
                epsilon: float = 0.2,     
                clip_value: float = 0.2,  
                batch_size: int = 64, 
                seed: int = 64,                       
                use_wandb: bool = False,              
                observation_normalize: bool = True,     
                advantage_normalize: bool = False):
        super().__init__(env, 
                         num_envs, 
                         n_rollout_steps, 
                         type_vector,
                         observation_normalize, 
                         gamma, 
                         gae_lambda, 
                         use_wandb, 
                         seed, 
                         device)
        
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef 
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.advantage_normalize = advantage_normalize
        self.batch_size = batch_size
        self.max_step_eval = max_step_eval
        self.feature_dim = feature_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        set_seed(seed)

        self.set_model()

    def set_model(self):
        obs_dim = self.vec_env.single_observation_space.shape[0]
        action_dim = self.vec_env.single_action_space.shape[0]
        max_action = self.vec_env.single_action_space.high 
        min_action = self.vec_env.single_action_space.low 

        self.actor = Actor(obs_dim= obs_dim, 
                           action_dim= action_dim,
                           hidden= self.feature_dim, 
                           min_action= min_action,
                           max_action= max_action).to(self.device)
        
        self.critic = Critic(obs_dim=obs_dim,
                             action_dim= action_dim,
                             hidden= self.feature_dim).to(self.device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.critic_lr)
    
    def predict(self, obs_features: torch.tensor , deterministic_bool: bool = False):
        
        action, log_prob = self.actor.sample_action(obs_features, deterministic_bool)

        value = self.critic(obs_features)
    
        return action, log_prob, value.squeeze(-1)
    
    def train(self):
        
        self.actor.train()
        self.critic.train()

        total_policy_loss = 0 
        total_value_loss = 0 
        total_entropy = 0 
        n_batches = 0                          
        mean_advantage , std_advantage = 0, 0   

        for batch in self.rollout_buffer.batch_data(batch_size= self.batch_size):
            obs = batch['obs']                    
            action = batch["action"]              
            advantage_value = batch['advantage']  
            return_value = batch['return']        
            log_prob_old = batch['log_prob']      
            value_old = batch['value']             

            if self.advantage_normalize:
                # normalize advantage value 
                advantage_value = (advantage_value - advantage_value.mean()) / torch.clamp(advantage_value.std(), min = 1e-6)
            
            # evaluation action 
            log_prob_new = self.actor.get_log_prob(obs, action)
            entropy = self.actor.get_entropy(obs)

            value = self.critic(obs).squeeze(-1)

            # surrogate objective 
            ratio = torch.exp(log_prob_new - log_prob_old)
            surr1 = ratio * advantage_value
            surr2 = torch.clamp(ratio, 1 - self.epsilon , 1 + self.epsilon)* advantage_value
            
            # policy loss
            policy_loss =  torch.mean(torch.min(surr1, surr2))
    
            # critic loss with value clip                                                       
            value_pred_clip = value_old + torch.clamp(value - value_old,-self.clip_value , self.clip_value)  
            value_loss_1 = (value - return_value)**2                                                         
            value_loss_2 = (value_pred_clip - return_value)**2               

            value_loss = torch.mean(torch.max(value_loss_1, value_loss_2)) 

            # actor loss 
            entropy_mean = entropy.mean()

            actor_loss = - self.ent_coef * entropy_mean - policy_loss
            
            # actor optimization step     
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # critic loss 
            critic_loss = value_loss 
            
            # actor optimization step     
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # accumulate 
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            mean_advantage += advantage_value.mean().item()
            std_advantage += advantage_value.std().item()
            n_batches += 1

        # logger store 
        logs = {
                "policy_loss": total_policy_loss / n_batches,
                "value_loss": total_value_loss / n_batches,
                "entropy": total_entropy / n_batches,
                "adv_mean": mean_advantage / n_batches,
                "adv_std": std_advantage / n_batches,
            }

        return logs    
    
    def eval(self, render = False, stats_observation = None ):
        
        eval_env = get_vec_env(env= self.env,
                                num_envs= 1,
                                type_vector= "Sync",
                                observation_normalize= False,
                                render = render,
                                stats_observation= stats_observation,
                                seed= self.seed)
        
        eval_env.training_mode = False 
        
        return_val = 0            
        frames = []               
        obs, _ = eval_env.reset(seed = self.seed) 

        return_val = 0.0 

        for i in range(self.max_step_eval):
            
            if render:
                frame = eval_env.render()
                frames.append(frame)
            
            obs_tensor = torch.as_tensor(
                obs, 
                dtype = torch.float32, 
                device = self.device
            )
            
            with torch.no_grad():
                action, _, _ = self.predict(obs_tensor, deterministic_bool = True)
            
            obs, reward , terminated, truncated, _ = eval_env.step(action.cpu().numpy())

            return_val += reward 

            if terminated or truncated:
                break 
        
        eval_env.close()
        return frames , return_val 
    

if __name__ == '__main__':
    # test 
    env = gym.make("Hopper-v5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(env = env,
                num_envs=4,
                feature_dim=64,
                device= device,
                n_rollout_steps=50,
                type_vector='Sync',
                actor_lr= 3e-4,
                critic_lr= 1e-5,
                gamma = 0.99,
                gae_lambda = 0.95,
                use_wandb= False,
                advantage_normalize=True, 
                observation_normalize= True,
                )
    
    model.learn(episodes= 40, epochs= 10)