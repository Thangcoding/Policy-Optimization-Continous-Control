import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import gymnasium as gym 
from .share_model import ActorCritic
from ...env.vectorize_env import get_vec_env
from ..policy import OnPolicyAlgorithm
from ...utils.feature_extractor import BaseFeatureExtractor
from ...utils.seed import set_seed 


class SharedPPO(OnPolicyAlgorithm):

    def __init__(self,env: gym.Env, 
            num_envs: int,
            feature_network: str | type[BaseFeatureExtractor],
            feature_dim: int, 
            device: torch.device, 
            n_rollout_steps: int = 100, 
            type_vector: str = "Async", 
            max_step_eval: int = 1000,
            learning_rate: float = 1e-5, 
            gamma: float = 0.99,       
            gae_lambda:float = 0.95,   
            ent_coef: float = 0.01,    
            vf_coef: float = 0.5,     
            epsilon: float = 0.2,     
            clip_value: float = 0.2,  
            batch_size: int = 64, 
            seed: int = 64,                       
            use_wandb: bool = False,              
            observation_normalize: bool = True,     
            advantage_normalize: bool = False, 
            ):

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
        self.feature_network = feature_network
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate

        set_seed(seed)

        self.set_model()
    
    def set_model(self):
        
        self.agent = ActorCritic(feature_network = self.feature_network,
                                observation_space= self.vec_env.single_observation_space,
                                action_space= self.vec_env.single_action_space,
                                feature_dim= self.feature_dim).to(self.device)

        # optimizer 
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr = self.learning_rate)

    def predict(self, obs_features : torch.tensor, deterministic_bool: bool = False ): 

        action, log_prob, value = self.agent.predict(obs_features, deterministic_bool)

        return action, log_prob , value 
    
    def train(self): 

        self.agent.train()
        total_loss = 0 
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
            log_prob_new , value, entropy = self.agent.evaluate_action(obs, action)
        
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
            mean_advantage += advantage_value.mean().item()
            std_advantage += advantage_value.std().item()
            n_batches += 1

        # logger store 
        logs = {
                "loss": total_loss / n_batches,
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

    model = SharedPPO(env = env,
                num_envs=4,
                feature_network= 'MLP',
                feature_dim=128,
                device= device,
                n_rollout_steps= 256,
                type_vector='Sync',
                learning_rate= 1e-5,
                gamma = 0.99,
                gae_lambda = 0.95,
                use_wandb= False ,
                advantage_normalize=True, 
                observation_normalize= True, 
                )
    
    model.learn(timesteps= 1000, epochs= 5)








