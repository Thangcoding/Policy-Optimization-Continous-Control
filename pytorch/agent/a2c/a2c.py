import torch
import torch.nn as nn
import torch.nn.functional as F 
import gymnasium as gym  
from ...env.vectorize_env import get_vec_env
from .model import ActorCritic
from ..policy import OnPolicyAlgorithm  
from ...utils.feature_extractor import BaseFeatureExtractor
from ...utils.seed import set_seed

class A2C(OnPolicyAlgorithm): 

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
                ent_coef: float = 0.0,
                vf_coef: float = 0.5,
                batch_size: int = 64,
                seed: int = 64, 
                use_wandb: bool = False,     
                observation_normalize: bool = False, 
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
        self.observation_normalize = observation_normalize
        self.advantage_normalize = advantage_normalize
        self.max_step_eval = max_step_eval
        self.batch_size = batch_size
        self.feature_network = feature_network
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        self.set_model()
        set_seed(seed)
    
    def set_model(self):
        # setup_model 

        self.agent = ActorCritic(feature_network = self.feature_network,
                                observation_space= self.vec_env.single_observation_space,
                                action_space = self.vec_env.single_action_space,
                                feature_dim= self.feature_dim).to(self.device)
        
        # optimizer 
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr = self.learning_rate)
    def predict(self, obs_features : torch.tensor, deterministic_bool: bool = False): 

        action, log_prob, value = self.agent.predict(obs_features, deterministic_bool)

        return action, log_prob , value 
    
    def train(self) -> None:

        self.agent.train()
        total_policy_loss = 0 
        total_value_loss = 0 
        total_entropy = 0 
        mean_advantage , std_advantage = 0, 0 
        n_batches = 0
        for batch in self.rollout_buffer.batch_data(batch_size = self.batch_size): 
            obs = batch["obs"]
            action = batch["action"]
            advantage_value = batch["advantage"]
            return_value = batch["return"]

            if self.advantage_normalize:
                # normalize advantage value 
                advantage_value = (advantage_value - advantage_value.mean()) / (advantage_value.std() + 1e-8)

            # evaluation action
            log_prob , value, entropy = self.agent.evaluate_action(obs, action)

            policy_loss = - (advantage_value.detach()*log_prob).mean()

            # value loss 
            value_loss = F.mse_loss(value, return_value)
            
            # entropy mean 
            entropy_mean = -entropy.mean()

            # total loss 
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_mean 

            # optimization step     
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()
            
            # accumulate 
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += -entropy_mean.item()
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

    def eval(self, render = False, stats_observation = None):
        eval_env = get_vec_env(env= self.env,
                        num_envs= 1,
                        type_vector= "Sync",
                        observation_normalize= False,
                        render = render,
                        stats_observation= stats_observation,
                        seed = self.seed)
        
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
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class feature_extract(BaseFeatureExtractor):

        def __init__(self, observation_space: gym.Space, feature_dim: int, **kwargs):
            super().__init__() 
            self.observation_dim = observation_space.shape[0]
            self.feature_dim = feature_dim 

            self.net = nn.Sequential(nn.Linear(self.observation_dim, 32),
                                     nn.ReLU(),
                                     nn.Linear(32,64),
                                     nn.ReLU(),
                                     nn.Linear(64, feature_dim),
                                     )
            
        def forward(self, obs: torch.tensor):
            return self.net(obs)
        
    net = feature_extract(observation_space= env.observation_space, feature_dim= 64)

    model = A2C(env = env,
                num_envs=4,
                feature_network=net,
                feature_dim=64,
                device= device,
                batch_size= 64,
                n_rollout_steps= 50,
                type_vector='Sync',
                learning_rate= 5e-5,
                gamma = 0.99,
                gae_lambda = 0.95,
                advantage_normalize= True, 
                observation_normalize= True, 
                use_wandb=False
                )
    
    model.learn(timesteps = 1000, epochs =5)