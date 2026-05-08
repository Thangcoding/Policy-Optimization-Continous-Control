import numpy as np 
import torch 
import torch.nn.functional as F 
import gymnasium as gym 
from ..env.vectorize_env import get_vec_env
from ..utils.seed import set_seed
from ..utils.feature_extractor import BaseFeatureExtractor
from ..model.sac_model import Actor , Critic
from .agent import OffPolicyAlgorithm

class SAC(OffPolicyAlgorithm):

    def __init__(self,env : gym.Env, 
                 num_envs: int,
                 device: torch.device,
                 num_critics: int = 2,
                 batch_size: int = 64, 
                 actor_lr : float = 1e-4, 
                 critic_lr : float = 1e-3, 
                 alpha: float = 0.2, 
                 alpha_lr: float = 1e-3, 
                 buffer_size: int = 100000,
                 type_vector: str = 'Sync', 
                 max_step_eval: int = 1000,
                 tau: float = 0.005,  
                 gamma: float = 0.99, 
                 warm_up_step: int = 300, 
                 seed: int = 64, 
                 observation_normalize: bool = False, 
                 auto_entropy: bool = True, 
                 use_wandb: bool = False
                 ):
        super().__init__(env, 
                         num_envs, 
                         buffer_size, 
                         type_vector,
                         gamma,
                         use_wandb,
                         warm_up_step,
                         observation_normalize, 
                         seed, 
                         device
                         )

        self.num_critics = num_critics 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr 
        self.alpha_lr = alpha_lr 
        self.batch_size = batch_size 
        self.alpha = alpha 
        self.tau = tau 
        self.max_step_eval = max_step_eval

        self.auto_entropy = auto_entropy

        self.set_model()

    def set_model(self):
        obs_dim = self.vec_env.single_observation_space.shape[0]
        action_dim = self.vec_env.single_action_space.shape[0]
        self.target_entropy = -action_dim

        self.actor = Actor(obs_dim=obs_dim,
                           action_dim= action_dim).to(self.device)
        
        self.lst_critic = []
        self.lst_critic_target = []

        for _ in range(self.num_critics):
            critic = Critic(obs_dim=obs_dim, action_dim= action_dim).to(self.device)
            critic_target = Critic(obs_dim=obs_dim, action_dim= action_dim).to(self.device)

            critic_target.load_state_dict(critic.state_dict())
            self.lst_critic.append(critic)
            self.lst_critic_target.append(critic_target)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)

        self.lst_critic_optimizer = []
        for i in range(self.num_critics):
            critic_optimizer = torch.optim.Adam(self.lst_critic[i].parameters(), lr = self.critic_lr)
            self.lst_critic_optimizer.append(critic_optimizer)
        
        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad = True, device = self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = self.alpha_lr)

    def select_action(self, obs: torch.tensor, deterministic: bool = False):
        with torch.no_grad():
            action,_ = self.actor.sample(obs, deterministic)
        return action.cpu().numpy()
        
    def train(self, step):
        sample = self.replay_buffer.sample(batch_size= self.batch_size)

        obs , action, reward , next_obs, done = sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']

        #================================
        # critic update 
        #================================ 

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)

            q_next_list = [critic_target(next_obs, next_action) for critic_target in self.lst_critic_target]

            min_q_next =  torch.min(torch.stack(q_next_list),dim = 0)[0]
            
            y = reward + self.gamma*(min_q_next - self.alpha*next_log_prob)*(1 - done)

        q_curr_list = [critic(obs, action) for critic in self.lst_critic]
        
        critic_loss = 0
        for q in q_curr_list:
            critic_loss += F.mse_loss(q, y)

        critic_loss /= self.num_critics

        for opt in self.lst_critic_optimizer:
            opt.zero_grad()
        
        critic_loss.backward()

        for opt in self.lst_critic_optimizer:
            opt.step()
    
        #===================
        # Actor Update 
        #===================

        new_action ,  log_prob = self.actor.sample(obs)

        q_new_list = [critic(obs, new_action) for critic in self.lst_critic]

        min_q_new = torch.min(torch.stack(q_new_list), dim = 0)[0]
        alpha = self.log_alpha.exp() if self.auto_entropy else self.alpha

        actor_loss = (alpha * log_prob -min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #=====================
        # Alpha Update
        #=====================

        if self.auto_entropy:
            alpha_loss = - (self.log_alpha *(log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for i in range(self.num_critics):
            self.soft_update(self.lst_critic[i], self.lst_critic_target[i])
        
        return critic_loss.item() , actor_loss.item()
        
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau)*target_param.data
        )
    
    def eval(self, render = False, stats_observation = None ):
            
        eval_env = get_vec_env(env= self.env,
                            num_envs= 1,
                            type_vector= "Sync",
                            observation_normalize= False,
                            render = render,
                            stats_observation= stats_observation)
        
        eval_env.training_mode = False 

        frames = []
        obs, _ = eval_env.reset()

        return_val = 0.0

        for i in range(self.max_step_eval):

            if render:
                frame = eval_env.render()
                frames.append(frame)

            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=self.device
            )

            action = self.select_action(
                obs_tensor,
                deterministic=True
            )

            obs, reward, terminated, truncated, _ = eval_env.step(action)

            return_val += reward
            # hoặc gamma**i * reward

            if terminated or truncated:
                break

        eval_env.close()

        return frames, return_val

if __name__ == '__main__':
    env = gym.make("Hopper-v5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SAC(env = env, 
                num_envs = 4 ,
                device = device ,
                num_critics = 2,
                batch_size = 64, 
                actor_lr = 1e-4, 
                critic_lr = 1e-3, 
                alpha = 0.2, 
                buffer_size = 100000,
                type_vector = 'Sync')
    
    model.learn(episodes = 10)

