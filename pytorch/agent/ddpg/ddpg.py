import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import gymnasium as gym 
from ...env.vectorize_env import get_vec_env
from ...utils.seed import set_seed
from .model import Actor , Critic
from ..policy import OffPolicyAlgorithm


class DDPG(OffPolicyAlgorithm):

    def __init__(self, env : gym.Env, 
                 num_envs: int, 
                 device: torch.device,
                 batch_size: int = 64, 
                 hidden : int = 256,
                 actor_lr:float = 1e-4,
                 critic_lr:float = 1e-3, 
                 buffer_size: int = 100000,
                 type_vector: str = 'Asyn',
                 max_step_eval: int = 1000, 
                 gamma: float = 0.99, 
                 tau: float = 0.005, 
                 warm_up_step: int = 3000, 
                 seed: int = 64, 
                 observation_normalize: bool = False, 
                 use_wandb: bool = False, 
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
                         device)

        self.tau = tau 
        self.gamma = gamma 
        self.env = env 
        self.num_envs = num_envs
        self.max_step_eval = max_step_eval
        self.batch_size = batch_size
        self.hidden = hidden 

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr 
        self.buffer_size = buffer_size
        self.type_vector = type_vector

        set_seed(seed)
        self.set_model()

    def set_model(self):
        obs_dim = self.vec_env.single_observation_space.shape[0]
        action_dim = self.vec_env.single_action_space.shape[0]

        self.actor = Actor(obs_dim = obs_dim,
                           action_dim= action_dim,
                           hidden= self.hidden).to(self.device)
        
        self.critic = Critic(obs_dim= obs_dim, 
                             action_dim= action_dim,
                             hidden= self.hidden).to(self.device)
        
        # target actor critic 
        self.target_actor = Actor(obs_dim= obs_dim, 
                                  action_dim= action_dim,
                                  hidden= self.hidden).to(self.device)

        self.target_critic = Critic(obs_dim= obs_dim,
                                    action_dim= action_dim,
                                    hidden= self.hidden).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.critic_lr)

    def select_action(self,obs: torch.tensor,
                        step: int, 
                        noise_std: float = 0.1,
                        deterministic: bool = False,
                        warm_up : bool = False):
        
        if  warm_up and step < self.warm_up_step and not deterministic:
            action = np.array([self.vec_env.single_action_space.sample() for _ in range(self.num_envs)])
            return action
        
        with torch.no_grad(): 
            action = self.actor(obs).cpu().numpy()
    
        # exploration noise 
        if not deterministic:
            noise= np.random.normal(0, noise_std, size = action.shape)

            action = action + noise
    
        return np.clip(action,-1,1)

    def train(self, step ):
        
        self.actor.train()
        self.critic.train()

        sample = self.replay_buffer.sample(batch_size=self.batch_size)
        obs, action, reward, next_obs, done = sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']

        #=================================================
        # critic update TD 
        # (r + gamma*Q_target(s', mu(s')) - Q_curr(s, mu(s)))^2 
        #=================================================

        with torch.no_grad(): 
            next_action = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_action)
            y = reward + self.gamma*target_q*(1 - done) 

        curr_q = self.critic(obs, action)
        critic_loss = F.mse_loss(y, curr_q)

        # optimize critic 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        #=================================================
        # actor update                                    
        #=================================================
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
        self.actor_optimizer.step()

        #  update target network 
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau)*target_param.data
            )

    def eval(self, render=False, stats_observation = None):
        eval_env = get_vec_env(env= self.env,
                            num_envs= 1,
                            type_vector= "Sync",
                            observation_normalize= False,
                            render = render,
                            stats_observation= stats_observation)
        
        eval_env.training_mode = False 

        frames = []
        obs, _ = eval_env.reset(seed = self.seed)

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
                obs = obs_tensor,
                step = i,
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

    model = DDPG(env=env,
                num_envs=4,
                device= device,
                actor_lr= 3e-4,
                critic_lr= 1e-4,
                buffer_size=100000,
                type_vector="Asyn",
                warm_up_step= 10, 
                max_step_eval=1000,
                observation_normalize= True
            )

    model.learn(episodes = 1, timesteps= 30)