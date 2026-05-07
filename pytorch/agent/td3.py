import numpy as np 
import torch 
import torch.nn.functional as F 
import gymnasium as gym 
from ..utils.seed import set_seed
from ..utils.feature_extractor import BaseFeatureExtractor
from ..model.deterministic_model import Actor , Critic
from .agent import OffPolicyAlgorithm


class TD3(OffPolicyAlgorithm):

    def __init__(self,env: gym.Env, 
                 num_envs: int, 
                 device: torch.device,  
                 batch_size: int = 64,  
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3, 
                 buffer_size: int = 100000, 
                 policy_delay: int = 2, 
                 target_policy_noise: float = 0.2, 
                 target_noise_clip: float = 0.5,
                 type_vector: str = 'Sync', 
                 max_step_eval: int = 1000,
                 gamma: float = 0.99, 
                 tau: float = 0.005, 
                 warm_up_step: int = 300, 
                 seed: int = 64, 
                 use_wandb: bool = False,
                 ): 
        super().__init__(env, 
                         num_envs,
                         buffer_size,
                         type_vector,
                         gamma,
                         use_wandb,
                         warm_up_step,
                         seed,
                         device)
        self.tau = tau 
        self.gamma = gamma 
        self.env = env 
        self.num_envs = num_envs
        self.max_step_eval = max_step_eval
        self.batch_size = batch_size
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr 
        self.buffer_size = buffer_size 
        self.type_vector = type_vector

        self.policy_delay = policy_delay 
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        self.set_model()
    
    def set_model(self):
        obs_dim = self.vec_env.observation_space.shape[0]
        action_dim = self.vec_env.action_space.shape[0]

        self.actor = Actor(obs_dim= obs_dim,
                           action_dim= action_dim).to(self.device)

        self.critic_1 = Critic(obs_dim= obs_dim,
                               action_dim= action_dim).to(self.device)

        self.critic_2 = Critic(obs_dim= obs_dim,
                               action_dim= action_dim).to(self.device)

        # target network 
        self.actor_target = Actor(obs_dim= obs_dim,
                                  action_dim=action_dim).to(self.device)

        self.critic_target_1 = Critic(obs_dim=obs_dim,
                                      action_dim=action_dim)

        self.critic_target_2 = Critic(obs_dim=obs_dim,
                                      action_dim=action_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # optimizer 
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = self.critic_lr)

    def select_action(self, obs: torch.tensor, 
                      noise_std: float = 0.1,
                      deterministic:bool = False):
        obs = obs.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]

        # exploration noise 
        if not deterministic:
            action += np.random.normal(0, noise_std, size = action.shape)

        return np.clip(action, -1, 1)

    def train(self, step_update ):
        sample = self.replay_buffer.sample(batch_size=self.batch_size)
        obs, action, reward, next_obs, done = sample['obs'], sample['action'], sample['reward'], sample['next_obs'], sample['done']

        #=============================================== 
        # critic update TD                               
        # a = mu_target(s') + epsilon                    
        # y = r + gamma * min_{i = 1,2} Q_target_i(s',a) 
        #=============================================== 

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            noise = torch.normal(0, std  = self.target_policy_noise, size = next_action.shape)
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)

            next_action_noise = torch.clamp(next_action + noise, -1 ,1)
            value_1 , value_2 = self.critic_1(next_obs, next_action_noise) , self.critic_2(next_obs, next_action_noise)
            y =  reward + self.gamma *  torch.min(value_1, value_2)*(1 - done)

        curr_q_1 = self.critic_1(obs, action)
        curr_q_2 = self.critic_2(obs,action)

        critic_loss_1 = F.mse_loss(y, curr_q_1)
        critic_loss_2 = F.mse_loss(y, curr_q_2)

        critic_loss = critic_loss_1 + critic_loss_2

        # optimize critic 
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #======================================
        # Delayed actor update 
        #======================================
        actor_loss = -self.critic_1(obs, self.actor(obs)).mean()
        if step_update % self.policy_delay == 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target network 
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1,self.critic_target_1)
            self.soft_update(self.critic_2,self.critic_target_2)
        
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau)*target_param.data
            )

    def eval(self, render=False):

        if render:
            eval_env = gym.make(self.env.spec.id, render_mode="rgb_array")
        else:
            eval_env = gym.make(self.env.spec.id)

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
    env = gym.make("Hopper-v4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TD3(env = env,
                num_envs=4, 
                device= device,
                actor_lr=3e-4,
                critic_lr= 1e-4,
                buffer_size=100000,
                type_vector="Asyc",
                max_step_eval=1000)
    
    model.learn(episodes=10)