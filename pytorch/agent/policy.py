import torch
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from ..env.vectorize_env import  get_vec_env
from ..env.rollout_buffer import RolloutBuffer 
from ..env.replay_buffer import ReplayBuffer   
from ..utils.logger import Logger 

class OnPolicyAlgorithm:
    def __init__(self,env : gym.Env,
                num_envs: int,
                n_rollout_steps: int, 
                type_vector: str,  
                observation_normalize: bool,
                gamma: float, 
                gae_lambda: float, 
                use_wandb: bool, 
                seed: int,
                device: torch.device, 
                ):

        self.n_rollout_steps = n_rollout_steps
        self.global_steps = 0 
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.observation_normalize = observation_normalize
        self.seed = seed 
        self.env = env 

        self.device = device
        self.logger = Logger(use_wandb= use_wandb)

        # setup vector env 
        self.vec_env = get_vec_env(env = env,
                                    num_envs = num_envs,
                                    type_vector = type_vector,
                                    observation_normalize = observation_normalize,
                                    seed= seed)
        
        # setup rollout 
        self.rollout_buffer = RolloutBuffer(buffer_size = n_rollout_steps, 
                                            num_envs = num_envs, 
                                            observation_space = self.vec_env.single_observation_space, 
                                            action_space = self.vec_env.single_action_space, 
                                            device = device)
        self.step = 0 
    
    def set_model(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError

    def collect_rollouts(self):
        self.rollout_buffer.reset()

        self.vec_env.training_mode = True 
        
        obs , _ = self.vec_env.reset(seed = [self.seed + i for i in range(self.num_envs)])

        for _ in range(self.n_rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                action, log_prob , value = self.predict(obs_tensor)

            action_np = action.cpu().numpy()

            next_obs, reward, terminated, _, _ = self.vec_env.step(action_np)

            done = terminated.astype(np.float32) 

            self.rollout_buffer.add(
                obs = obs,
                action = action_np,
                reward= reward,
                value = value.cpu().numpy(),
                log_prob = log_prob.cpu().numpy(),
                done = done
            )

            obs = next_obs

        # bootstrap value  
        obs_tensor = torch.tensor(obs, dtype = torch.float32).to(self.device)
        with torch.no_grad(): 
            _, _, last_value = self.predict(obs_tensor)
            last_value = last_value.cpu().numpy()

        # mask done 
        last_value = last_value *(1 - done)
        self.rollout_buffer.gae_and_return_value(
            last_value, self.gamma, self.gae_lambda
        )

    def train(self):
        raise NotImplementedError

    def learn(self,timesteps = 1000000 , epochs: int = 1):
        render_ratio = max(int(timesteps/1000),1)
        
        while self.step < timesteps:
            self.collect_rollouts()
            self.step += self.n_rollout_steps

            policy_loss , value_loss , entropy , avd_mean, avd_std = 0, 0, 0, 0, 0

            # train
            for _ in range(epochs):
                batch_logs = self.train()

                policy_loss += batch_logs['policy_loss']
                value_loss += batch_logs['value_loss']
                entropy += batch_logs['entropy']
                avd_mean += batch_logs['adv_mean']
                avd_std += batch_logs['adv_std']      

            # evaluation 
            if self.observation_normalize:
                stats_observation = {'mean': self.vec_env.mean, 
                                    'var': self.vec_env.var, 
                                    'count': self.vec_env.count}
            else:
                stats_observation = None 
            
            total_return = 0 
            for _ in range(10):
                _, return_val = self.eval(stats_observation = stats_observation)
                total_return += return_val 

            self.logger.set_step(self.step)
            if (self.step % render_ratio == 0 ) and self.logger.use_wandb:
                frames, _  = self.eval(render = True, stats_observation = stats_observation)
                self.logger.log_video(frames)

            # log store 
            logs = {"policy_loss": policy_loss / epochs,
                    "value_loss": value_loss / epochs, 
                    "entropy": entropy / epochs,
                    "eval_return": total_return.item() / 10,
                    "avd_mean": avd_mean / epochs,
                    "avd_std": avd_std / epochs}

            self.logger.log(logs)

    def save(self, path):
        torch.save({
            "model": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.agent.load_state_dict(checkpoint["model"])

        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])

class OffPolicyAlgorithm:

    def __init__(self, env: gym.Env,
                num_envs: int,
                buffer_size: int,
                batch_size: int, 
                type_vector: str,
                gamma: float,
                use_wandb: bool,
                warm_up_step: int,
                observation_normalize: bool, 
                seed: int,
                device: torch.device,
                ):
        self.env = env 
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.type_vector = type_vector
        self.gamma = gamma 
        self.seed = seed
        self.use_wandb = use_wandb
        self.warm_up_step = warm_up_step
        self.observation_normalize = observation_normalize
        self.batch_size = batch_size

        self.device = device 
        self.logger = Logger(use_wandb= use_wandb)   

        # setup vector env 
        self.vec_env = get_vec_env(env= env,
                                   num_envs= num_envs,
                                   type_vector= type_vector,
                                   observation_normalize= observation_normalize,
                                   seed= self.seed
                                   )

        # replay buffer 
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size,
                                          num_envs= self.num_envs,
                                          observation_space= self.vec_env.single_observation_space,
                                          action_space= self.vec_env.single_action_space, 
                                          device= self.device)
        self.step = 0 
           

    def set_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def learn(self, timesteps = 1000000):
        obs , _ = self.vec_env.reset(seed = [self.seed + i for i in range(self.num_envs)])
        count = 0 
        for step in range(timesteps):
            self.step = step
            total_critic_loss = 0 
            total_actor_loss = 0 
            total_return = 0 

            obs_tensor = torch.tensor(obs, dtype = torch.float32).to(self.device)
            action  = self.select_action(obs = obs_tensor,
                                        warm_up = True)

            next_obs, reward, terminated , truncated, _ = self.vec_env.step(action)

            self.replay_buffer.add(obs = obs,       
                                next_obs= next_obs, 
                                action=action,      
                                reward=reward,      
                                done = terminated)  

            obs = next_obs 

            if len(self.replay_buffer) >= self.batch_size:
                critic_loss , actor_loss = self.train()

                total_actor_loss += actor_loss 
                total_critic_loss += critic_loss 
                count += 1

            if ((step % 1000 == 0) and (count > 0)) or (step == timesteps - 1) or (step == 0):
                if self.observation_normalize:
                    stats_observation = {
                        'mean': self.vec_env.mean, 
                        'var': self.vec_env.var, 
                        'count': self.vec_env.count 
                    }
                else:
                    stats_observation = None 

                # evaluation
                for _ in range(10):
                    frames, return_val = self.eval(stats_observation = stats_observation)
                    total_return += return_val 

                logs = {'actor_loss': total_actor_loss/max(count,1),
                    'critic_loss': total_critic_loss/max(count,1), 
                    'eval_return': total_return.item()/10}

                # render evaluation 
                if self.use_wandb and (step % 10000 == 0):
                    frames, _ = self.eval(render = True,
                                        stats_observation = stats_observation)
                    self.logger.log_video(frames)

                self.logger.set_step(step)
                self.logger.log(logs)

    def eval(self):
        raise NotImplementedError
