import torch
import numpy as np
import gymnasium as gym 
from gymnasium import spaces
from ..env.vectorize_env import VectorEnv
from ..env.rollout_buffer import RolloutBuffer 
from ..env.replay_buffer import ReplayBuffer
from ..model.stochastic_model import ActorCritic
from ..utils.feature_extractor import BaseFeatureExtractor
from ..utils.logger import Logger , record_video

class OnPolicyAlgorithm:
    def __init__(self,env : gym.Env,
                num_envs: int,
                feature_network: str | type[BaseFeatureExtractor],
                feature_dim: int,  
                learning_rate: float, 
                n_rollout_steps: int, 
                type_vector: str,  
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
        self.seed = seed 

        self.device = device
        self.logger = Logger(use_wandb= use_wandb)

        # setup vector env 
        self.eval_env = gym.make(env.spec.id, render_mode = "rgb_array")
        self.vec_env = VectorEnv(env,num_envs,type_vector)

        # setup_model 
        self.agent = ActorCritic(feature_network, self.vec_env.observation_space, self.vec_env.action_space, feature_dim).to(device)

        # setup rollout 
        self.rollout_buffer = RolloutBuffer(buffer_size = n_rollout_steps, 
                                            num_envs = num_envs, 
                                            observation_space = self.vec_env.observation_space, 
                                            action_space = self.vec_env.action_space, 
                                            device = device)
        # optimizer 
        self.optimizer = torch.optim.Adam(self.agent.parameters(),lr = learning_rate)

    def collect_rollouts(self):
        self.rollout_buffer.reset()

        obs , _ = self.vec_env.reset(seed = self.seed)

        for _ in range(self.n_rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                action, log_prob , value = self.agent.predict(obs_tensor)

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
            _, _, last_value = self.agent.predict(obs_tensor)
            last_value = last_value.cpu().numpy()

        # mask done 
        last_value = last_value *(1 - done)
        self.rollout_buffer.gae_and_return_value(
            last_value, self.gamma, self.gae_lambda
        )

    def train(self):
        raise NotImplementedError

    def learn(self,total_timesteps, n_epochs: int = 1):
        render_ratio = int(total_timesteps / 10) 

        while self.global_steps < total_timesteps:
            self.collect_rollouts()

            self.global_steps += self.n_rollout_steps
            loss, policy_loss , value_loss , entropy , avg_return , avd_mean, avd_std = 0, 0, 0, 0, 0, 0, 0

            for _ in range(n_epochs):
                batch_logs = self.train()

                loss += batch_logs['loss']
                policy_loss += batch_logs['policy_loss']
                value_loss += batch_logs['value_loss'] 
                entropy += batch_logs['entropy']
                avg_return += batch_logs['avg_return']
                avd_mean += batch_logs['adv_mean']
                avd_std += batch_logs['adv_std']

            # log store 
            logs = {"loss": loss / n_epochs, 
                    "policy_loss": policy_loss / n_epochs,
                    "value_loss": value_loss / n_epochs, 
                    "entropy": entropy / n_epochs,
                    "avg_return": avg_return / n_epochs,
                    "avd_mean": avd_mean / n_epochs,
                    "avd_std": avd_std / n_epochs}
            
            self.logger.set_step(self.global_steps)
            self.logger.log(logs)

            # evaluation 
            if (self.global_steps >= render_ratio) and self.logger.use_wandb:
                frames = record_video(self.eval_env, self.agent, self.device)
                self.logger.set_step(self.global_steps)
                self.logger.log_video(frames)
                render_ratio *= 2

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
                type_vector: str, 
                gamma: float, 
                use_wandb: bool, 
                seed: int, 
                device: torch.device, 
                ):
        self.env = env 
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.type_vector = type_vector
        self.gamma = gamma 

        self.device = device 
        self.logger = Logger(use_wandb= use_wandb)   

        # setup vector env 
        self.eval_env = gym.make(self.env.spec.id, render_mode = "rgb_array")
        self.vec_env = VectorEnv(self.env,self.num_envs, self.type_vector)

        # replay buffer 
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size,
                                          num_envs= self.num_envs,
                                          observation_space= self.vec_env.observation_space,
                                          action_space= self.vec_env.action_space, 
                                          device= self.device)
           

    def set_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def learn(self, timesteps = 200, episodes = 10):
        render_ratio = max(int(episodes/10),1)

        for ep in range(episodes):

            for step in range(timesteps):
                obs , _ = self.vec_env.reset()
                obs_tensor = torch.tensor(obs, dtype = torch.float32).to(self.device)
                action  = self.select_action(obs_tensor)

                next_obs, reward, terminated , truncated, _ = self.vec_env.step(action)


                self.replay_buffer.add(obs = obs,
                                    next_obs= next_obs, 
                                    action=action,
                                    reward=reward,
                                    done = terminated)

                obs = next_obs 

                self.train(step)

            # evaluation 
            if episodes % render_ratio and self.logger.use_wandb:
                frames = self.eval()

                self.logger.log_video(frames)
                self.logger.set_step(ep*step)

    def eval(self):
        raise NotImplementedError





