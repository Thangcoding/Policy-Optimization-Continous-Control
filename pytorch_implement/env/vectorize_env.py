import gymnasium as gym 
import numpy as np 
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

class VectorEnv:

    def __init__(self,env: gym.Env,
                num_envs: int = 4, 
                type_vector: str = "Async"
                ):
        '''
        Args:
            name_env : name of environment simulation in mujuco 
            num_envs : dimension of the vector environment 
            type_vector : have two types 
                        - Async (parallel vector running in multiprocess)
                        - Sync (sequential vector)

        '''
        self.env = env 
        self.type_vector = type_vector
        self.num_envs = num_envs

        def make_env(rank):
            def _init():
                e = gym.make(self.env.spec.id)
                e.reset(seed = rank)
                return e 
            return _init
        
        env_fns = [make_env(i) for i in range(self.num_envs)]
        if self.type_vector == "Async":
            self.vec_env = AsyncVectorEnv(env_fns)
        else:
            self.vec_env = SyncVectorEnv(env_fns) 
        
        # normalize observation
        self.vec_env = NormalizeObservation(self.vec_env)
        
        # normalize reward 
        self.vec_env = NormalizeReward(self.vec_env)
        
        self.action_space = self.vec_env.single_action_space
        self.observation_space = self.vec_env.single_observation_space

    def step(self, actions : np.ndarray) -> tuple:
        next_obs, rewards, terminated, truncated , infos = self.vec_env.step(actions)

        return next_obs, rewards, terminated, truncated, infos
    
    def action_sample(self) -> np.ndarray[np.float32]:
        return self.vec_env.action_space.sample()
    
    def reset(self)-> tuple:
        obs, infos = self.vec_env.reset()
        return obs, infos 
    
    def close(self):
        self.vec_env.close()

