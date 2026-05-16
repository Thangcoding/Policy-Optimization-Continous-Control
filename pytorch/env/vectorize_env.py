import gymnasium as gym 
import numpy as np 
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from .normalize import NormalizeObservation


def get_vec_env(env: gym.Env,
                num_envs: int = 4, 
                type_vector: str = "Async", 
                observation_normalize: bool = True, 
                stats_observation = None, 
                render: bool = False, 
                seed : int = 64

                ):

    '''
    Args:
        name_env : name of environment simulation in mujuco 
        num_envs : dimension of the vector environment 
        type_vector : have two types 
                    - Async (parallel vector running in multiprocess)
                    - Sync (sequential vector)
    '''

    def make_env(i):
        def _init():
            e = gym.make(env.spec.id, render_mode = "rgb_array") if render else gym.make(env.spec.id)
            e.reset(seed = seed + i)
            e.action_space.seed(seed + i)
            return e 
        return _init
    
    env_fns = [make_env(i) for i in range(num_envs)]

    if type_vector == "Async":
        vector = AsyncVectorEnv(env_fns)
    else:
        vector = SyncVectorEnv(env_fns) 
    
    if observation_normalize:
        vec_env = NormalizeObservation(vector, stats_observation)
    else:
        vec_env = vector
    
    return vec_env