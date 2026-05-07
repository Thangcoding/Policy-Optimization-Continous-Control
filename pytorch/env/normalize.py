import gymnasium as gym 
import numpy as np 

class NormalizeObservation(gym.ObservationWrapper):
    
    def __init__(self, env, clip = 10.0, mean = None, var = None , count = None):
        super().__init__(env)

        obs_shape = self.observation_space.shape 
        self.obs_dim = obs_shape[-1]

        if mean is None:
            self.mean = np.zeros(self.obs_dim, dtype = np.float64)
            self.var = np.ones(self.obs_dim, dtype = np.float64)

            self.count = 1e-4 
        else:
            self.mean = mean 
            self.var = var 
            self.count = count 

        self.clip = clip 

        self.training_mode = True 
    
    def observation(self, obs):

        if self.training_mode:
            self._update(obs)

        obs = (obs - self.mean) / np.sqrt(self.var + 1e-8)

        obs = np.clip(obs, -self.clip, self.clip)

        return obs 

    def _update(self, obs):

        obs = np.asarray(obs)

        batch_mean = np.mean(obs, axis = 0)
        batch_var = np.var(obs, axis = 0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean 
        total = self.count + batch_count 
        
        new_mean = self.mean + delta * batch_count / total 

        m_a = self.var * self.count 
        m_b = batch_var * batch_count 

        M2 = m_a + m_b + delta **2 * self.count * batch_count / total 

        self.mean = new_mean
        self.var = M2/ total 
        self.count = total 


