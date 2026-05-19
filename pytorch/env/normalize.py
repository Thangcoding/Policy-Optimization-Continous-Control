import gymnasium as gym 
import numpy as np 

class NormalizeObservation(gym.vector.VectorWrapper):

    def __init__(self,env,
                clip=10.0,
                stats_observation=None
    ):
        super().__init__(env)

        obs_shape = (
            self.single_observation_space.shape
        )

        self.obs_dim = obs_shape[0]

        if stats_observation is None:

            self.mean = np.zeros(
                self.obs_dim,
                dtype=np.float64
            )

            self.var = np.ones(
                self.obs_dim,
                dtype=np.float64
            )

            self.count = 1e-4

        else:

            self.mean = (
                stats_observation["mean"]
            )

            self.var = (
                stats_observation["var"]
            )

            self.count = (
                stats_observation["count"]
            )

        self.clip = clip
        self.training_mode = True

    def _normalize(self, obs):

        normalized_obs = (
            obs - self.mean
        ) / np.sqrt(
            self.var + 1e-8
        )

        normalized_obs = np.clip(
            normalized_obs,
            -self.clip,
            self.clip
        )

        if self.training_mode:
            self._update(obs)

        return normalized_obs

    def reset(self, **kwargs):

        obs, info = self.env.reset(
            **kwargs
        )

        obs = self._normalize(obs)

        return obs, info

    def step(self, actions):

        (
            obs,
            rewards,
            terms,
            truncs,
            infos
        ) = self.env.step(actions)

        obs = self._normalize(obs)

        return (
            obs,
            rewards,
            terms,
            truncs,
            infos
        )

    def _update(self, obs):

        batch_mean = np.mean(
            obs,
            axis=0
        )

        batch_var = np.var(
            obs,
            axis=0
        )

        batch_count = obs.shape[0]

        delta = (
            batch_mean - self.mean
        )

        total = (
            self.count + batch_count
        )

        new_mean = (
            self.mean
            + delta
            * batch_count
            / total
        )

        m_a = (
            self.var * self.count
        )

        m_b = (
            batch_var * batch_count
        )

        M2 = (
            m_a
            + m_b
            + delta**2
            * self.count
            * batch_count
            / total
        )

        self.mean = new_mean
        self.var = M2 / total
        self.count = total