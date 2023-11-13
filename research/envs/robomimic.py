from typing import List, Optional

import gym
import h5py
import numpy as np
from robomimic.utils import env_utils, file_utils


class RobomimicEnv(gym.Env):
    def __init__(
        self, path: str, keys: Optional[List] = None, terminate_early: bool = True, horizon: Optional[int] = None
    ):
        # Get the observation space
        f = h5py.File(path, "r")
        demo_id = list(f["data"].keys())[0]
        demo = f["data/{}".format(demo_id)]
        if keys is None:
            self.keys = list(demo["obs"].keys())
        else:
            self.keys = keys
        spaces = {}
        use_image_obs = False
        for k in self.keys:
            if "image" in k:
                use_image_obs = True
            obs_modality = demo["obs/{}".format(k)]
            if obs_modality.dtype == np.uint8:
                low, high = 0, 255
            elif obs_modality.dtype == np.float32 or obs_modality.dtype == np.float64:
                low, high = -np.inf, np.inf
            else:
                raise ValueError("Unsupported dtype in Robomimic Env.")
            dtype = np.float32 if obs_modality.dtype == np.float64 else obs_modality.dtype
            spaces[k] = gym.spaces.Box(low=low, high=high, shape=obs_modality.shape[1:], dtype=dtype)
        self.observation_space = gym.spaces.Dict(spaces)
        f.close()

        # Create the environment.
        env_meta = file_utils.get_env_metadata_from_dataset(dataset_path=path)
        self.env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=use_image_obs,
        ).env
        self.env.ignore_done = False
        if horizon is not None:
            self.env.horizon = horizon
        self.env._max_episode_steps = self.env.horizon
        self.terminate_early = terminate_early

        # Get the action space
        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def _filter_obs(self, obs):
        if "object-state" in obs:
            # Need to duplicate because of robomimic bug.
            obs["object"] = obs["object-state"]
        return {k: obs[k] for k in self.keys}

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        if self.terminate_early and self.env._check_success():
            done = True
        return self._filter_obs(obs), reward, done, info

    def reset(self, *args, **kwargs):
        return self._filter_obs(self.env.reset(*args, **kwargs))