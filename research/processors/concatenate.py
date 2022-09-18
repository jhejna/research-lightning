from typing import Dict

import gym
import numpy as np
import torch

from .base import Processor


class ConcatenateProcessor(Processor):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, concat_obs: bool = True, concat_action: bool = True
    ) -> None:
        super().__init__(observation_space, action_space)
        self.concat_action = concat_action and isinstance(action_space, gym.spaces.Dict)
        if self.concat_action:
            self.action_order = list(action_space.keys())
        self.concat_obs = concat_obs and isinstance(observation_space, gym.spaces.Dict)
        if self.concat_obs:
            self.obs_order = list(observation_space.keys())

    @property
    def observation_space(self):
        # Concatenate the spaces on the last dim
        low = np.concatenate([space.low for space in self._observation_space.values()], axis=-1)
        high = np.concatenate([space.high for space in self._observation_space.values()], axis=-1)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)  # force float32 conversion

    @property
    def action_space(self):
        # Concatenate the spaces on the last dim
        low = np.concatenate([space.low for space in self._action_space.values()], axis=-1)
        high = np.concatenate([space.high for space in self._action_space.values()], axis=-1)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)  # force float32 conversion

    def forward(self, batch: Dict) -> Dict:
        batch = {k: v for k, v in batch.items()}  # Perform a shallow copy of the batch
        if self.concat_action and "action" in batch:
            batch["action"] = torch.cat([batch["action"][k] for k in self.action_order], dim=-1)
        if self.concat_obs and "obs" in batch:
            batch["obs"] = torch.cat([batch["obs"][k] for k in self.obs_order], dim=-1)
            if "next_obs" in batch:
                batch["next_obs"] = torch.cat([batch["next_obs"][k] for k in self.obs_order], dim=-1)
        return batch
