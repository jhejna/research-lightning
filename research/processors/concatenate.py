from typing import Dict, List, Optional

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
        if self.concat_obs:
            # Concatenate the spaces on the last dim
            low = np.concatenate([space.low for space in self._observation_space.values()], axis=-1)
            high = np.concatenate([space.high for space in self._observation_space.values()], axis=-1)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)  # force float32 conversion
        else:
            return self._observation_space

    @property
    def action_space(self):
        if self.concat_action:
            # Concatenate the spaces on the last dim
            low = np.concatenate([space.low for space in self._action_space.values()], axis=-1)
            high = np.concatenate([space.high for space in self._action_space.values()], axis=-1)
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)  # force float32 conversion
        else:
            return self._action_space

    def forward(self, batch: Dict) -> Dict:
        batch = {k: v for k, v in batch.items()}  # Perform a shallow copy of the batch
        if self.concat_action and "action" in batch:
            batch["action"] = torch.cat([batch["action"][k] for k in self.action_order], dim=-1)
        if self.concat_obs and "obs" in batch:
            batch["obs"] = torch.cat([batch["obs"][k] for k in self.obs_order], dim=-1)
            if "next_obs" in batch:
                batch["next_obs"] = torch.cat([batch["next_obs"][k] for k in self.obs_order], dim=-1)
        return batch


class SelectProcessor(Processor):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        obs_include: Optional[List[str]] = None,
        obs_exclude: Optional[List[str]] = None,
        action_include: Optional[List[str]] = None,
        action_exclude: Optional[List[str]] = None,
    ):
        super().__init__(observation_space, action_space)
        assert not (action_include is not None and action_exclude is not None)
        assert not (obs_include is not None and obs_exclude is not None)

        if action_include is not None:
            self.action_keys = [k for k in action_space.keys() if k in action_include]
        elif action_exclude is not None:
            self.action_keys = [k for k in action_space.keys() if k not in action_exclude]
        else:
            self.action_keys = None
        if self.action_keys is not None:
            self._action_space = gym.spaces.Dict({k: v for k, v in self._action_space.items() if k in self.action_keys})

        if obs_include is not None:
            self.obs_keys = [k for k in observation_space.keys() if k in obs_include]
        elif obs_exclude is not None:
            self.obs_keys = [k for k in observation_space.keys() if k not in obs_exclude]
        else:
            self.obs_keys = None
        if self.obs_keys is not None:
            self._observation_space = gym.spaces.Dict(
                {k: v for k, v in self._observation_space.items() if k in self.obs_keys}
            )

    def forward(self, batch: Dict) -> Dict:
        if "action" in batch and self.action_keys is not None:
            batch["action"] = {k: batch["action"][k] for k in self.action_keys}
        if "obs" in batch and self.obs_keys is not None:
            batch["obs"] = {k: batch["obs"][k] for k in self.obs_keys}
            if "next_obs" in batch:
                batch["next_obs"] = {k: batch["next_obs"][k] for k in self.obs_keys}
        return batch
