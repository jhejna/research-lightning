from typing import Tuple

import torch
from gym import spaces

from .base import Processor


class Flatten(Processor):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        assert isinstance(observation_space, spaces.Box)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        # Flatten the x's
        return x.flatten(start_dim=1, end_dim=-1), y

    def unprocess(self, batch):
        x, y = batch
        x = x.reshape(x.shape[0], *self.observation_space.shape)
        return x, y
