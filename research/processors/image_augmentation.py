from typing import Dict

import gym
import numpy as np
import torch
from torch.nn import functional as F

from .base import Processor


class RandomShiftsAug(Processor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, pad: int = 4) -> None:
        super().__init__(observation_space, action_space)
        self.pad = pad
        self._padding = tuple([self.pad] * 4)

        # Get the image keys and sequence lengths
        if isinstance(observation_space, gym.spaces.Box):
            assert RandomShiftsAug.is_image_space(observation_space)
            self.is_sequence = len(observation_space.shape) == 4
            self.image_keys = None
        elif isinstance(observation_space, gym.spaces.Dict):
            image_keys = []
            sequence = []
            for k, v in observation_space.items():
                if RandomShiftsAug.is_image_space(v):
                    image_keys.append(k)
                    if len(v.shape) == 4:
                        sequence.append(v.shape[0])  # Append the sequence dim
                    else:
                        sequence.append(0)
            assert all(sequence) or (not any(sequence)), "All image keys must be sequence or not"
            self.is_sequence = sequence[0]
            self.image_keys = image_keys
        else:
            raise ValueError("Invalid observation space specified")

    @staticmethod
    def is_image_space(space):
        shape = space.shape
        is_image_space = (len(shape) == 3 or len(shape) == 4) and space.dtype == np.uint8
        if is_image_space:
            assert shape[-1] == shape[-2], "Height must equal width"
        return is_image_space

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        assert len(size) == 4, "_aug supports images of shape (b, c, h, w)"
        n, _, h, w = size
        assert h == w, "RandomShiftsAug only works on square images."
        x = F.pad(x, self._padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    def forward(self, batch: Dict) -> Dict:
        if not self.training:
            return batch

        # Images are assumed to be of shape (B, S, C, H, W) or (B, C, H, W) if there is no sequence dimension
        images = []
        split = []
        for k in ("obs", "next_obs", "init_obs"):
            if k in batch:
                if self.image_keys is None:
                    images.append(batch[k])
                    split.append(batch[k].shape[1])
                else:
                    images.extend([batch[k][img_key] for img_key in self.image_keys])
                    split.extend([batch[k][img_key].shape[1] for img_key in self.image_keys])

        with torch.no_grad():
            images = torch.cat(images, dim=1)  # This is either the seq dim or channel dim
            if self.is_sequence:
                n, s, c, h, w = images.size()
                images = images.view(n, s * c, h, w)
            images = self._aug(images.float())  # Apply the same augmentation to each data pt.
            if self.is_sequence:
                images = images.view(n, s, c, h, w)
            # Split according to the dimension 1 splits
            images = torch.split(images, split, dim=1)

        # Iterate over everything in the same order and overwrite in the batch
        i = 0
        for k in ("obs", "next_obs", "init_obs"):
            if k in batch:
                if self.image_keys is None:
                    batch[k] = images[i]
                    i += 1
                else:
                    for img_key in self.image_keys:
                        batch[k][img_key] = images[i]
                        i += 1
        assert i == len(images), "Did not write batch all augmented images."
        return batch
