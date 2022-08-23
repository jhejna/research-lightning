from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import torch
import torchvision


# Define an easy way of fetching transforms
def get_transforms(transforms):
    assembled_transforms = []
    for transform_name, transform_kwargs in transforms:
        transform = vars(torchvision.transforms)[transform_name](**transform_kwargs)
        assembled_transforms.append(transform)
    return torchvision.transforms.Compose(assembled_transforms)


class TorchVisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        dataset: str,
        root: str,
        transform: List[Tuple[str, Dict]] = [
            ("ToTensor", {}),
        ],
        **kwargs,
    ):
        super().__init__()
        transform = get_transforms(transform)
        dataset_class = vars(torchvision.datasets)[dataset]
        self.dataset = dataset_class(root, transform=transform, download=True, **kwargs)
        # Verify the shape of the dataset
        x, y = self.dataset[0]
        assert x.shape == observation_space.shape, "Model did not have correct observation shape"
        assert isinstance(action_space, gym.spaces.Discrete)
        assert np.isscalar(y)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)
