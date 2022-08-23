from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        batch = {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    else:
        raise ValueError("Unsupported type passed to `to_device`")
    return batch


def to_tensor(batch: Any) -> Any:
    if isinstance(batch, dict):
        batch = {k: to_tensor(v) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [to_tensor(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        # Special case to handle float64 -- which we never want to use with pytorch
        if batch.dtype == np.float64:
            batch = batch.astype(np.float32)
        batch = torch.from_numpy(batch)
    else:
        raise ValueError("Unsupported type passed to `to_tensor`")
    return batch


def to_np(batch: Any) -> Any:
    if isinstance(batch, dict):
        batch = {k: to_np(v) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [to_np(v) for v in batch]
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu().numpy()
    else:
        raise ValueError("Unsupported type passed to `to_np`")
    return batch


def unsqueeze(batch: Any, dim: int) -> Any:
    if isinstance(batch, dict):
        batch = {k: unsqueeze(v, dim) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [unsqueeze(v, dim) for v in batch]
    elif isinstance(batch, np.ndarray):
        batch = np.expand_dims(batch, dim)
    elif isinstance(batch, torch.Tensor):
        batch = batch.unsqueeze(dim)
    elif isinstance(batch, (int, float, np.generic)):
        batch = np.array([batch])
    else:
        raise ValueError("Unsupported type passed to `unsqueeze`")
    return batch


def squeeze(batch: Any, dim: int) -> Any:
    if isinstance(batch, dict):
        batch = {k: squeeze(v, dim) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [squeeze(v, dim) for v in batch]
    elif isinstance(batch, np.ndarray):
        batch = np.squeeze(batch, axis=dim)
    elif isinstance(batch, torch.Tensor):
        batch = batch.squeeze(dim)
    else:
        raise ValueError("Unsupported type passed to `squeeze`")
    return batch


def get_from_batch(batch: Any, start: int, end: Optional[int] = None) -> Any:
    if isinstance(batch, dict):
        batch = {k: get_from_batch(v, start, end=end) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [get_from_batch(v, start, end=end) for v in batch]
    elif isinstance(batch, np.ndarray) or isinstance(batch, torch.Tensor):
        if end is None:
            batch = batch[start]
        else:
            batch = batch[start:end]
    else:
        raise ValueError("Unsupported type passed to `get_from_batch`")
    return batch


def set_in_batch(batch: Any, value: Any, start: int, end: Optional[int] = None) -> None:
    if isinstance(batch, dict):
        for v in batch.values():
            set_in_batch(v, value, start, end=end)
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for v in batch:
            set_in_batch(v, value, start, end=end)
    elif isinstance(batch, np.ndarray) or isinstance(batch, torch.Tensor):
        if end is None:
            batch[start] = value
        else:
            batch[start:end] = value
    else:
        raise ValueError("Unsupported type passed to `set_in_batch`")


def batch_copy(batch: Any) -> Any:
    if isinstance(batch, dict):
        batch = {k: batch_copy(v) for k, v in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        batch = [batch_copy(v) for v in batch]
    elif isinstance(batch, np.ndarray):
        batch = batch.copy()
    elif isinstance(batch, torch.Tensor):
        batch = batch.clone()
    # Note that if we have scalars etc. we just return the value, thus no ending check.
    return batch


def contains_tensors(batch: Any) -> bool:
    if isinstance(batch, dict):
        return any([contains_tensors(v) for v in batch.values()])
    if isinstance(batch, list):
        return any([contains_tensors(v) for v in batch])
    elif isinstance(batch, torch.Tensor):
        return True
    else:
        return False


def get_device(batch: Any) -> Optional[torch.device]:
    if isinstance(batch, dict):
        return get_device(list(batch.values()))
    elif isinstance(batch, list):
        devices = [get_device(d) for d in batch]
        for d in devices:
            if d is not None:
                return d
        else:
            return None
    elif isinstance(batch, torch.Tensor):
        return batch.device
    else:
        return None


def concatenate(*args, dim: int = 0):
    assert all([isinstance(arg, type(args[0])) for arg in args]), "Must concatenate tensors of the same type"
    if isinstance(args[0], dict):
        return {k: concatenate(*[arg[k] for arg in args], dim=dim) for k in args[0].keys()}
    elif isinstance(args[0], list) or isinstance(args[0], tuple):
        return [concatenate(*[arg[i] for arg in args], dim=dim) for i in range(len(args[0]))]
    elif isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=dim)
    elif isinstance(args[0], torch.Tensor):
        return torch.concatenate(args, dim=dim)
    else:
        raise ValueError("Unsupported type passed to `concatenate`")


class PrintNode(torch.nn.Module):
    def __init__(self, name: str = ""):
        super().__init__()
        self.name = name

    def forward(self, x: Any) -> Any:
        print(self.name, x.shape)
        return x


def np_dataset_alloc(
    space: gym.Space, capacity: int, begin_pad: Tuple[int] = tuple(), end_pad: Tuple[int] = tuple()
) -> np.ndarray:
    if isinstance(space, gym.spaces.Dict):
        return {k: np_dataset_alloc(v, capacity, begin_pad=begin_pad, end_pad=end_pad) for k, v in space.items()}
    elif isinstance(space, gym.spaces.Box):
        dtype = np.float32 if space.dtype == np.float64 else space.dtype
        return np.empty((capacity,) + begin_pad + space.shape + end_pad, dtype=dtype)
    elif isinstance(space, gym.spaces.Discrete) or isinstance(space, np.int64):
        return np.empty((capacity,) + begin_pad + end_pad, dtype=np.int64)
    elif isinstance(space, np.ndarray):
        dtype = np.float32 if space.dtype == np.float64 else space.dtype
        return np.empty((capacity,) + begin_pad + space.shape + end_pad, dtype=dtype)
    elif isinstance(space, float) or isinstance(space, np.float32):
        return np.empty((capacity,) + begin_pad + end_pad, dtype=np.float32)
    elif isinstance(space, bool):
        return np.empty((capacity,) + begin_pad + end_pad, dtype=np.bool_)
    else:
        raise ValueError("Invalid space provided")


def fetch_from_dict(data_dict: Dict, keys: Union[str, List, Tuple]) -> List[Any]:
    """
    inputs:
        data_dict: a nested dictionary datastrucutre
        keys: a list of string keys, with '.' separating nested items.
    """
    outputs = []
    if not isinstance(keys, list) and not isinstance(keys, tuple):
        keys = [keys]
    for key in keys:
        key_parts = key.split(".")
        current_dict = data_dict
        while len(key_parts) > 1:
            current_dict = current_dict[key_parts[0]]
            key_parts.pop(0)
        outputs.append(current_dict[key_parts[0]])
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
