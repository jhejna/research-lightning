import copy
from typing import Callable, Optional, Tuple

import numpy as np

from research.utils import utils

from .storage import Storage

"""
This file defines a number of sampling functions used by the replay buffer.

Each sample function returns tensors of the following shape:
(Batch, Time, dims...)
and requires `storage` and `discount` arguments.

Many of these functions have large blocks of repeated code, but
are implemented separately for readability and performance optimiztaion.

Sequences are sampled as follows:
-stack_length ...  -1, 0, 1, 2, ..., seq_length
|         stack      |idx|        seq          |
The stack parameter will always be sampled immediately, and is desinged to be used as context
to the network.
Stack will not obey nstep returns. (negative indexing)

Everything is sampled in batches directly from memory (preferred)
If batch_size is set to one, then a squeeze operation will be performed at the very end.

Samples are returned as with shape: (Batch, Time, Dims...)
if seq or stack dims are set to 1, then these parameters are ignored.
"""


def _get_ep_idxs(storage: Storage, batch_size: int = 1, sample_by_timesteps: bool = True, min_length: int = 2):
    if batch_size is None or batch_size > 1:
        ep_idxs = np.arange(len(storage.lengths))[storage.lengths >= min_length]
        if sample_by_timesteps:
            # Lower the lengths by the min_length - 1 to give the number of valid sequences.
            lengths = storage.lengths[ep_idxs] - (min_length - 1)
            p = lengths / lengths.sum()
            ep_idxs = np.random.choice(ep_idxs, size=(batch_size,), replace=True, p=p)
        else:
            ep_idxs = ep_idxs[np.random.randint(0, len(ep_idxs), size=(batch_size,))]
        return ep_idxs
    else:
        # Use a different, much faster sampling scheme for batch_size = 1
        assert sample_by_timesteps is False, "Cannot sample by timesteps with batch_size=1, it's too slow!"
        ep_idx = np.random.randint(0, len(storage.lengths))
        if storage.lengths[ep_idx] < min_length:
            return _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
        else:
            return np.array([ep_idx], np.int64)


def sample(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    stack: int = 1,
    stack_keys: Tuple = (),
    seq: int = 1,
    seq_keys: Tuple = (),
    pad: int = 0,
):
    """
    Default sampling for imitation learning.
    Returns (obs, action, ... keys) batches.
    """
    assert seq > pad, "seq lenght must be larger than pad."
    min_length = seq - pad + 1
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # Distance we can sample past the end is end + pad - (seq - 1)
    # Afterwards, add one for the ep start offset.
    idxs = np.random.randint(starts, ends + (pad - seq + 1)) + 1
    batch = {}  # initialize the batch so we can write masks.

    if stack > 1:
        assert len(stack_keys) > 0, "Provided stack > 1 but no stack keys"
        stack_idxs = np.expand_dims(idxs, axis=-1) + np.arange(-stack + 1, 1)
        stack_idxs = np.maximum(stack_idxs, np.expand_dims(starts, axis=-1))

    if seq > 1:
        assert len(seq_keys) > 0, "Provided seq_length > 1 but no seq keys"
        seq_idxs = np.expand_dims(idxs, axis=-1) + np.arange(seq)  # (B, T)
        # Compute the mask BEFORE we trim down to end
        mask = seq_idxs > np.expand_dims(ends, axis=-1)
        batch["mask"] = mask
        seq_idxs = np.minimum(seq_idxs, np.expand_dims(ends, axis=-1))

    # Sample from the dataset
    for k in storage.keys():
        if k in stack_keys:
            sample_idxs = stack_idxs
        elif k in seq_keys:
            sample_idxs = seq_idxs
        else:
            sample_idxs = idxs

        if k == "obs":
            sample_idxs = sample_idxs - 1  # NOTE: cannot modify in place, would be sneaky bug.

        batch[k] = utils.get_from_batch(storage[k], sample_idxs)
    return batch


def sample_qlearning(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    nstep: int = 1,
    stack: int = 1,
    stack_keys: Tuple = (),
    discount: float = 0.99,
):
    """
    Default sampling for reinforcement learning.
    Returns (obs, action, reward, discount, next_obs) batches.

    Similar to the default `sample` method, but removes the sequence option and
    limits sampling to the keys used for RL.
    """
    min_length = 1 + nstep
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # We cannot sample less than nstep from the end.
    # Afterwards, add one for the ep start offset.
    idxs = np.random.randint(starts, ends - (nstep - 1)) + 1

    if stack > 1:
        assert len(stack_keys) > 0, "Provided stack > 1 but no stack keys"
        stack_idxs = np.expand_dims(idxs, axis=-1) + np.arange(-stack + 1, 1)
        stack_idxs = np.maximum(stack_idxs, np.expand_dims(starts, axis=-1))

    # Get the observation indexes
    obs_idxs = stack_idxs if "obs" in stack_keys else idxs
    next_obs_idxs = obs_idxs + nstep - 1
    obs_idxs = obs_idxs - 1

    # Get the action indexes
    action_idxs = stack_idxs if "action" in stack_keys else idxs

    obs = utils.get_from_batch(storage["obs"], obs_idxs)
    action = utils.get_from_batch(storage["action"], action_idxs)
    reward = np.zeros(idxs.shape, dtype=np.float32)
    discount_batch = np.ones(idxs.shape, dtype=np.float32)
    for i in range(nstep):
        reward += discount_batch * storage["reward"][idxs + i]
        discount_batch *= discount * storage["discount"][idxs + i]
    next_obs = utils.get_from_batch(storage["obs"], next_obs_idxs)

    return dict(obs=obs, action=action, reward=reward, discount=discount_batch, next_obs=next_obs)


def sample_her(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    achieved_key: str = "achieved_goal",
    goal_key: str = "desired_goal",
    strategy: str = "future",
    relabel_fraction: float = 0.5,
    stack: int = 1,
    stack_keys: Tuple = (),
    seq: int = 1,
    seq_keys: Tuple = (),
    pad: int = 0,
):
    """
    Default sampling for imitation learning.
    Returns (obs, action, ... keys) batches.
    """
    assert isinstance(storage["obs"], dict)
    min_length = seq - pad + 1
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # Distance we can sample past the end is end + pad - (seq - 1)
    # Afterwards, add one for the ep start offset.
    idxs = np.random.randint(starts, ends + (pad - seq + 1)) + 1
    batch = {}  # initialize the batch so we can write masks.

    if stack > 1:
        assert len(stack_keys) > 0, "Provided stack > 1 but no stack keys"
        stack_idxs = np.expand_dims(idxs, axis=-1) + np.arange(-stack + 1, 1)
        stack_idxs = np.maximum(stack_idxs, np.expand_dims(starts, axis=-1))

    if seq > 1:
        assert len(seq_keys) > 0, "Provided seq_length > 1 but no seq keys"
        seq_idxs = np.expand_dims(idxs, axis=-1) + np.arange(seq)  # (B, T)
        # Compute the mask BEFORE we trim down to end
        mask = seq_idxs > np.expand_dims(ends, axis=-1)
        batch["mask"] = mask
        seq_idxs = np.minimum(seq_idxs, np.expand_dims(ends, axis=-1))
        last_idxs = seq_idxs[..., -1]  # Get the last index from every sequence
    else:
        last_idxs = idxs  # only a single transition

    her_idxs = np.where(np.random.uniform(size=idxs.shape) < relabel_fraction)
    if strategy == "last":
        goal_idxs = ends[her_idxs]
    elif strategy == "next":
        goal_idxs = last_idxs[her_idxs]
    elif strategy == "future":
        goal_idxs = np.random.randint(last_idxs[her_idxs], ends[her_idxs] + 1)
    elif strategy == "future_inclusive":
        goal_idxs = np.random.randint(last_idxs[her_idxs] - 1, ends[her_idxs] + 1)
    else:
        raise ValueError("Invalid HER strategy chosen.")

    if relabel_fraction < 1.0:
        # Need to copy out existing values.
        desired = copy.deepcopy(utils.get_from_batch(storage["obs"][goal_key], idxs - 1))
        achieved = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        utils.set_in_batch(desired, achieved, her_idxs)
        if "horizon" in storage:
            horizon = utils.get_from_batch(storage["horizon"], idxs - 1)
        else:
            horizon = -100 * np.ones_like(idxs, dtype=np.int)
        horizon[her_idxs] = goal_idxs - idxs[her_idxs] + 1
    else:
        # Grab directly from the buffer.
        desired = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        horizon = goal_idxs - idxs + 1  # Horizon = goal_idxs - obs_idxs

    if "obs" in stack_keys or "obs" in seq_keys:
        desired = utils.unsqueeze(desired, dim=1)  # Add temporal dim.

    # Sample from the dataset
    for k in storage.keys():
        if k in stack_keys:
            sample_idxs = stack_idxs
        elif k in seq_keys:
            sample_idxs = seq_idxs
        else:
            sample_idxs = idxs

        if k == "obs":
            # Sample all obs keys except the goal key with a -1 offset!
            batch[k] = {
                obs_key: utils.get_from_batch(storage[k][obs_key], sample_idxs - 1)
                for obs_key in storage[k].keys()
                if obs_key != goal_key
            }
        else:
            batch[k] = utils.get_from_batch(storage[k], sample_idxs)

    # Update the batch to use the newly set desired goals
    batch["obs"][goal_key] = desired
    batch["horizon"] = horizon

    return batch


def sample_her_qlearning(
    storage: Storage,
    batch_size: int = 1,
    sample_by_timesteps: bool = True,
    nstep: int = 1,
    achieved_key: str = "achieved_goal",
    goal_key: str = "desired_goal",
    stack: int = 1,
    stack_keys: Tuple = (),
    strategy: str = "future",
    relabel_fraction: float = 0.5,
    reward_fn: Optional[Callable] = None,
    discount=0.99,
):
    """
    Default sampling for reinforcement learning with HER
    Returns (obs, action, reward, discount, next_obs) batches.

    Similar to the default `sample` method, but includes limits sampling
    to the required keys.
    """
    min_length = 1 + nstep
    ep_idxs = _get_ep_idxs(storage, batch_size, sample_by_timesteps, min_length)
    starts, ends = storage.starts[ep_idxs], storage.ends[ep_idxs]
    # We cannot sample less than nstep from the end.
    # Afterwards, add one for the ep start offset.
    idxs = np.random.randint(starts, ends - (nstep - 1)) + 1

    if stack > 1:
        assert len(stack_keys) > 0, "Provided stack > 1 but no stack keys"
        stack_idxs = np.expand_dims(idxs, axis=-1) + np.arange(-stack + 1, 1)
        stack_idxs = np.maximum(stack_idxs, np.expand_dims(starts, axis=-1))

    her_idxs = np.where(np.random.uniform(size=idxs.shape) < relabel_fraction)
    if strategy == "last":
        goal_idxs = ends[her_idxs]
    elif strategy == "next":
        goal_idxs = idxs[her_idxs]
    elif strategy == "future":
        goal_idxs = np.random.randint(idxs[her_idxs], ends[her_idxs] + 1)
    elif strategy == "future_inclusive":
        goal_idxs = np.random.randint(idxs[her_idxs] - 1, ends[her_idxs] + 1)
    else:
        raise ValueError("Invalid HER strategy chosen.")

    if relabel_fraction < 1.0:
        # Need to copy out existing values.
        desired = copy.deepcopy(utils.get_from_batch(storage["obs"][goal_key], idxs - 1))
        achieved = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        utils.set_in_batch(desired, achieved, her_idxs)
        if "horizon" in storage:
            horizon = utils.get_from_batch(storage["horizon"], idxs - 1)
        else:
            horizon = -100 * np.ones_like(idxs, dtype=np.int)
        horizon[her_idxs] = goal_idxs - idxs[her_idxs] + 1
    else:
        # Grab directly from the buffer.
        desired = utils.get_from_batch(storage["obs"][achieved_key], goal_idxs)
        horizon = goal_idxs - idxs + 1  # Horizon = goal_idxs - obs_idxs

    # Get the observation indexes
    obs_idxs = stack_idxs if "obs" in stack_keys else idxs
    next_obs_idxs = obs_idxs + nstep - 1
    obs_idxs = obs_idxs - 1

    # Get the action indexes
    action_idxs = stack_idxs if "action" in stack_keys else idxs

    reward = np.zeros(idxs.shape, dtype=np.float32)
    discount_batch = np.ones(idxs.shape, dtype=np.float32)
    for i in range(nstep):
        if reward_fn is None:
            # If reward function is None, use sparse indicator reward for all timesteps
            # after reaching the goal
            # If the next obs is the goal (aka horizon = 1), then reward = 1!
            step_reward = (horizon <= i + 1).astype(np.float32)
        else:
            achieved = utils.get_from_batch(storage["obs"][achieved_key], idxs + i)
            step_reward = reward_fn(achieved, desired)
        reward += discount_batch * step_reward
        discount_batch *= discount * storage["discount"][idxs + i]

    # Now round the horizon key according to nstep
    if nstep > 1:
        horizon[horizon >= 0] = np.ceil(horizon[horizon >= 0] / nstep)

    if "obs" in stack_keys:
        # Add temporal dimension if we stack the achieved frames.
        desired = utils.unsqueeze(desired, dim=1)  # Add temporal dim.

    obs = {k: utils.get_from_batch(storage["obs"][k], obs_idxs) for k in storage["obs"].keys() if k != goal_key}
    obs[goal_key] = desired
    next_obs = {
        k: utils.get_from_batch(storage["obs"][k], next_obs_idxs) for k in storage["obs"].keys() if k != goal_key
    }
    next_obs[goal_key] = desired
    action = utils.get_from_batch(storage["action"], action_idxs)

    return dict(obs=obs, action=action, reward=reward, discount=discount_batch, next_obs=next_obs, horizon=horizon)
