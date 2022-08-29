import collections
import copy
import datetime
import io
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np
import torch

from research.utils.utils import get_from_batch, np_dataset_alloc, squeeze


def save_episode(episode: Dict, path: str) -> None:
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def load_episode(path: str) -> Dict:
    with open(path, "rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    This replay buffer is carefully implemented to run efficiently and prevent multiprocessing
    memory leaks and errors.
    All variables starting with an underscore ie _variable are used only by the child processes
    All other variables are used by the parent process.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discount: float = 0.99,
        nstep: int = 1,
        preload_path: str = None,
        storage_path: str = None,
        capacity: int = 100000,
        fetch_every: int = 1000,
        cleanup: bool = True,
        batch_size: Optional[int] = None,
        sample_multiplier: float = 1.5,
        stack: int = 1,
        pad: int = 0,
        use_next_obs: bool = True,
    ):
        # Observation and action space values
        self.observation_space = observation_space
        self.action_space = action_space

        # Queuing values
        self.discount = discount
        self.nstep = nstep
        self.stack = stack
        self.batch_size = 1 if batch_size is None else batch_size
        if pad > 0:
            assert self.stack > 1, "Pad > 0 doesn't make sense if we are not padding."
        self.pad = pad
        self.use_next_obs = use_next_obs

        # Data storage values
        self.capacity = capacity
        self.cleanup = cleanup  # whether or not to remove loaded episodes from disk
        self.fetch_every = fetch_every
        self.sample_multiplier = sample_multiplier

        # worker values to be shared across processes.
        self.preload_path = preload_path
        self.num_episodes = 0
        if storage_path is None:
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
        else:
            self.storage_path = storage_path
            os.makedirs(storage_path, exist_ok=False)
        print("[research] Replay Buffer Storage Path", self.storage_path)

    @property
    def is_parallel(self) -> bool:
        return not hasattr(self, "is_serial")

    @property
    def is_setup(self) -> bool:
        return hasattr(self, "_setup")

    def save(self, path: str) -> None:
        """
        Save the replay buffer to the specified path. This is literally just copying the files
        from the storage path to the desired path. By default, we will also delete the original files.
        """
        if self.cleanup:
            print("[research] Warning, attempting to save a cleaned up replay buffer. There are likely no files")
        os.makedirs(path, exist_ok=True)
        srcs = os.listdir(self.storage_path)
        for src in srcs:
            shutil.move(os.path.join(self.storage_path, src), os.path.join(path, src))
        print("Successfully saved", len(srcs), "episodes.")

    def __del__(self):
        if not self.cleanup:
            return
        paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
        for path in paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(self.storage_path)
        except:
            pass

    def setup(self) -> None:
        if self.is_setup:
            assert (
                not self.is_parallel
            ), "Recalled setup on parallel replay buffer! This means __iter__ was called twice."
            return  # We are in serial mode, we can create another iterator
        else:
            self._setup = True

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Be EXTREMELEY careful here to not modify any values that are in the parent object.
            # This is only called if we are in the serial case!
            self.is_serial = True
        # Setup values to be used by this worker in setup
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0

        # Setup the buffers
        self._idx = 0
        self._size = 0
        self._capacity = self.capacity // self._num_workers

        self._obs_buffer = np_dataset_alloc(self.observation_space, self._capacity)
        self._action_buffer = np_dataset_alloc(self.action_space, self._capacity)
        self._reward_buffer = np_dataset_alloc(0.0, self._capacity)
        self._discount_buffer = np_dataset_alloc(0.0, self._capacity)
        self._done_buffer = np_dataset_alloc(False, self._capacity)
        self._info_buffers = dict()

        # setup episode tracker to track loaded episodes
        self._episode_filenames = set()
        self._samples_since_last_load = 0

        # Preload the data if needed
        if self.preload_path is not None:
            self._load(self.preload_path, cleanup=False)  # Load any initial episodes
        self._load(self.storage_path, cleanup=self.cleanup)  # Load anything we added before starting the iterator.

    def _add_to_buffer(self, obs: Any, action: Any, reward: Any, done: Any, discount: Any, **kwargs) -> None:
        # Can add in batches or serially.
        if isinstance(reward, list) or isinstance(reward, np.ndarray):
            num_to_add = len(reward)
            assert num_to_add > 1, "If inputting lists or arrays should have more than one timestep"
        else:
            num_to_add = 1

        if self._idx + num_to_add > self._capacity:
            # Add all we can at first, then add the rest later
            num_b4_wrap = self._capacity - self._idx
            self._add_to_buffer(
                get_from_batch(obs, 0, num_b4_wrap),
                get_from_batch(obs, 0, num_b4_wrap),
                reward[:num_b4_wrap],
                done[:num_b4_wrap],
                discount[:num_b4_wrap],
                **get_from_batch(kwargs, 0, num_b4_wrap),
            )
            self._add_to_buffer(
                get_from_batch(obs, num_b4_wrap, num_to_add),
                get_from_batch(obs, num_b4_wrap, num_to_add),
                reward[num_b4_wrap:],
                done[num_b4_wrap:],
                discount[num_b4_wrap:],
                **get_from_batch(kwargs, num_b4_wrap, num_to_add),
            )
        else:
            # Add the segment
            def add_to_buffer_helper(buffer, value):
                if isinstance(buffer, dict):
                    for k in buffer.keys():
                        add_to_buffer_helper(buffer[k], value[k])
                elif isinstance(buffer, np.ndarray):
                    buffer[self._idx : self._idx + num_to_add] = value
                else:
                    raise ValueError("Invalid buffer type given.")

            add_to_buffer_helper(self._obs_buffer, obs)
            add_to_buffer_helper(self._action_buffer, action)
            add_to_buffer_helper(self._reward_buffer, reward)
            add_to_buffer_helper(self._discount_buffer, discount)
            add_to_buffer_helper(self._done_buffer, done)

            for k, v in kwargs.items():
                if k not in self._info_buffers:
                    sample_value = get_from_batch(v, 0) if num_to_add > 1 else v
                    self._info_buffers[k] = np_dataset_alloc(sample_value, self._capacity)
                add_to_buffer_helper(self._info_buffers[k], v.copy())

            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def add_to_current_ep(self, key: str, value: Any, extend: bool = False):
        if isinstance(value, dict):
            for k, v in value.items():
                self.add_to_current_ep(key + "_" + k, v, extend=extend)
        else:
            if extend:
                self.current_ep[key].extend(value)
            else:
                self.current_ep[key].append(value)

    def add(
        self,
        obs: Any,
        action: Optional[Any] = None,
        reward: Optional[Any] = None,
        done: Optional[Any] = None,
        discount: Optional[Any] = None,
        next_obs: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # Make sure that if we are adding the first transition, it is consistent
        assert (action is None) == (reward is None) == (done is None) == (discount is None)
        if action is None:
            # construct dummy transition
            # This won't be sampled because we base everything off of the next_obs index
            action = self.action_space.sample()
            reward = 0.0
            done = False
            discount = 1.0

        # Direclty add to the buffer if we are serial and have setup the buffers.
        if not self.is_parallel and self.is_setup:
            # Deep copy to make sure we don't mess up references.
            if next_obs is None:
                self._add_to_buffer(copy.deepcopy(obs), copy.deepcopy(action), reward, done, discount, **kwargs)
                if self.cleanup:
                    return  # Exit if we clean up and don't save the buffer.
            else:
                assert (
                    action is not None
                ), "When using next obs must provide intermediate action, reward, done, discount"
                assert self.nstep == 1, "Adding individual transitions only supported with nstep = 1."
                # Add a single transition to the buffer.
                # We have to do two calls, one for the initial observation, and one for the next one
                self._add_to_buffer(
                    copy.deepcopy(obs), copy.deepcopy(action), reward, done, discount, **kwargs
                )  # these are dummy args
                self._add_to_buffer(copy.deepcopy(next_obs), copy.deepcopy(action), reward, True, discount, **kwargs)
                return  # We do not add to episode streams when we add individual transitions.

        assert next_obs is None, "Must add via episode streams in parallel mode."

        # If we don't have a current episode list, construct one.
        if not hasattr(self, "current_ep"):
            self.current_ep = collections.defaultdict(list)

        # Add values to the current episode
        extend = isinstance(reward, list) or isinstance(reward, np.ndarray)
        self.add_to_current_ep("obs", obs, extend)
        self.add_to_current_ep("action", action, extend)
        self.add_to_current_ep("reward", reward, extend)
        self.add_to_current_ep("done", done, extend)
        self.add_to_current_ep("discount", discount, extend)
        self.add_to_current_ep("kwargs", kwargs, extend)  # supports dict spaces

        if (isinstance(done, (bool, float, int)) and done) or (isinstance(done, (np.ndarray, list)) and done[-1]):
            # save the episode
            keys = list(self.current_ep.keys())
            assert len(self.current_ep["reward"]) == len(self.current_ep["done"])
            obs_keys = [key for key in keys if "obs" in key]
            action_keys = [key for key in keys if "action" in key]
            assert len(obs_keys) > 0, "No observation key"
            assert len(action_keys) > 0, "No action key"
            assert len(self.current_ep[obs_keys[0]]) == len(self.current_ep["reward"])
            # Commit to disk.
            ep_idx = self.num_episodes
            ep_len = len(self.current_ep["reward"])
            episode = {}
            for k, v in self.current_ep.items():
                first_value = v[0]
                if isinstance(first_value, (np.ndarray, np.generic)):
                    dtype = first_value.dtype
                elif isinstance(first_value, int):
                    dtype = np.int64
                elif isinstance(first_value, float):
                    dtype = np.float32
                elif isinstance(first_value, bool):
                    dtype = np.bool_
                episode[k] = np.array(v, dtype=dtype)
            # Delete the current_ep reference
            self.current_ep = collections.defaultdict(list)
            # Store the ep
            self.num_episodes += 1
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
            save_episode(episode, os.path.join(self.storage_path, ep_filename))

    def _load(self, path: str, cleanup: bool = False) -> None:
        ep_filenames = sorted([os.path.join(path, f) for f in os.listdir(path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_idx % self._num_workers != self._worker_id:
                continue
            if ep_filename in self._episode_filenames:
                break  # We found something we have already loaded
            if fetched_size + ep_len > self._capacity:
                break  # Cannot fetch more than the size of the replay buffer
            # Load the episode from disk
            try:
                episode = load_episode(ep_filename)
            except:
                continue
            # Add the episode to the buffer
            obs_keys = [key for key in episode.keys() if "obs" in key]
            action_keys = [key for key in episode.keys() if "action" in key]
            kwargs_keys = [key for key in episode.keys() if "kwargs" in key]
            obs = {k[len("obs_") :]: episode[k] for k in obs_keys} if len(obs_keys) > 1 else episode[obs_keys[0]]
            action = (
                {k[len("action_") :]: episode[k] for k in action_keys}
                if len(action_keys) > 1
                else episode[action_keys[0]]
            )
            kwargs = {k[len("kwargs_") :]: episode[k] for k in kwargs_keys}

            self._add_to_buffer(obs, action, episode["reward"], episode["done"], episode["discount"], **kwargs)
            # maintain the file list and storage
            self._episode_filenames.add(ep_filename)
            if cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

    def _get_one_idx(self, stack: int, pad: int) -> Union[int, np.ndarray]:
        # Add 1 for the first dummy transition
        idx = np.random.randint(0, self._size - self.nstep * stack) + 1
        done_idxs = idx + np.arange(self.nstep * (stack - pad)) - 1
        if np.any(self._done_buffer[done_idxs]):
            # If the episode is done at any point in the range, we need to sample again!
            # Note that we removed the pad length, as we can check the padding later
            return self._get_one_idx(stack, pad)
        if stack > 1:
            idx = idx + np.arange(stack) * self.nstep
        return idx

    def _get_many_idxs(self, batch_size: int, stack: int, pad: int, depth: int = 0) -> np.ndarray:
        idxs = np.random.randint(0, self._size - self.nstep * stack, size=int(self.sample_multiplier * batch_size)) + 1

        done_idxs = np.expand_dims(idxs, axis=-1) + np.arange(self.nstep * (stack - pad)) - 1
        valid = np.logical_not(
            np.any(self._done_buffer[done_idxs], axis=-1)
        )  # Compute along the done axis, not the index axis.

        valid_idxs = idxs[valid == True]  # grab only the idxs that are still valid.
        if len(valid_idxs) < batch_size and depth < 100:  # try a max of 100 times
            print(
                "[research ReplayBuffer] Buffer Sampler did not recieve batch_size number of valid indices. Consider"
                " increasing sample_multiplier."
            )
            return self._get_many_idxs(batch_size, stack, pad, depth=depth + 1)
        idxs = valid_idxs[:batch_size]  # Return the first [:batch_size] of them.
        if stack > 1:
            stack_idxs = np.arange(stack) * self.nstep
            idxs = np.expand_dims(idxs, axis=-1) + stack_idxs
        return idxs

    def _compute_mask(self, idxs: np.ndarray) -> np.ndarray:
        # Check the validity via the done buffer to determine the padding mask
        mask = np.zeros(idxs.shape, dtype=np.bool_)
        for i in range(self.nstep):
            mask = (
                mask + self._done_buffer[idxs + (i - 1)]
            )  # Subtract one when checking for parity with index sampling.
        # Now set everything past the first true to be true
        mask_inds = np.argmax(mask, axis=-1)

        if len(mask.shape) == 1:  # Single data point case
            mask_inds = mask.shape[-1] if np.sum(mask, axis=-1) == 0 else mask_inds
            mask[mask_inds:] = True
        elif len(mask.shape) == 2:  # Multiple data point case
            mask_inds[np.sum(mask, axis=-1) == 0] = mask.shape[-1]
            for i in range(mask.shape[0]):
                mask[i, mask_inds[i] :] = True
        else:
            raise ValueError("Mask was an invalid size")
        return mask

    def sample(self, batch_size: Optional[int] = None, stack: int = 1, pad: int = 0) -> Dict:
        if self._size <= self.nstep * stack + 2:
            return {}
        # NOTE: one small bug is that we won't end up being able to sample segments that span
        # Across the barrier of the replay buffer. We lose 1 to self.nstep transitions.
        # This is only a problem if we keep the capacity too low.
        if batch_size > 1:
            idxs = self._get_many_idxs(batch_size, stack, pad)
        else:
            idxs = self._get_one_idx(stack, pad)
        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1

        obs = (
            {k: v[obs_idxs] for k, v in self._obs_buffer.items()}
            if isinstance(self._obs_buffer, dict)
            else self._obs_buffer[obs_idxs]
        )
        action = (
            {k: v[idxs] for k, v in self._action_buffer.items()}
            if isinstance(self._action_buffer, dict)
            else self._action_buffer[idxs]
        )
        reward = np.zeros_like(self._reward_buffer[idxs])
        discount = np.ones_like(self._discount_buffer[idxs])
        for i in range(self.nstep):
            reward += discount * self._reward_buffer[idxs + i]
            discount *= self._discount_buffer[idxs + i] * self.discount

        kwargs = {k: v[next_obs_idxs] for k, v in self._info_buffers.items()}
        if self.use_next_obs:
            kwargs["next_obs"] = (
                {k: v[next_obs_idxs] for k, v in self._obs_buffer.items()}
                if isinstance(self._obs_buffer, dict)
                else self._obs_buffer[next_obs_idxs]
            )

        batch = dict(obs=obs, action=action, reward=reward, discount=discount, **kwargs)
        if pad > 0:
            batch["mask"] = self._compute_mask(idxs)

        return batch

    def __iter__(self):
        self.setup()
        while True:
            yield self.sample(batch_size=self.batch_size, stack=self.stack, pad=self.pad)
            if self.is_parallel:
                self._samples_since_last_load += 1
                if self._samples_since_last_load >= self.fetch_every:
                    self._load(self.storage_path, cleanup=self.cleanup)
                    self._samples_since_last_load = 0


class HindsightRepalyBuffer(ReplayBuffer):

    """
    An efficient class for replay buffer sampling. For efficiency, one must
    pass in a version of the desired reward function that works on batches of data.

    TODO: Documentation
    """

    def __init__(
        self,
        *args,
        reward_fn: Optional[Callable] = None,
        discount_fn: Optional[Callable] = None,
        goal_key: str = "desired_goal",
        achieved_key: str = "achieved_goal",
        strategy: str = "future",
        relabel_fraction: float = 0.5,
        max_lookahead: int = 500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reward_fn = reward_fn
        self.discount_fn = discount_fn
        self.goal_key = goal_key
        self.achieved_key = achieved_key
        self.strategy = strategy
        self.max_lookahead = max_lookahead
        self.relabel_fraction = relabel_fraction
        assert isinstance(self.observation_space, gym.spaces.Dict), "HER Replay Buffer depends on Dict Spaces."

    def sample(self, batch_size: Optional[int] = None, stack: int = 1, pad: int = 0) -> Dict:
        if self._size <= self.nstep * stack + 2:
            return {}
        # NOTE: one small bug is that we won't end up being able to sample segments that span
        # Across the barrier of the replay buffer. We lose 1 to self.nstep transitions.
        # This is only a problem if we keep the capacity too low.
        if batch_size > 1:
            idxs = self._get_many_idxs(batch_size, stack, pad)
        else:
            idxs = self._get_one_idx(stack, pad)
            # Convert idxs to numpy array for HER sampling
            idxs = np.array([idxs])
        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1

        # copies are added so we do not modify the original buffers when relabeling
        obs = {k: v[obs_idxs].copy() for k, v in self._obs_buffer.items()}
        action = (
            {k: v[idxs] for k, v in self._action_buffer.items()}
            if isinstance(self._action_buffer, dict)
            else self._action_buffer[idxs]
        )
        kwargs = {k: v[next_obs_idxs] for k, v in self._info_buffers.items()}

        her_idxs = np.where(np.random.uniform(size=batch_size) < self.relabel_fraction)

        # Note that we carefully have to handle padding, as we can sample some invalid indexes on stacking.
        last_idxs = (idxs[..., -1] if stack > 1 else idxs)[her_idxs]  # Get the last indexes of the sample.
        start_idxs = last_idxs - pad * self.nstep  # adjust backwards for padding
        # Get indexes of all values to check ahead. The shape is (B, max_lookahead + pad*self.nstep)
        lookahead_idxs = np.expand_dims(start_idxs, axis=-1) + np.arange(self.max_lookahead + pad * self.nstep)
        lookahead_idxs = np.clip(lookahead_idxs, 0, self._size - 1)  # Avoid going over or under
        # Look at the done buffer to see where the episode finishes
        sample_limits = np.argmax(self._done_buffer[lookahead_idxs], axis=-1)
        sample_limits[np.sum(self._done_buffer[lookahead_idxs], axis=-1) == 0] = self.max_lookahead
        end_idxs = start_idxs + sample_limits  # get the last valid point
        end_idxs = np.minimum(end_idxs, self._size)  # Make sure we trim down so we don't go over the end of the buffer.
        start_idxs = np.minimum(
            last_idxs, end_idxs
        )  # Recompute start index so we start at the last observation, or the end index in case of a middle terminal

        if self.strategy == "last":  # Just use the last transtion of the episode or the max look ahead
            goal_idxs = end_idxs
        elif self.strategy == "next":  # Use the last valid transition of the stack.
            goal_idxs = last_idxs
        elif self.strategy == "future":  # Sample a future state
            goal_idxs = np.random.randint(low=start_idxs, high=end_idxs + 1)
        else:
            raise ValueError("Invalid Strategy Selected.")

        # Perform relabeling on the observations
        if len(idxs.shape) == 2:
            goal_idxs = np.expand_dims(goal_idxs, axis=-1)

        obs[self.goal_key][her_idxs] = self._obs_buffer[self.achieved_key][goal_idxs]
        if self.use_next_obs:
            next_obs = {k: v[next_obs_idxs].copy() for k, v in self._obs_buffer.items()}
            next_obs[self.goal_key][her_idxs] = self._obs_buffer[self.achieved_key][goal_idxs]
            kwargs["next_obs"] = next_obs

        # Compute the reward
        reward = np.zeros_like(self._reward_buffer[idxs], dtype=np.float32)
        discount = np.ones_like(self._discount_buffer[idxs], dtype=np.float32)
        for i in range(self.nstep):
            # Get the relabeled observations
            desired = self._obs_buffer[self.goal_key][idxs + i].copy()
            desired[her_idxs] = self._obs_buffer[self.achieved_key][goal_idxs]
            achieved = self._obs_buffer[self.achieved_key][idxs + i]
            # Compute the reward and discounts
            reward += discount * self.reward_fn(achieved, desired)
            step_discount = self.discount_fn(achieved, desired) if self.discount_fn is not None else 1.0
            discount *= step_discount * self.discount

        batch = dict(obs=obs, action=action, reward=reward, discount=discount, **kwargs)
        if pad > 0:
            batch["mask"] = self._compute_mask(idxs)

        # We need to unsqueeze everything if we used batch size == 1
        if batch_size == 1:
            batch = squeeze(batch, 0)

        return batch
