import random
from typing import Optional

import h5py
import numpy as np
import torch

from research.utils import utils

from .replay_buffer.buffer import ReplayBuffer


class RobomimicDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the RoboMimicDatasets into a ReplayBuffer
    """

    def __init__(self, *args, action_eps: Optional[float] = 1e-5, train=True, **kwargs):
        self.action_eps = action_eps
        self.train = train
        super().__init__(*args, **kwargs)

    def _data_generator(self):
        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        f = h5py.File(self.path, "r")

        if self.train:
            # Extract the training demonstrations
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/train"][:])]
        else:
            # Extract the validation
            demos = [elem.decode("utf-8") for elem in np.array(f["mask/train"][:])]

        # Assign demos to each worker
        demos = sorted(demos)  # Deterministic ordering
        demos = demos[worker_id::num_workers]
        # Shuffle the data ordering
        random.shuffle(demos)

        for _i, demo in enumerate(demos):
            # Get obs from the start to the end.
            obs = utils.get_from_batch(f["data"][demo]["obs"], 0, len(f["data"][demo]["dones"]))
            last_obs = utils.unsqueeze(utils.get_from_batch(f["data"][demo]["next_obs"], -1), 0)
            obs = utils.concatenate(obs, last_obs)
            obs = utils.remove_float64(obs)

            dummy_action = np.expand_dims(self.dummy_action, axis=0)
            action = np.concatenate((dummy_action, f["data"][demo]["actions"]), axis=0)
            action = utils.remove_float64(action)

            if self.action_eps is not None:
                lim = 1 - self.action_eps
                action = np.clip(action, -lim, lim)

            reward = np.concatenate(([0], f["data"][demo]["rewards"]), axis=0)
            reward = utils.remove_float64(reward)

            done = np.concatenate(([0], f["data"][demo]["dones"]), axis=0).astype(np.bool_)
            done[-1] = True

            discount = (1 - done).astype(np.float32)

            obs_len = obs[next(iter(obs.keys()))].shape[0]
            assert all([len(obs[k]) == obs_len for k in obs.keys()])
            assert obs_len == len(action) == len(reward) == len(done) == len(discount)

            yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)

        f.close()  # Close the file handler.
