import gc
import os
import pickle
from typing import List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from research.utils.utils import remove_float64

from .replay_buffer.buffer import ReplayBuffer


class WGCSLDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the WGCSL buffers into a HindsightReplayBuffer
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Union[str, Tuple[str]] = (),
        percents: Optional[List[float]] = None,
        train: bool = True,
        terminal_threshold: Optional[float] = None,
        **kwargs,
    ):  
        assert path is not None
        if isinstance(path, str):
            path = [path]
        percents = [1.0] * len(path) if percents is None else percents
        self.percents = percents
        self.train = train
        self.terminal_threshold = terminal_threshold
        super().__init__(observation_space, action_space, path=path, **kwargs)
        assert not self.distributed, "WGCSL datasets do not support distributed training."

    def _data_generator(self):
        for path, percent in zip(self.path, self.percents):
            with open(path, "rb") as f:
                data = pickle.load(f)
            num_ep = data["ag"].shape[0]
            # Add the episodes
            ep_idxs = range(int(num_ep * percent)) if self.train else range(num_ep - int(num_ep * percent), num_ep)
            for i in ep_idxs:
                # We need to make sure we appropriately handle the dummy transition
                obs = dict(achieved_goal=data["ag"][i].copy())
                if "o" in data:
                    obs["observation"] = data["o"][i].copy()
                if "g" in data:
                    goal = data["g"][i]
                    obs["desired_goal"] = np.concatenate((goal[:1], goal), axis=0)
                obs = remove_float64(obs)
                dummy_action = np.expand_dims(self.dummy_action, axis=0)
                action = np.concatenate((dummy_action, data["u"][i]), axis=0)
                action = remove_float64(action)

                # If we have a terminal threshold compute and store the horizon
                if self.terminal_threshold is not None:
                    goal_distance = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"], axis=-1)
                    done = (goal_distance < self.terminal_threshold).astype(np.bool_)
                else:
                    done = np.zeros(action.shape[0], dtype=np.bool_)
                done[-1] = True  # Add the episode delineation
                discount = np.ones(action.shape[0])  # Gets recomputed with HER
                reward = np.zeros(action.shape[0])  # Gets recomputed with HER
                assert len(obs["achieved_goal"]) == len(action) == len(reward) == len(done) == len(discount)
                yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)
                
            # Explicitly delete the data objects to save memory
            del data
            del obs
            del action
            del reward
            del done
            del discount
            gc.collect()
