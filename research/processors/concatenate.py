import torch
import gym
from .base import Processor

class ConcatenateProcessor(Processor):

    def __init__(self, observation_space, action_space, concat_obs=True, concat_action=True):
        super().__init__(observation_space, action_space)
        self.concat_action = concat_action and isinstance(action_space, gym.spaces.Dict)
        if self.concat_action:
            self.action_order = list(action_space.keys())
        self.concat_obs = concat_obs and isinstance(observation_space, gym.spaces.Dict)
        if self.concat_obs:
            self.obs_order = list(observation_space.keys())

    def forward(self, batch):
        if self.concat_action:
            batch['action'] = torch.cat([batch['action'][k] for k in self.action_order], dim=-1)
        if self.concat_obs:
            batch['obs'] = torch.cat([batch['obs'][k] for k in self.obs_order], dim=-1)
        return batch
