import torch
import gym
from .base import Processor

class ConcatenateProcessor(Processor):

    def __init__(self, observation_space, action_space, concat_obs=True, concat_action=True):
        super().__init__(observation_space, action_space)
        self.concat_action = concat_action and isinstance(action_space, gym.spaces.Dict)
        self.concat_obs = concat_obs and isinstance(observation_space, gym.spaces.Dict)

    def forward(self, batch):
        if self.concat_action:
            batch['action'] = torch.cat(list(batch['action'].values()), dim=-1)
        if self.concat_obs:
            batch['obs'] = torch.cat(list(batch['obs'].values()), dim=-1)
        return batch
