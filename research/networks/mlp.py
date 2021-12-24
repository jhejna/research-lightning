import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from gym import spaces

from . common import MLP as mlp_extractor

class MLP(nn.Module):
    '''
    A simple MLP classification network.
    Assumes that the input has already been flattened.
    '''
    def __init__(self, observation_space, action_space, hidden_layers=[256, 256], act=nn.ReLU):
        super().__init__()
        assert isinstance(observation_space, spaces.Box), "MLP supports box spaces."
        assert isinstance(action_space, spaces.Discrete)
        input_dim = np.prod(observation_space.shape)
        output_dim = action_space.n
        self.mlp = mlp_extractor(input_dim, output_dim, hidden_layers=hidden_layers, act=act, output_act=None)

    def forward(self, x):
        return self.mlp(x)

