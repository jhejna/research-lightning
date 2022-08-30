from functools import partial
from typing import List, Optional, Type

import gym
import torch
from torch import distributions, nn
from torch.nn import functional as F

from .common import MLP, EnsembleMLP, LinearEnsemble


def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
    ):
        assert len(hidden_layers) > 1, "Must have at least one hidden layer for a shared MLP Extractor"
        self.mlp = MLP(observation_space.shape[0], hidden_layers[-1], hidden_layers=hidden_layers[:-1], act=act)
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0


class ContinuousMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        num_q_fns: int = 2,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.num_q_fns = num_q_fns
        if self.num_q_fns > 1:
            self.q = EnsembleMLP(
                observation_space.shape[0] + action_space.shape[0],
                1,
                ensemble_size=num_q_fns,
                hidden_layers=hidden_layers,
                act=act,
            )
        else:
            self.q = MLP(observation_space.shape[0] + action_space.shape[0], 1, hidden_layers=hidden_layers, act=act)

        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, action), dim=-1)
        q = self.q(x).squeeze(-1)  # Remove the last dim
        if self.num_q_fns == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q


class DiscreteMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.q = MLP(observation_space.shape[0], action_space.n, hidden_layers=hidden_layers, act=act)
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(obs)


class MLPValue(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.mlp = MLP(observation_space.shape[0], 1, hidden_layers=hidden_layers, act=act)
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)  # Return only scalar values, no final dim


class ContinuousMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        output_act: Optional[Type[nn.Module]] = None,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
    ):
        super().__init__()
        self.mlp = MLP(
            observation_space.shape[0],
            action_space.shape[0],
            hidden_layers=hidden_layers,
            act=act,
            output_act=output_act,
        )
        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class SquashedNormal(distributions.TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        self._loc = loc
        self.scale = scale
        self.base_dist = distributions.Normal(loc, scale)
        transforms = [distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self) -> torch.Tensor:
        loc = self._loc
        for transform in self.transforms:
            loc = transform(loc)
        return loc


class DiagonalGaussianMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: List[int] = [256, 256],
        act: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        log_std_bounds: List[int] = [-5, 2],
        state_dependent_log_std: bool = True,
    ):
        super().__init__()
        self.state_dependent_log_std = state_dependent_log_std
        self.log_std_bounds = log_std_bounds
        if log_std_bounds is not None:
            assert log_std_bounds[0] < log_std_bounds[1]

        if self.state_dependent_log_std:
            self.mlp = MLP(
                observation_space.shape[0],
                2 * action_space.shape[0],
                hidden_layers=hidden_layers,
                act=act,
                output_act=None,
            )
        else:
            self.mlp = MLP(
                observation_space.shape[0], action_space.shape[0], hidden_layers=hidden_layers, act=act, output_act=None
            )
            self.log_std = nn.Parameter(
                torch.zeros(action_space.shape[0]), requires_grad=True
            )  # initialize a single parameter vector

        if ortho_init:
            self.apply(partial(weight_init, gain=float(ortho_init)))  # use the fact that True converts to 1.0
            if output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=output_gain))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.state_dependent_log_std:
            mu, log_std = self.mlp(obs).chunk(2, dim=-1)
        else:
            mu, log_std = self.mlp(obs), self.log_std

        if self.log_std_bounds is not None:
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            dist_class = SquashedNormal
        else:
            dist_class = distributions.Normal

        dist = dist_class(mu, log_std.exp())
        return dist
