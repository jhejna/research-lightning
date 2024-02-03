from functools import partial

import gym
import numpy as np
import torch

import research

"""
There are two special network functions used by research lightning
1. output_space - this is used to give the observation_space to different networks in a container group
2. compile - this is used when torch.compile is called.
"""


def reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class ModuleContainer(torch.nn.Module):
    CONTAINERS = []

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs) -> None:
        super().__init__()
        # save the classes and containers
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}

        output_space = observation_space
        for container in self.CONTAINERS:
            module_class = kwargs.get(container + "_class", torch.nn.Identity)
            module_class = vars(research.networks)[module_class] if isinstance(module_class, str) else module_class
            if module_class is torch.nn.Identity:
                module_kwargs = dict()
            else:
                module_kwargs = base_kwargs.copy()
                module_kwargs.update(kwargs.get(container + "_kwargs", dict()))
            # Create the module, and attach it to self
            module = module_class(output_space, action_space, **module_kwargs)
            setattr(self, container, module)

            # Set a reset function
            setattr(self, "reset_" + container, partial(self._reset, container))

            if hasattr(getattr(self, container), "output_space"):
                # update the output space
                output_space = getattr(self, container).output_space

        # Done creating all sub-modules.

    @classmethod
    def create_subset(cls, containers):
        assert all([container in cls.CONTAINERS for container in containers])
        name = "".join([container.capitalize() for container in containers]) + "Subset"
        return type(name, (ModuleContainer,), {"CONTAINERS": containers})

    def _reset(self, container: str) -> None:
        module = getattr(self, container)
        with torch.no_grad():
            module.apply(reset)

    def compile(self, **kwargs):
        for container in self.CONTAINERS:
            attr = getattr(self, container)
            if type(attr).forward == torch.nn.Module.forward:
                assert hasattr(attr, "compile"), (
                    "container " + container + " is nn.Module without forward() but didn't define `compile`."
                )
                attr.compile(**kwargs)
            else:
                setattr(self, container, torch.compile(attr, **kwargs))

    def forward(self, x):
        # Use all of the modules in order
        for container in self.CONTAINERS:
            x = getattr(self, container)(x)
        return x


class MultiEncoder(torch.nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Dict)
        base_kwargs = {k: v for k, v in kwargs.items() if not k.endswith("_class") and not k.endswith("_kwargs")}
        # parse unique modalities from args that are passed with "class"
        self.obs_keys = sorted([k[: -len("_class")] for k in kwargs if k.endswith("_class")])
        assert all([k in observation_space for k in self.obs_keys])

        modules = dict()
        for k in self.obs_keys:
            # Build the modules
            module_class = kwargs[k + "_class"]
            module_class = vars(research.networks)[module_class] if isinstance(module_class, str) else module_class
            module_kwargs = base_kwargs.copy()
            module_kwargs.update(kwargs.get(k + "_kwargs", dict()))
            module = module_class(observation_space[k], action_space, **module_kwargs)
            modules[k] = module

        # register all the modules
        self.modules = torch.nn.ModuleDict(modules)

        # compute the output space
        output_dim = 0
        for k in self.obs_keys:
            output_shape = self.modules[k].output_space
            assert len(output_shape) == 1
            output_dim += output_shape[0]

        self.output_dim = output_dim

    @property
    def output_space(self) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.output_dim,), dtype=np.float32)

    def forward(self, obs):
        return torch.cat([self.modules[k](obs[k]) for k in self.obs_keys], dim=-1)


class ActorCriticPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic"]


class ActorCriticValuePolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "critic", "value"]


class ActorValuePolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor", "value"]


class ActorPolicy(ModuleContainer):
    CONTAINERS = ["encoder", "actor"]
