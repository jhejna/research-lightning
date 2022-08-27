from typing import Any, Dict, Optional, Type, Union

import gym
import torch
from torch import nn

import research


class ActorCriticPolicy(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_class: Type[nn.Module],
        critic_class: Type[nn.Module],
        encoder_class: Optional[Type[nn.Module]] = None,
        actor_kwargs: Dict = {},
        critic_kwargs: dict = {},
        encoder_kwargs: Dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        # Update all dictionaries with the generic kwargs
        self.action_space = action_space
        self.observation_space = observation_space

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)

        self.encoder_class, self.encoder_kwargs = encoder_class, encoder_kwargs
        self.actor_class, self.actor_kwargs = actor_class, actor_kwargs
        self.critic_class, self.critic_kwargs = critic_class, critic_kwargs

        self.reset_encoder()
        self.reset_actor()
        self.reset_critic()

    def reset_encoder(self, device: Optional[Union[str, torch.device]] = None) -> None:
        encoder_class = (
            vars(research.networks)[self.encoder_class] if isinstance(self.encoder_class, str) else self.encoder_class
        )
        if encoder_class is not None:
            self._encoder = encoder_class(self.observation_space, self.action_space, **self.encoder_kwargs)
        else:
            self._encoder = nn.Identity()
        if device is not None:
            self._encoder = self._encoder.to(device)

    def reset_actor(self, device: Optional[Union[str, torch.device]] = None) -> None:
        observation_space = (
            self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        )
        actor_class = (
            vars(research.networks)[self.actor_class] if isinstance(self.actor_class, str) else self.actor_class
        )
        self._actor = actor_class(observation_space, self.action_space, **self.actor_kwargs)
        if device is not None:
            self._actor = self._actor.to(self.device)

    def reset_critic(self, device: Optional[Union[str, torch.device]] = None) -> None:
        observation_space = (
            self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        )
        critic_class = (
            vars(research.networks)[self.critic_class] if isinstance(self.critic_class, str) else self.critic_class
        )
        self._critic = critic_class(observation_space, self.action_space, **self.critic_kwargs)
        if device is not None:
            self._critic = self._critic.to(device)

    @property
    def actor(self) -> nn.Module:
        return self._actor

    @property
    def critic(self) -> nn.Module:
        return self._critic

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    def predict(self, obs: Any, **kwargs) -> Any:
        obs = self._encoder(obs)
        if hasattr(self._actor, "predict"):
            return self._actor.predict(obs, **kwargs)
        else:
            return self._actor(obs)
