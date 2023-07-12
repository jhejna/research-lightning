import itertools
from typing import Any, Dict, Tuple, Type, Union

import gym
import numpy as np
import torch

from research.networks.base import ActorCriticPolicy

from ..off_policy_algorithm import OffPolicyAlgorithm

"""
Note: this implementation is untested!
"""


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=None):
        shape = self._extended_shape(torch.Size() if sample_shape is None else sample_shape)
        eps = torch.distributions.utils._standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class DRQV2(OffPolicyAlgorithm):
    """
    NOTE: DrQv2 implementation is untested and not verified yet.
    Please do not use this implementation for baseline comparisons.
    """

    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        tau: float = 0.005,
        critic_freq: int = 1,
        actor_freq: int = 1,
        target_freq: int = 1,
        init_steps: int = 1000,
        std_schedule: Tuple[float, float, int] = (1.0, 0.1, 500000),
        noise_clip: float = 0.3,
        **kwargs,
    ):
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        # Save extra parameters
        self.tau = tau
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.std_schedule = std_schedule
        self.init_steps = init_steps
        self.noise_clip = noise_clip

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        self.optim["actor"] = self.optim_class(self.network.actor.parameters(), **self.optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
        self.optim["critic"] = self.optim_class(critic_params, **self.optim_kwargs)

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            mu = self.network.actor(batch["next_obs"])
            std = self._get_std() * torch.ones_like(mu)
            next_action = TruncatedNormal(mu, std).sample(clip=self.noise_clip)
            target_qs = self.target_network.critic(batch["next_obs"], next_action)
            target_v = torch.min(target_qs, dim=0)[0]
            target_q = batch["reward"] + batch["discount"] * target_v

        qs = self.network.critic(batch["obs"], batch["action"])
        q_loss = (
            torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean(dim=-1).sum()
        )  # averages over the ensemble. No for loop!

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())

    def _update_actor(self, batch: Dict) -> Dict:
        obs = batch["obs"].detach()  # Detach the encoder so it isn't updated.
        mu = self.network.actor(obs)
        std = self._get_std() * torch.ones_like(mu)
        dist = TruncatedNormal(mu, std)
        action = dist.sample(clip=self.noise_clip)
        log_prob = dist.log_prob(action).sum(dim=-1)

        q1, q2 = self.network.critic(obs, action)
        q = torch.min(q1, q2)
        actor_loss = -q.mean()

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        return dict(actor_loss=actor_loss.item(), log_prob=log_prob.mean().item())

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            mu = self.predict(batch)
            mu = torch.as_tensor(mu, device=self.device)
            init, final, duration = self.std_schedule
            mix = np.clip(step / duration, 0.0, 1.0)
            std = (1.0 - mix) * init + mix * final
            std = std * torch.ones_like(mu)
            action = TruncatedNormal(mu, std).sample(clip=None).cpu().numpy()
        return action

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if "obs" not in batch or step < self.random_steps:
            return all_metrics

        batch["obs"] = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        if step % self.critic_freq == 0:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if step % self.actor_freq == 0:
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        # NOTE: The original DrQv2 does not use a target encoder. We use one here.
        if step % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _predict(self, batch: Any) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            return self.network.actor(z)
