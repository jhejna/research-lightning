import itertools
from typing import Any, Dict, Type

import gym
import numpy as np
import torch

from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_device, to_tensor

from ..off_policy_algorithm import OffPolicyAlgorithm


class TD3(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau=0.005,
        policy_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        critic_freq=1,
        actor_freq=2,
        target_freq=2,
        average_actor_q=True,
        bc_coeff=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        # Save extra parameters
        self.tau = tau
        self.policy_noise = policy_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.average_actor_q = average_actor_q
        self.bc_coeff = bc_coeff
        self.action_range = (self.processor.action_space.low, self.processor.action_space.high)
        self.action_range_tensor = to_device(to_tensor(self.action_range), self.device)

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

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False)
        action += self.policy_noise * np.random.randn(action.shape[0])
        return action

    def _update_critic(self, batch: Dict) -> Dict:
        with torch.no_grad():
            noise = (torch.randn_like(batch["action"]) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.target_network.actor(batch["next_obs"])
            noisy_next_action = (next_action + noise).clamp(*self.action_range_tensor)
            target_q = self.target_network.critic(batch["next_obs"], noisy_next_action)
            target_q = torch.min(target_q, dim=0)[0]
            target_q = batch["reward"] + batch["discount"] * target_q

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
        action = self.network.actor(obs)
        qs = self.network.critic(obs, action)
        if self.average_actor_q:
            q = qs.mean(dim=0)  # average the qs over the ensemble
        else:
            q = qs[0]  # Take only the first Q function
        actor_loss = -q.mean()

        if self.bc_coeff > 0.0:
            bc_loss = torch.nn.functional.mse_loss(action, batch["action"])
            actor_loss = actor_loss + self.bc_coeff * bc_loss

        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        return dict(actor_loss=actor_loss.item())

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if "obs" not in batch:
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

        if step % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _predict(self, batch: Any) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            return self.network.actor(z)
