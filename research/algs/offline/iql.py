import itertools
from typing import Any, Dict, Optional, Type

import numpy as np
import torch

from research.networks.base import ActorCriticValuePolicy

from ..off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class IQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.clip_score = clip_score
        assert isinstance(self.network, ActorCriticValuePolicy)

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
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        self.optim["actor"] = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim["critic"] = self.optim_class(self.network.critic.parameters(), **self.optim_kwargs)
        self.optim["value"] = self.optim_class(self.network.value.parameters(), **self.optim_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # We use the online encoder for everything in this IQL implementation
        # That is because we need to use the current obs for the target critic and online value.
        # This is done by default in DrQv2.
        with torch.no_grad():
            batch["next_obs"] = self.network.encoder(batch["next_obs"])
        batch["obs"] = self.network.encoder(batch["obs"])

        # First compute the value loss
        with torch.no_grad():
            target_q = self.target_network.critic(batch["obs"], batch["action"])
            target_q = torch.min(target_q, dim=0)[0]
        vs = self.network.value(batch["obs"].detach())  # Always detach for value learning
        v_loss = iql_loss(vs, target_q.expand(vs.shape[0], -1), self.expectile).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        v_loss.backward()
        self.optim["value"].step()

        # Next, compute the critic loss
        with torch.no_grad():
            next_vs = self.network.value(batch["next_obs"])
            next_v = torch.min(next_vs, dim=0)[0]
            target = batch["reward"] + batch["discount"] * next_v
        qs = self.network.critic(batch["obs"].detach(), batch["action"])
        q_loss = torch.nn.functional.mse_loss(qs, target.expand(qs.shape[0], -1), reduction="none").mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim["critic"].step()

        # Next, update the actor. We detach and use the old value, v for computational efficiency
        # though the JAX IQL recomputes it, while Pytorch IQL versions do not.
        with torch.no_grad():
            adv = target_q - torch.min(vs, dim=0)[0]
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(batch["obs"])
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(batch["action"])
        elif torch.is_tensor(dist):
            assert dist.shape == batch["action"].shape
            bc_loss = torch.nn.functional.mse_loss(dist, batch["action"], reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        actor_loss = (exp_adv * bc_loss).mean()

        # Update the networks. These are done in a stack to support different grad options for the encoder.
        self.optim["actor"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim["actor"].step()

        if step % self.target_freq == 0:
            with torch.no_grad():
                # Only run on the critic and encoder, those are the only weights we update.
                for param, target_param in zip(
                    self.network.critic.parameters(), self.target_network.critic.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
            advantage=adv.mean().item(),
        )

    def _predict(self, batch: Dict, sample: bool = False, noise: float = 0.0) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.base_dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            if noise > 0.0:
                action = action + noise * torch.randn_like(action)
            action = action.clamp(*self.action_range)
        return action

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
