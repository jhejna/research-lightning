import itertools
from typing import Any, Dict, Optional, Type

import diffusers
import numpy as np
import torch

from research.networks.base import ActorCriticValuePolicy
from research.utils import utils

from ..off_policy_algorithm import OffPolicyAlgorithm


def iql_loss(pred, target, expectile=0.5):
    err = target - pred
    weight = torch.abs(expectile - (err < 0).float())
    return weight * torch.square(err)


class IDQL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        target_freq: int = 1,
        expectile: Optional[float] = None,
        beta: float = 1,
        noise_scheduler=diffusers.schedulers.DDIMScheduler,
        noise_scheduler_kwargs: Optional[Dict] = None,
        num_inference_steps: Optional[int] = 10,
        num_samples: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.target_freq = target_freq
        self.expectile = expectile
        self.beta = beta
        self.num_samples = num_samples
        assert isinstance(self.network, ActorCriticValuePolicy)
        noise_scheduler_kwargs = {} if noise_scheduler_kwargs is None else noise_scheduler_kwargs
        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        if num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = num_inference_steps

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)

        self.target_network = self.network.create_subset(["encoder", "critic"])(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        )
        # Delete the unneeded things from the target network.
        del self.target_network.encoder
        self.target_network = self.target_network.to(self.device)

        # Set up the target network.
        self.target_network.critic.load_state_dict(self.network.critic.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        actor_groups = utils.create_optim_groups(actor_params, self.optim_kwargs)
        # NOTE: Optim class only affects the Actor.
        self.optim["actor"] = self.optim_class(actor_groups)

        # Remove weight decay from critics.
        value_optim_kwargs = self.optim_kwargs.copy()
        if "weight_decay" in value_optim_kwargs:
            del value_optim_kwargs["weight_decay"]
        self.optim["critic"] = torch.optim.Adam(self.network.critic.parameters(), **value_optim_kwargs)
        self.optim["value"] = torch.optim.Adam(self.network.value.parameters(), **value_optim_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # We use the online encoder for everything in this IQL implementation
        # That is because we need to use the current obs for the target critic and online value.
        # This is done by default in DrQv2.
        batch["obs"] = self.network.encoder(batch["obs"])

        with torch.no_grad():
            batch["next_obs"] = self.network.encoder(batch["next_obs"])

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

        # Update the actor, just with BC. We will sample from it later using re-weighting.
        bs = batch["action"].shape[0]
        noise = torch.randn_like(batch["action"])
        timesteps = torch.randint(
            low=0, high=self.scheduler.config.num_train_timesteps, size=(bs,), device=self.device
        ).long()
        noisy_actions = self.scheduler.add_noise(batch["action"], noise, timesteps)
        (~batch["mask"]).float()

        noise_pred = self.network.actor(noisy_actions, timesteps, cond=batch["obs"])
        actor_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean(dim=2)

        if "mask" in batch:
            mask = (~batch["mask"]).float()
            actor_loss = actor_loss * mask
            size = mask.sum()
        else:
            size = actor_loss.numel()
        actor_loss = actor_loss.sum() / size

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

        return dict(
            q_loss=q_loss.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            q=qs.mean().item(),
        )

    def _predict(self, batch: Dict):
        with torch.no_grad():
            obs = self.network.encoder(batch["obs"])
            B, D = obs.shape
            obs = obs.unsqueeze(1)
            obs = obs.expand(B, self.num_samples, D)

            noisy_actions = torch.randn(B, self.num_samples, self.processor.action_space.shape[0], device=self.device)
            self.scheduler.set_timesteps(self.num_inference_steps)
            for timestep in self.scheduler.timesteps:
                noise_pred = self.network.actor(noisy_actions, timestep.unsqueeze(0).to(self.device), cond=obs)
                noisy_actions = self.scheduler.step(
                    model_output=noise_pred, timestep=timestep, sample=noisy_actions
                ).prev_sample

            # Now we have finished generating the actions, now we need to figure out their weights
            v = self.network.value(obs).mean(dim=0)
            q = torch.min(self.target_network.critic(obs, noisy_actions), dim=0)[0]
            adv = q - v  # Shape (B, self.num_samples)
            expectile_weights = torch.where(adv > 0, self.expectile, 1 - self.expectile)
            sample_idx = torch.multinomial(expectile_weights / expectile_weights.sum(), 1)  # (B, 1)
            actions = torch.gather(noisy_actions, dim=1, index=sample_idx)

        return actions

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False)
        return action[0]  # return the first one.
