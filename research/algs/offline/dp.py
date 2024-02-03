from typing import Any, Dict, Optional

import diffusers
import numpy as np
import torch

from research.networks.base import ActorPolicy
from research.utils import utils

from ..off_policy_algorithm import OffPolicyAlgorithm


class DiffusionPolicy(OffPolicyAlgorithm):
    """
    BC Implementation.
    Uses MSE loss for continuous, and CE for discrete
    Supports arbitrary obs -> action networks or ActorPolicy ModuleContainers.
    """

    def __init__(
        self,
        *args,
        noise_scheduler=diffusers.schedulers.DDIMScheduler,
        noise_scheduler_kwargs: Optional[Dict] = None,
        num_inference_steps: Optional[int] = 10,
        horizon: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.network, ActorPolicy), "Must use an ActorPolicy with DiffusionPolicy"
        noise_scheduler_kwargs = {} if noise_scheduler_kwargs is None else noise_scheduler_kwargs
        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        if num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = num_inference_steps
        self.horizon = horizon

    def setup_optimizers(self) -> None:
        """
        Decay support added explicitly. Maybe move this to base implementation?
        """
        # create optim groups. Any parameters that is 2D or higher will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        groups = utils.create_optim_groups(self.network.parameters(), self.optim_kwargs)
        self.optim["network"] = self.optim_class(groups)

    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        obs = self.network.encoder(batch["obs"])
        B, T = batch["action"].shape[:2]
        assert T == self.horizon, "Received unexpected temporal dimension."
        noise = torch.randn_like(batch["action"])
        timesteps = torch.randint(
            low=0, high=self.scheduler.config.num_train_timesteps, size=(B,), device=self.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(batch["action"], noise, timesteps)

        noise_pred = self.network.actor(noisy_actions, timesteps, cond=obs)
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").sum(dim=-1)  # Sum over action Dim
        if "mask" in batch:
            mask = (~batch["mask"]).float()
            loss = loss * mask
            size = mask.sum()
        else:
            size = loss.numel()
        loss = loss.sum() / size
        return loss

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        """
        Overriding the Algorithm BaseClass Method train_step.
        Returns a dictionary of training metrics.
        """
        self.optim["network"].zero_grad(set_to_none=True)
        loss = self._compute_loss(batch)
        # Update the networks. These are done in a stack to support different grad options for the encoder.
        loss.backward()
        self.optim["network"].step()
        return dict(loss=loss.item())

    def validation_step(self, batch: Any) -> Dict:
        """
        Overriding the Algorithm BaseClass Method validation_step.
        Returns a dictionary of validation metrics.
        """
        with torch.no_grad():
            loss = self._compute_loss(batch)
        return dict(loss=loss.item())

    def _predict(self, batch: Dict) -> torch.Tensor:
        B = batch["obs"].shape[0]
        noisy_actions = torch.randn(B, self.horizon, self.processor.action_space.shape[0], device=self.device)
        with torch.no_grad():
            obs = self.network.encoder(batch["obs"])
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for timestep in self.noise_scheduler.timesteps:
                noise_pred = self.network.actor(noisy_actions, timestep.unsqueeze(0).to(self.device), cond=obs)
                noisy_actions = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=timestep, sample=noisy_actions
                ).prev_sample
        return noisy_actions

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
