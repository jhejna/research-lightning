from typing import Any, Dict, Optional

import gym
import numpy as np
import torch

from research.networks.base import ActorPolicy

from ..off_policy_algorithm import OffPolicyAlgorithm

IGNORE_INDEX = -100


class BehaviorCloning(OffPolicyAlgorithm):
    """
    BC Implementation.
    Uses MSE loss for continuous, and CE for discrete
    Supports arbitrary obs -> action networks or ActorPolicy ModuleContainers.
    """

    def __init__(self, *args, grad_norm_clip: Optional[float] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(self.action_space, gym.spaces.Discrete):
            self._discrete = True
            self.criterion = torch.nn.CrossEntropyLoss(ignore_idx=IGNORE_INDEX)
        else:
            self._discrete = False
            self.criterion = torch.nn.MSELoss(reduction="none")

        self.grad_norm_clip = grad_norm_clip

    def setup_optimizers(self) -> None:
        """
        Decay support added explicitly. Maybe move this to base implementation?
        """
        named_params = {pn: p for pn, p in self.network.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D or higher will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_group = {"params": [p for n, p in named_params.items() if p.dim() >= 2]}
        no_decay_group = {"params": [p for n, p in named_params.items() if p.dim() < 2]}
        decay_group.update(self.optim_kwargs)
        no_decay_group.update(self.optim_kwargs)
        no_decay_group["weight_decay"] = 0.0
        self.optim["network"] = self.optim_class((decay_group, no_decay_group))

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        """
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of training metrics.
        """
        self.optim["network"].zero_grad(set_to_none=True)
        if isinstance(self.network, ActorPolicy):
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
        else:
            dist = self.network(batch)

        if isinstance(dist, torch.distributions.Distribution):
            loss = -dist.log_prob(batch["action"])  # NLL Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Box):
            loss = torch.nn.functional.mse_loss(dist, batch["label"], reduction="none")  # MSE Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Discrete):
            loss = torch.nn.functional.cross_entropy(dist, batch["label"], ignore_index=IGNORE_INDEX, reduction="none")
        else:
            raise ValueError("Invalid Policy output")

        # Aggregate the losses
        if "mask" in batch:
            assert batch["mask"].shape == loss.shape
            mask = 1 - batch["mask"].float()
            loss = mask * loss
            size = mask.sum()  # how many elements we train on.
        else:
            size = loss.numel()

        loss = loss.sum() / size

        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clip)
        self.optim["network"].step()
        metrics = dict(loss=loss.item())
        return metrics

    def validation_step(self, batch: Any) -> Dict:
        """
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of validation metrics.
        """
        with torch.no_grad():
            if isinstance(self.network, ActorPolicy):
                z = self.network.encoder(batch["obs"])
                dist = self.network.actor(z)
            else:
                dist = self.network(batch)

            if isinstance(dist, torch.distributions.Distribution):
                loss = -dist.log_prob(batch["action"])  # NLL Loss
            elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Box):
                loss = torch.nn.functional.mse_loss(dist, batch["label"], reduction="none")  # MSE Loss
            elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Discrete):
                loss = torch.nn.functional.cross_entropy(
                    dist, batch["label"], ignore_index=IGNORE_INDEX, reduction="none"
                )
            else:
                raise ValueError("Invalid Policy output")

        return dict(loss=loss.item())

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
