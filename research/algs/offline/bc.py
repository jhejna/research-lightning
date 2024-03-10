from typing import Any, Dict, Optional

import gym
import numpy as np
import torch

from research.utils import utils

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
        self.grad_norm_clip = grad_norm_clip

    def setup_optimizers(self) -> None:
        # create optim groups. Any parameters that is 2D or higher will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        groups = utils.create_optim_groups(self.network.parameters(), self.optim_kwargs)
        self.optim["network"] = self.optim_class(groups)

    def _compute_loss(self, batch: Dict):
        dist = self.network(batch["obs"])

        if isinstance(dist, torch.distributions.Distribution):
            loss = -dist.log_prob(batch["action"])  # NLL Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Box):
            loss = torch.nn.functional.mse_loss(dist, batch["action"], reduction="none")  # MSE Loss
        elif torch.is_tensor(dist) and isinstance(self.processor.action_space, gym.spaces.Discrete):
            loss = torch.nn.functional.cross_entropy(dist, batch["action"], ignore_index=IGNORE_INDEX, reduction="none")
        else:
            raise ValueError("Invalid Policy output")

        # Aggregate the losses
        if "mask" in batch:
            assert batch["mask"].shape == loss.shape
            mask = (1 - batch["mask"]).float()
            loss = mask * loss
            size = mask.sum()  # how many elements we train on.
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
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clip)
        self.optim["network"].step()
        metrics = dict(loss=loss.item())
        return metrics

    def validation_step(self, batch: Any) -> Dict:
        """
        Overriding the Algorithm BaseClass Method validation_step.
        Returns a dictionary of validation metrics.
        """
        with torch.no_grad():
            loss = self._compute_loss(batch)

        return dict(loss=loss.item())

    def _get_train_action(self, obs: Any, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action
