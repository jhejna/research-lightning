from typing import Any, Dict, Optional, Tuple

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
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self._discrete = True
            self.criterion = torch.nn.CrossEntropyLoss(ignore_idx=IGNORE_INDEX)
        else:
            self._discrete = False
            self.criterion = torch.nn.MSELoss(reduction="none")

        self.grad_norm_clip = grad_norm_clip

    def setup_optimizers(self) -> None:
        """
        Decay support added explicitly for Transformer based BC models.
        """

        def get_decay_parameters(m):
            # Returns the names of the model parameters that are not LayerNorms or Embeddings
            result = []
            for name, child in m.named_children():
                result += [
                    f"{name}.{n}"
                    for n in get_decay_parameters(child)
                    if not isinstance(child, (torch.nn.LayerNorm, torch.nn.Embedding))
                ]
            # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
            result += list(m._parameters.keys())
            return result

        decay_parameters = get_decay_parameters(self.network)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]  # add all bias parameters
        decay_parameters = [
            name for name in decay_parameters if "embedding" not in name
        ]  # ignore layers with "embedding"

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.network.named_parameters() if n in decay_parameters],
            },
            {
                "params": [p for n, p in self.network.named_parameters() if n not in decay_parameters],
            },
        ]
        optimizer_grouped_parameters[0].update(self.optim_kwargs)
        optimizer_grouped_parameters[1].update(self.optim_kwargs)
        optimizer_grouped_parameters[1]["weight_decay"] = 0.0  # set weight decay to zero for these parameters
        self.optim["network"] = self.optim_class(optimizer_grouped_parameters)

    def _compute_loss_and_accuracy(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Helper function used to reduce repeated computation
        """
        # Predictions are expected to be shape (B, D) or (B, S, D)
        if isinstance(self.network, ActorPolicy):
            z = self.network.encoder(batch["obs"])
            pred = self.network.actor(z)
        else:
            pred = self.network(batch)
        label = batch["action"]  # WARNING: this could cause errors in discrete if data isn't copied. Need to check.

        if self._discrete:
            if "mask" in batch:
                label[batch["mask"]] = IGNORE_INDEX  # ignore index
            if len(pred.shape) == 3:
                pred = pred.view(-1, pred.size(-1))
                label = label.view(-1)
            loss = self.criterion(pred, label)
            # Compute the accuracy
            with torch.no_grad():
                accuracy = (torch.argmax(pred, dim=-1) == label).sum() / (label != IGNORE_INDEX).sum()

        else:
            loss = self.criterion(pred, label).sum(dim=-1)  # sum over the last dim
            if "mask" in batch:
                mask = 1 - batch["mask"].float()
                loss = mask * loss
                size = mask.sum()
            else:
                size = loss.numel()
            loss = loss.sum() / size
            accuracy = None

        return loss, accuracy

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        """
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of training metrics.
        """
        self.optim["network"].zero_grad(
            set_to_none=True
        )  # By default, the optimizer is stored under 'network' if no custom optimizer setup is performed.
        loss, accuracy = self._compute_loss_and_accuracy(batch)
        loss.backward()
        if self.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clip)
        self.optim["network"].step()
        metrics = dict(loss=loss.item())
        if accuracy is not None:
            metrics["accuracy"] = accuracy.item()
        return metrics

    def validation_step(self, batch: Any) -> Dict:
        """
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of validation metrics.
        """
        with torch.no_grad():
            loss, accuracy = self._compute_loss_and_accuracy(batch)
        metrics = dict(loss=loss.item())
        if accuracy is not None:
            metrics["accuracy"] = accuracy.item()
        return metrics

    def _predict(self, batch: Any, **kwargs) -> Any:
        if isinstance(self.network, ActorPolicy):
            with torch.no_grad():
                if len(kwargs) > 0:
                    raise ValueError("Default predict method does not accept key word args, but they were provided.")
                z = self.network.encoder(batch["obs"])
                pred = self.network.actor(z)
        else:
            pred = super()._predict(batch, **kwargs)
        return pred

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False)
        return action
