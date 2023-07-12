import random
from typing import Any, Dict, Type

import numpy as np
import torch

from research.utils import utils

from ..off_policy_algorithm import OffPolicyAlgorithm


class DQN(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        target_freq: int = 1000,
        tau: float = 1.0,
        max_grad_norm: float = 10,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_frac: float = 0.1,
        loss: str = "huber",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Save extra parameters
        self.tau = tau
        self.target_freq = target_freq
        self.max_grad_norm = max_grad_norm
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frac = eps_frac
        self.loss = self._get_loss(loss)

    def _get_loss(self, loss: str):
        if loss == "mse":
            return torch.nn.MSELoss()
        elif loss == "huber":
            return torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid loss specification")

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

    def _compute_action(self) -> Any:
        return self.predict(dict(obs=self._current_obs))

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        if self.eps_frac > 0:
            frac = min(1.0, step / (total_steps * self.eps_frac))
            eps = (1 - frac) * self.eps_start + frac * self.eps_end
        else:
            eps = 0.0

        if random.random() < eps:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                action = self.predict(dict(obs=self._current_obs))
        return action

    def _compute_value(self, batch: Any) -> torch.Tensor:
        next_q = self.target_network(batch["next_obs"])
        next_v, _ = next_q.max(dim=-1)
        return next_v

    def train_step(self, batch: Any, step: int, total_steps: int) -> Dict:
        all_metrics = {}

        if step < self.random_steps or "obs" not in batch:
            return all_metrics

        # Update the agent
        with torch.no_grad():
            next_v = self._compute_value(batch)
            target_q = batch["reward"] + batch["discount"] * next_v

        q = self.network(batch["obs"])
        q = torch.gather(q, dim=-1, index=batch["action"].long().unsqueeze(-1)).squeeze(-1)
        loss = self.loss(q, target_q)

        self.optim["network"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optim["network"].step()

        all_metrics["q_loss"] = loss.item()
        all_metrics["target_q"] = target_q.mean().item()

        if step % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _predict(self, batch: Any) -> torch.Tensor:
        with torch.no_grad():
            q = self.network(batch["obs"])
            action = q.argmax(dim=-1)
            return action

    def _validation_step(self, batch: Any):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")


class DoubleDQN(DQN):
    def _compute_value(self, batch: Any) -> torch.Tensor:
        next_a = self.network(batch["next_obs"]).argmax(dim=-1)
        next_q = self.target_network(batch["next_obs"])
        next_v = torch.gather(next_q, dim=-1, index=next_a.unsqueeze(-1)).squeeze(-1)
        return next_v


class SoftDQN(DQN):
    def __init__(self, *args, exploration_alpha=0.01, target_alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_alpha = exploration_alpha
        self.target_alpha = target_alpha

    def _compute_action(self) -> Any:
        obs = utils.unsqueeze(self._current_obs, 0)
        obs = self._format_batch(obs)
        q = self.network(obs)
        dist = torch.nn.functional.softmax(q / self.exploration_alpha, dim=-1)
        dist = torch.distributions.categorical.Categorical(dist)
        action = dist.sample()
        action = utils.get_from_batch(action, 0)
        action = utils.to_np(action)
        return action

    def _compute_value(self, batch: Any) -> torch.Tensor:
        next_q = self.target_network(batch["next_obs"])
        next_v = self.target_alpha * torch.logsumexp(next_q / self.target_alpha, dim=-1)
        return next_v


class SoftDoubleDQN(DQN):
    def __init__(self, *args, exploration_alpha=0.01, target_alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_alpha = exploration_alpha
        self.target_alpha = target_alpha

    def _compute_action(self) -> torch.Tensor:
        obs = utils.unsqueeze(self._current_obs, 0)
        obs = self._format_batch(obs)
        q = self.network(obs)
        dist = torch.nn.functional.softmax(q / self.exploration_alpha, dim=-1)
        dist = torch.distributions.categorical.Categorical(dist)
        action = dist.sample()
        action = utils.get_from_batch(action, 0)
        action = utils.to_np(action)
        return action

    def _compute_value(self, batch: Any) -> torch.Tensor:
        log_pi = torch.nn.functional.log_softmax(self.network(batch["next_obs"]), dim=-1)
        next_q = self.target_network(batch["next_obs"])
        next_v = self.target_alpha * torch.logsumexp(next_q / self.target_alpha + log_pi, dim=-1)
        return next_v
