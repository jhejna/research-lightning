import random
from typing import Any, Dict, Type, Union

import gym
import numpy as np
import torch

from research.utils import utils

from .base import Algorithm


class DQN(Algorithm):
    def __init__(
        self,
        env: gym.Env,
        network_class: Type[torch.nn.Module],
        dataset_class: Union[Type[torch.utils.data.IterableDataset], Type[torch.utils.data.Dataset]],
        train_freq: int = 4,
        target_freq: int = 1000,
        tau: float = 1.0,
        init_steps: int = 1000,
        max_grad_norm: float = 10,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_frac: float = 0.1,
        loss: str = "huber",
        **kwargs,
    ):
        super().__init__(env, network_class, dataset_class, **kwargs)
        # Save extra parameters
        self.tau = tau
        self.train_freq = train_freq
        self.target_freq = target_freq
        self.max_grad_norm = max_grad_norm
        self.init_steps = init_steps
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

    def _setup_train(self) -> None:
        # Now setup the logging parameters
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self.dataset.add(self._current_obs)  # Store the initial reset observation!

    def _compute_action(self) -> Any:
        return self.predict(dict(obs=self._current_obs))

    def _compute_value(self, batch: Any) -> torch.Tensor:
        next_q = self.target_network(batch["next_obs"])
        next_v, _ = next_q.max(dim=-1)
        return next_v

    def _train_step(self, batch: Any) -> Dict:
        all_metrics = {}

        if self.eps_frac > 0:
            frac = min(1.0, self.steps / (self.total_steps * self.eps_frac))
            eps = (1 - frac) * self.eps_start + frac * self.eps_end
        else:
            eps = 0.0

        if self.steps < self.init_steps or random.random() < eps:
            action = self.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self._compute_action()
            self.train_mode()

        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if "discount" in info:
            discount = info["discount"]
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences
        self.dataset.add(next_obs, action, reward, done, discount)

        if done:
            self._num_ep += 1
            # update metrics
            all_metrics["reward"] = self._episode_reward
            all_metrics["length"] = self._episode_length
            all_metrics["num_ep"] = self._num_ep
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs)  # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        if self.steps < self.init_steps or "obs" not in batch:
            return all_metrics

        if self.steps % self.train_freq == 0:
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

        if self.steps % self.target_freq == 0:
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
        # TODO: test this implementation, it might be broken due to API updates.
        next_a = self.network.predict(batch["next_obs"])
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
