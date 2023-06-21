import collections
from typing import Any, Dict, Optional

import gym
import numpy as np
import torch

from research.datasets import RolloutBuffer
from research.networks.base import ActorValuePolicy
from research.processors.normalization import RunningMeanStd, RunningObservationNormalizer
from research.utils import utils

from ..base import Algorithm


class PPO(Algorithm):
    def __init__(
        self,
        *args,
        num_epochs: int = 10,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coeff: float = 0.0,
        vf_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
        normalize_returns: bool = False,
        reward_clip: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Perform initial type checks
        assert isinstance(self.network, ActorValuePolicy)

        # Store algorithm values
        self.num_epochs = num_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coeff = ent_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        # Return normalization
        self.normalize_returns = normalize_returns
        self.reward_clip = reward_clip
        if self.normalize_returns:
            self.return_rms = RunningMeanStd(shape=())

        # Losses
        self.value_criterion = torch.nn.MSELoss()

    def _collect_rollouts(self) -> Dict:
        # Setup the dataset and network
        self.dataset.setup()
        self.eval()

        # Setup metrics
        metrics = dict(reward=[], length=[], success=[])
        ep_reward, ep_length, ep_return, ep_success = 0, 0, 0, False

        obs = self.env.reset()
        self.dataset.add(obs=obs)  # Add the first observation
        while not self.dataset.is_full:
            with torch.no_grad():
                obs = utils.unsqueeze(utils.to_tensor(obs), 0)
                if isinstance(self.processor, RunningObservationNormalizer):
                    self.processor.update(obs)
                batch = self.format_batch(dict(obs=obs))  # Preprocess obs
                latent = self.network.encoder(batch["obs"])
                dist = self.network.actor(latent)
                # Collect relevant information
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.network.value(latent).mean(dim=0)  # Account for Ensemble Dim
                # Unprocess back to numpy
                action = utils.to_np(utils.get_from_batch(action, 0))
                log_prob = utils.to_np(utils.get_from_batch(log_prob, 0))
                value = utils.to_np(utils.get_from_batch(value, 0))
                extras = self._compute_extras(dist)

            if isinstance(self.env.action_space, gym.spaces.Box):  # Clip the actions
                clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            obs, reward, done, info = self.env.step(clipped_action)

            ep_reward += reward
            ep_length += 1
            ep_return = self.dataset.discount * ep_return + reward
            if self.normalize_returns:
                self.return_rms.update(ep_return)
                reward = reward / self.return_rms.std.numpy()
                if self.reward_clip is not None:
                    reward = np.clip(reward, -self.reward_clip, self.reward_clip)

            if ("success" in info and info["success"]) or ("is_success" in info and info["is_success"]):
                ep_success = True

            if done:
                # Update metrics
                metrics["reward"].append(ep_reward)
                metrics["length"].append(ep_length)
                metrics["success"].append(ep_success)
                ep_reward, ep_length, ep_return, ep_success = 0, 0, 0, False
                # If its done, we need to update the observation as well as the terminal reward
                with torch.no_grad():
                    obs = utils.unsqueeze(utils.to_tensor(obs), 0)
                    if isinstance(self.processor, RunningObservationNormalizer):
                        self.processor.update(obs)
                    batch = self.format_batch(dict(obs=obs))  # Preprocess obs
                    terminal_value = self.network.value(self.network.encoder(batch["obs"])).mean(dim=0)  # Ensemble Avg
                    terminal_value = utils.to_np(utils.get_from_batch(terminal_value, 0))
                reward += self.dataset.discount * terminal_value
                obs = self.env.reset()

            # Note: Everything is from the last observation except for the observation, which is really next_obs
            self.dataset.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, **extras)

            self._env_steps += 1

        self.train()
        metrics["env_steps"] = self._env_steps  # Log env steps because it's not proportional to train steps
        metrics["reward_std"] = np.std(metrics["reward"])
        metrics["reward"] = np.mean(metrics["reward"])
        metrics["length"] = np.mean(metrics["length"])
        metrics["success"] = np.mean(metrics["success"])
        return metrics

    def _compute_extras(self, dist):
        # Used for computing extras values for different versions of PPO
        return {}

    def setup(self) -> None:
        self.setup_train_dataset()
        assert isinstance(self.dataset, RolloutBuffer)
        # Logging metrics
        self._env_steps = 0
        self._collect_rollouts()
        # Track the number of epochs. This is used for training.
        self._epochs = 0

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        metrics = dict(env_steps=self._env_steps)
        self._epochs += int(self.dataset.last_batch)
        if self.dataset.last_batch and self._epochs % self.num_epochs == 0:
            # On the last batch of the epoch recollect data.
            metrics.update(self._collect_rollouts())
            return metrics  # Return immediatly so we don't do a gradient step on old data.

        # Run the policy to predict the values, log probs, and entropies
        latent = self.network.encoder(batch["obs"])
        dist = self.network.actor(latent)
        log_prob = dist.log_prob(batch["action"]).sum(dim=-1)
        value = self.network.value(latent)

        advantage = batch["advantage"]
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = torch.exp(log_prob - batch["log_prob"])
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        last_value = batch["value"].expand(value.shape[0], -1)
        if self.clip_range_vf is not None:
            value = last_value + torch.clamp(value - last_value, -self.clip_range_vf, self.clip_range_vf)
        value_loss = self.value_criterion(batch["returns"].expand(value.shape[0], -1), value)

        entropy_loss = -torch.mean(dist.entropy().sum(dim=-1))

        total_loss = policy_loss + self.vf_coeff * value_loss + self.ent_coeff * entropy_loss

        self.optim["network"].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optim["network"].step()

        # Update all of the metrics at the end to not break computation flow
        metrics["policy_loss"] = policy_loss.item()
        metrics["value_loss"] = value_loss.item()
        metrics["entropy_loss"] = entropy_loss.item()
        metrics["loss"] = total_loss.item()
        return metrics

    def validation_step(self, batch: Any):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _predict(self, batch: Any, sample=False) -> torch.Tensor:
        with torch.no_grad():
            latent = self.network.encoder(batch["obs"])
            dist = self.network.actor(latent)
            if sample:
                action = dist.sample()
            else:
                action = dist.loc
        return action


class AdaptiveKLPPO(PPO):
    def __init__(self, *args, target_kl: float = 0.025, kl_window: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.clip_range is None, "Clip range is not used in Adaptive KL based PPO"
        self.target_kl = target_kl
        self.kl_window = kl_window
        self.beta = 1

    def _compute_extras(self, dist):
        mu = utils.to_np(utils.get_from_batch(dist.loc, 0))
        sigma = utils.to_np(utils.get_from_batch(dist.scale, 0))
        return dict(mu=mu, sigma=sigma)

    def setup(self):
        # Logging metrics
        self._env_steps = 0
        self._collect_rollouts()
        self._kl_divs = collections.deque(maxlen=self.kl_window)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        metrics = dict(env_steps=self._env_steps)
        self._epochs += int(self.dataset.last_batch)
        if self.dataset.last_batch and self._epochs % self.num_epochs == 0:
            # On the last batch of the epoch recollect data.
            metrics.update(self._collect_rollouts())
            # set flag for updating KL divergence
            update_kl_beta = True
        else:
            update_kl_beta = False
        # Run the policy to predict the values, log probs, and entropies
        latent = self.network.encoder(batch["obs"])
        dist = self.network.actor(latent)
        log_prob = dist.log_prob(batch["action"]).sum(dim=-1)
        value = self.network.value(latent)

        advantage = batch["advantage"]
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratio = torch.exp(log_prob - batch["log_prob"])
        policy_loss = -(advantage * ratio).mean()

        # Compute the KL divergence here.
        # Note that this could be done more numerically stable by using log_sigma instead of sigma
        old_dist = torch.distributions.Normal(batch["mu"], batch["sigma"])
        kl_div = torch.distributions.kl.kl_divergence(old_dist, dist).sum(dim=-1).mean()

        if self.clip_range_vf is not None:
            value = batch["value"] + torch.clamp(value - batch["value"], -self.clip_range_vf, self.clip_range_vf)
        value_loss = self.value_criterion(batch["returns"], value)

        entropy_loss = -torch.mean(dist.entropy().sum(dim=-1))

        total_loss = policy_loss + self.beta * kl_div + self.vf_coeff * value_loss + self.ent_coeff * entropy_loss

        self.optim["network"].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optim["network"].step()

        # Update KL Divergences
        self._kl_divs.append(kl_div.detach().cpu().numpy())
        if update_kl_beta:
            avg_kl = np.mean(self._kl_divs)
            if avg_kl < self.target_kl / 1.5:
                self.beta = self.beta / 2
            elif avg_kl > self.target_kl * 1.5:
                self.beta = self.beta * 2
            # Empty the KL buffer
            self._kl_divs = collections.deque(maxlen=self.kl_window)

        # Update all of the metrics at the end to not break computation flow
        metrics["policy_loss"] = policy_loss.item()
        metrics["kl_divergence"] = kl_div.item()
        metrics["value_loss"] = value_loss.item()
        metrics["entropy_loss"] = entropy_loss.item()
        metrics["loss"] = total_loss.item()
        metrics["beta"] = self.beta
        return metrics

    def validation_step(self, batch: Any) -> Dict:
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
