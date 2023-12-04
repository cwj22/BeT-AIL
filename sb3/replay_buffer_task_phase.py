import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
import torch as th
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple

from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class TaskPhaseRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    taken_actions: th.Tensor
    pid_actions: th.Tensor
    demo_action_flags: th.Tensor
    rewards: th.Tensor


class TaskPhaseRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.demo_action_flags, self.taken_actions = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self) -> None:

        self.taken_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.pid_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.demo_action_flags = np.zeros((self.buffer_size, self.n_envs), dtype=np.int)
        super().reset()

    ###     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
    ## need to redefine this if you want to use demo acctions



    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        taken_action: np.ndarray,
        pid_action: np.ndarray,
        demo_action_flag: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param demo_action_flag: flag saying 1 if taken by demonstrator
        :param agent_action_flag: flag saying 1 if taken by agent
        """
        taken_action = taken_action.reshape((self.n_envs, self.action_dim))
        self.taken_actions[self.pos] = np.array(taken_action).copy()
        pid_action = pid_action.reshape((self.n_envs, self.action_dim))
        self.pid_actions[self.pos] = np.array(pid_action).copy()
        self.demo_action_flags[self.pos] = np.array(demo_action_flag).copy()

        super().add(obs, action, reward, episode_start, value, log_prob)


    def get(self, batch_size: Optional[int] = None) -> Generator[TaskPhaseRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "taken_actions",
                "pid_actions",
                "demo_action_flags",
                'rewards'
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> TaskPhaseRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.taken_actions[batch_inds],
            self.pid_actions[batch_inds],
            self.demo_action_flags[batch_inds].flatten(),
            self.rewards[batch_inds].flatten(),

        )
        return TaskPhaseRolloutBufferSamples(*tuple(map(self.to_torch, data)))