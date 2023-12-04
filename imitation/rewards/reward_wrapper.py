"""Common wrapper for adding custom reward values to an environment."""

import collections
from typing import Deque

import numpy as np
from stable_baselines3.common import callbacks
from stable_baselines3.common import logger as sb_logger
from stable_baselines3.common import vec_env

from imitation.rewards import reward_function


class WrappedRewardCallback(callbacks.BaseCallback):
    """Logs mean wrapped reward as part of RL (or other) training."""

    def __init__(self, episode_rewards: Deque[float], *args, episode_rewards_gail =None,
                 episode_rewards_aug = None, **kwargs):
        """Builds WrappedRewardCallback.

        Args:
            episode_rewards: A queue that episode rewards will be placed into.
            *args: Passed through to `callbacks.BaseCallback`.
            **kwargs: Passed through to `callbacks.BaseCallback`.
        """
        self.episode_rewards = episode_rewards
        self.episode_rewards_gail = episode_rewards_gail
        self.episode_rewards_aug = episode_rewards_aug
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if len(self.episode_rewards) == 0:
            return
        mean = sum(self.episode_rewards) / len(self.episode_rewards)
        assert isinstance(self.logger, sb_logger.Logger)
        self.logger.record("rollout/ep_rew_wrapped_mean", mean)
        if self.episode_rewards_gail is not None and len(self.episode_rewards_gail)>0:
            mean = sum(self.episode_rewards_gail) / len(self.episode_rewards_gail)
            self.logger.record("rollout/ep_rew_wrapped_mean_gail", mean)
        if self.episode_rewards_aug is not None and len(self.episode_rewards_aug)>0:
            mean = sum(self.episode_rewards_aug) / len(self.episode_rewards_aug)
            self.logger.record("rollout/ep_rew_wrapped_mean_aug", mean)


class OptionsVecEnvWrapper(vec_env.VecEnvWrapper):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        ep_history: int = 100,
        use_options: bool = False,
        latent_option_dim: int = None,
    ):
        assert not isinstance(venv, OptionsVecEnvWrapper)
        super().__init__(venv)
        self._old_obs = None
        self._actions = None

        self.use_options = use_options
        if use_options:
            self.latent_option_dim = latent_option_dim
            assert latent_option_dim is not None
            self._old_option = None
            self._option = None
        self.reset()

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        if self.use_options:
            self._old_option = np.ones((self.venv.num_envs,1), dtype=np.int64)*self.latent_option_dim
            return self._old_obs, self._old_option
        else:
            return self._old_obs

    def step(self, actions: np.ndarray, new_options: np.ndarray = None):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions, new_options)
        return self.step_wait()

    def step_async(self, actions, new_options=None):
        self._actions = actions
        if self.use_options:
            assert new_options is not None
            self._option = new_options
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        return obs, old_rews, dones, infos


class RewardVecEnvWrapper(vec_env.VecEnvWrapper):
    """Uses a provided reward_fn to replace the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization. A tricky part
    about this class is keeping track of the most recent observation from each
    environment.

    Will also include the previous reward given by the inner VecEnv in the
    returned info dict under the `original_env_rew` key.
    """

    def __init__(
        self,
        venv: vec_env.VecEnv,
        reward_fn: reward_function.RewardFn,
        ep_history: int = 100,
        use_options: bool = False,
        latent_option_dim: int = None,
        gen_policy_observation_extension: int = 0,
        gen_augmented_lambda: float = 0.0,  # augment with augment env reward
        gen_augmented_lambda_schedule: str = 'constant',
    ):
        """Builds RewardVecEnvWrapper.

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
        """
        if 'end_at_' in gen_augmented_lambda_schedule:
            self.end_augment = int(gen_augmented_lambda_schedule[7:])
        elif not gen_augmented_lambda_schedule == 'constant':
            raise NotImplementedError
        self.gen_augmented_lambda = gen_augmented_lambda
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self.episode_rewards: Deque = collections.deque(maxlen=ep_history)
        if self.gen_augmented_lambda>0.0:
            self.episode_rewards_gail: Deque = collections.deque(maxlen=ep_history)
            self.episode_rewards_aug: Deque = collections.deque(maxlen=ep_history)
            self._cumulative_rew_gail = np.zeros((venv.num_envs,))
            self._cumulative_rew_aug = np.zeros((venv.num_envs,))
        else:
            self.episode_rewards_gail = None
            self.episode_rewards_aug = None
        self._cumulative_rew = np.zeros((venv.num_envs,))

        self.reward_fn = reward_fn
        self._old_obs = None
        self._actions = None
        self.gen_policy_observation_extension = gen_policy_observation_extension

        self.use_options = use_options
        if use_options:
            self.latent_option_dim = latent_option_dim
            assert latent_option_dim is not None
            self._old_option = None
            self._option = None
        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        """Creates `WrappedRewardCallback` connected to this `RewardVecEnvWrapper`."""
        return WrappedRewardCallback(self.episode_rewards,
                                     episode_rewards_gail = self.episode_rewards_gail,
                                     episode_rewards_aug = self.episode_rewards_aug)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        if self.use_options:
            self._old_option = np.ones((self.venv.num_envs,1), dtype=np.int64)*self.latent_option_dim
            return self._old_obs, self._old_option
        else:
            return self._old_obs


    def update_gen_augmented_lambda(self, value):
        self.gen_augmented_lambda = value
        return  self.gen_augmented_lambda

    def step(self, actions: np.ndarray, new_options: np.ndarray = None):
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions, new_options)
        return self.step_wait()

    def step_async(self, actions, new_options=None):
        self._actions = actions
        if self.use_options:
            assert new_options is not None
            self._option = new_options
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(single_obs)
        obs_fixed = np.stack(obs_fixed)

        if self.use_options:
            assert self.gen_policy_observation_extension == 0
            rews = self.reward_fn(self._old_obs, self._actions, obs_fixed, np.array(dones),
                                  self._old_option, self._option)

        else:
            if self.gen_policy_observation_extension > 0:
                rews = self.reward_fn(self._old_obs[:, :-self.gen_policy_observation_extension],
                                      self._actions, obs_fixed[:, :-self.gen_policy_observation_extension],
                                      np.array(dones))

            else:
                rews = self.reward_fn(self._old_obs, self._actions, obs_fixed, np.array(dones))
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))


        if self.gen_augmented_lambda > 0:
            augmented_reward = np.array([info_dict['augmented_reward'] for info_dict in infos])
            self._cumulative_rew_gail += rews
            rews = self.gen_augmented_lambda * augmented_reward + (1 - self.gen_augmented_lambda)*rews
            self._cumulative_rew_aug += augmented_reward



        # Update statistics
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        if self.episode_rewards_aug is not None:
            assert self.episode_rewards_gail is not None
            for single_done, single_ep_rew_aug, single_ep_rew_gail in zip(dones, self._cumulative_rew_aug, self._cumulative_rew_gail):
                if single_done:
                    self.episode_rewards_gail.append(single_ep_rew_gail)
                    self.episode_rewards_aug.append(single_ep_rew_aug)
            self._cumulative_rew_aug[done_mask] = 0
            self._cumulative_rew_gail[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        self._old_obs = obs
        for info_dict, old_rew, rew in zip(infos, old_rews, rews):
            info_dict["original_env_rew"] = old_rew




        return obs, rews, dones, infos
