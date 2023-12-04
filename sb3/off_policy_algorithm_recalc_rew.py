import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from dataclasses import dataclass
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from sb3.process_store_states import safe_std, safe_sum

OffPolicyAlgorithmSelf = TypeVar("OffPolicyAlgorithmSelf", bound="OffPolicyAlgorithm")

@dataclass
class DecisionTransformerConfig:
    enable: bool = False
    path_to_model: Optional[str] = None
    path_to_exp_config: Optional[str] = None
    number_of_states_in_aug_policy: int = 1 # how far forward to look. One is only the current obs
    number_of_actions_in_aug_policy: int = 1 # how far forward to look. One is the current action sent
    add_DT_action_to_aug_policy: bool = True
    aug_action_range: float = .2
    use_mean_DT_action: bool = True

@dataclass
class ResidualBCPolicyConfig:
    enable: bool = False
    path_to_model: Optional[str] = None
    path_to_exp_config: Optional[str] = None
    number_of_states_in_aug_policy: int = 1  # how far forward to look. One is only the current obs
    number_of_actions_in_aug_policy: int = 1  # how far forward to look. One is the current action sent
    add_BC_action_to_aug_policy: bool = True
    aug_action_range: float = .2
    use_mean_BC_action: bool = True

class OffPolicyAlgorithmRecalc(OffPolicyAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        decision_transformer_policy=None,
        decision_transformer_augment_options=None,
        residual_BC_policy=None,
        residual_BC_augment_options=None,
    ):
        if decision_transformer_augment_options:
            if decision_transformer_augment_options.enable:
                self.decision_transformer_policy = decision_transformer_policy
            else:
                self.decision_transformer_policy = None
        else:
            self.decision_transformer_policy = None
        self.decision_transformer_augment_options = decision_transformer_augment_options

        if residual_BC_augment_options:
            if residual_BC_augment_options.enable:
                self.residual_BC_policy = residual_BC_policy
            else:
                self.residual_BC_policy = None
        else:
            self.residual_BC_policy = None
        self.residual_BC_augment_options = residual_BC_augment_options

        self.keys_to_store = ['change_in_steering', 'steering', 'throttle_brake', 'change_in_throttle_brake',
                              'hit_wall_time', 'course_off_time']


        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
        )


    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    device=self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                device=self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )
        if self.residual_BC_policy is not None or self.decision_transformer_policy is not None:
            cfg = self.residual_BC_augment_options if self.residual_BC_policy else self.decision_transformer_augment_options
            self.action_space = self.action_space
            self.residual_action_space = gym.spaces.Box(
                low = self.action_space.low * cfg.aug_action_range,
                high = self.action_space.high * cfg.aug_action_range,
                shape = self.action_space.shape,
            )
        else:
            self.residual_action_space = self.action_space

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.residual_action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

    def learn(
        self: OffPolicyAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        recalculate_rewards=None,
    ) -> OffPolicyAlgorithmSelf:

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self._last_obs = self.env.reset()
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps,
                               recalculate_rewards=recalculate_rewards)

        callback.on_training_end()

        return self

    def train(self, gradient_steps: int, batch_size: int, recalculate_rewards = None) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()


    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.residual_action_space.low, self.residual_action_space.high
        if np.any(high==low):
            if self.residual_BC_policy or self.decision_transformer_policy:
                return action * 0.0
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def _unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.residual_action_space.low, self.residual_action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        last_obs = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if last_obs is None:
            last_obs = self._last_obs
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.residual_action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.residual_action_space, gym.spaces.Box):
            scaled_action = self._scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self._unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record('rollout/ep_rew_std', safe_std([ep_info["r"] for ep_info in self.ep_info_buffer]))

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, )
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    if self.decision_transformer_policy is not None or self.residual_BC_policy is not None:
                        act_dim = buffer_action.shape[-1]
                        DT_act = next_obs[i, -act_dim:]
                        next_obs_true = infos[i]["terminal_observation"]
                        next_obs[i] = np.concatenate((next_obs_true, DT_act), axis = -1)
                    else:
                        next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        if self.decision_transformer_policy is None:
            return self._collect_rollouts_orig(env, callback, train_freq, replay_buffer, action_noise, learning_starts, log_interval, )
        else:
            print('collecting rollouts...')
            with th.no_grad():
                return self._collect_rollouts_with_transformer(env,
                                                            callback, train_freq, replay_buffer, action_noise, learning_starts,
                                            log_interval, )

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)




    def _collect_rollouts_with_transformer(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        assert self.decision_transformer_policy is not None
        assert self.decision_transformer_augment_options.enable
        self.policy.set_training_mode(False)


        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        #########################
        DT_eval_rtg = self.decision_transformer_policy.variant['eval_rtg']
        DT_use_mean = True
        DT_target_return = [DT_eval_rtg * self.decision_transformer_policy.reward_scale] * env.num_envs
        state_dim = self.decision_transformer_policy.state_dim
        act_dim = self.decision_transformer_policy.act_dim
        device = self.device
        ## inside vec_ function
        assert len(DT_target_return) == env.num_envs
        assert self._last_obs.shape[0] == len(DT_target_return)
        self.decision_transformer_policy.eval()

        state_mean = th.from_numpy(self.decision_transformer_policy.state_mean).to(device=self.device)
        state_std = th.from_numpy(self.decision_transformer_policy.state_std).to(device=self.device)
        num_envs = env.num_envs

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        DT_states = (
            th.from_numpy(self._last_obs)
            .reshape(num_envs, state_dim)
            .to(device=device, dtype=th.float32)
        ).reshape(num_envs, -1, state_dim)
        DT_actions = th.zeros(0, device=self.device, dtype=th.float32)
        DT_rewards = th.zeros(0, device=self.device, dtype=th.float32)

        DT_ep_return = DT_target_return
        DT_target_return = th.tensor(DT_ep_return, device=device, dtype=th.float32).reshape(
            num_envs, -1, 1
        )
        DT_timesteps = th.tensor([0] * num_envs, device=device, dtype=th.long).reshape(
            num_envs, -1
        )

        # episode_return, episode_length = 0.0, 0
        DT_episode_return = np.zeros((num_envs, 1)).astype(float)
        DT_episode_length = np.full(num_envs, np.inf)

        DT_unfinished = np.ones(num_envs).astype(bool)

        #########################
        callback.on_rollout_start()
        continue_training = True
        t = 0

        ########################################
        # add padding
        DT_actions = th.cat(
            [
                DT_actions,
                th.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        DT_rewards = th.cat(
            [
                DT_rewards,
                th.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )
        with th.no_grad():
            DT_state_pred, DT_action_dist, DT_reward_pred = self.decision_transformer_policy.get_predictions(
                (DT_states.to(dtype=th.float32) - state_mean) / state_std,
                DT_actions.to(dtype=th.float32),
                DT_rewards.to(dtype=th.float32),
                DT_target_return.to(dtype=th.float32),
                DT_timesteps.to(dtype=th.long),
                num_envs=num_envs,
            )
            DT_state_pred = DT_state_pred.detach().cpu().numpy().reshape(num_envs, -1)
            if self.decision_transformer_policy.return_embedding:
                DT_reward_pred = DT_reward_pred.detach().cpu().numpy().reshape(num_envs)

            # the return action is a SquashNormal distribution
            assert self.decision_transformer_augment_options.number_of_states_in_aug_policy == 1
            assert self.decision_transformer_augment_options.number_of_actions_in_aug_policy == 1
            DT_action = DT_action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
            # if self.decision_transformer_augment_options.use_mean_DT_action:
            if DT_use_mean:
                DT_action = DT_action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
            DT_action = DT_action.clamp(*self.decision_transformer_policy.action_range)

        previous_obs = np.concatenate((self._last_obs, DT_action.detach().cpu().numpy()), axis=-1)
        self._last_obs = previous_obs
        ########################################
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            ########################################
            clipped_actions = actions
            # Clip the actions  before adding to the decision transformer actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions,
                                          self.action_space.low * self.decision_transformer_augment_options.aug_action_range,
                                          self.action_space.high * self.decision_transformer_augment_options.aug_action_range)

            if self.decision_transformer_augment_options.add_DT_action_to_aug_policy:
                DT_action_np = DT_action.detach().cpu().numpy()
                clipped_actions = clipped_actions + DT_action_np
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            ########################################

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(clipped_actions)


            ########################################
            DT_state = new_obs
            DT_reward = rewards
            DT_done = dones
            DT_infos = infos

            assert not 'phase_action' in infos[0]
            DT_action = th.from_numpy(clipped_actions.reshape(DT_action_np.shape)).to(device=device)
            DT_episode_return[DT_unfinished] += DT_reward[DT_unfinished].reshape(-1, 1)

            DT_actions[:, -1] = DT_action
            DT_state = (
                th.from_numpy(DT_state).to(device=device).reshape(num_envs, -1, state_dim)
            )
            DT_states = th.cat([DT_states, DT_state], dim=1)
            DT_reward = th.from_numpy(DT_reward).to(device=device).reshape(num_envs, 1)
            DT_rewards[:, -1] = DT_reward

            if self.decision_transformer_policy.mode != "delayed":
                DT_reward_scale = self.decision_transformer_policy.reward_scale
                DT_pred_return = DT_target_return[:, -1] - (DT_reward * DT_reward_scale)
            else:
                DT_pred_return = DT_target_return[:, -1]
            DT_target_return = th.cat(
                [DT_target_return, DT_pred_return.reshape(num_envs, -1, 1)], dim=1
            )

            DT_timesteps = th.cat(
                [
                    DT_timesteps,
                    th.ones((num_envs, 1), device=device, dtype=th.long).reshape(
                        num_envs, 1
                    )
                    * (t + 1),
                ],
                dim=1,
            )
            t+=1
            ########################################

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            if np.any(DT_done):
                ind = np.where(DT_done)[0]
                DT_unfinished[ind] = False
                DT_episode_length[ind] = np.minimum(DT_episode_length[ind], t + 1)

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            ########################################
            # add padding
            DT_actions = th.cat(
                [
                    DT_actions,
                    th.zeros((num_envs, act_dim), device=device).reshape(
                        num_envs, -1, act_dim
                    ),
                ],
                dim=1,
            )
            DT_rewards = th.cat(
                [
                    DT_rewards,
                    th.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
                ],
                dim=1,
            )
            with th.no_grad():
                DT_state_pred, DT_action_dist, DT_reward_pred = self.decision_transformer_policy.get_predictions(
                    (DT_states.to(dtype=th.float32) - state_mean) / state_std,
                    DT_actions.to(dtype=th.float32),
                    DT_rewards.to(dtype=th.float32),
                    DT_target_return.to(dtype=th.float32),
                    DT_timesteps.to(dtype=th.long),
                    num_envs=num_envs,
                )
                DT_state_pred = DT_state_pred.detach().cpu().numpy().reshape(num_envs, -1)
                if self.decision_transformer_policy.return_embedding:
                    DT_reward_pred = DT_reward_pred.detach().cpu().numpy().reshape(num_envs)

                # the return action is a SquashNormal distribution
                assert self.decision_transformer_augment_options.number_of_states_in_aug_policy == 1
                assert self.decision_transformer_augment_options.number_of_actions_in_aug_policy == 1
                DT_action = DT_action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
                # if self.decision_transformer_augment_options.use_mean_DT_action:
                if DT_use_mean:
                    DT_action = DT_action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
                DT_action = DT_action.clamp(*self.decision_transformer_policy.action_range)

            new_obs = np.concatenate((new_obs, DT_action.detach().cpu().numpy()), axis=-1)
            ########################################


            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

            if np.any(dones):
                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _collect_rollouts_orig(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        use_residual_BC = False
        if self.residual_BC_policy:
            assert self.residual_BC_augment_options.enable
            use_residual_BC = True
            self.residual_BC_policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        ##### sample BC action
        if use_residual_BC:
            obs_th = obs_as_tensor(self._last_obs, device = self.device)
            BC_action, _, _ = self.residual_BC_policy(obs_th)
            BC_action = BC_action.detach().cpu().numpy()
            BC_action = BC_action.clip(self.residual_BC_policy.action_space.low,
                                       self.residual_BC_policy.action_space.high)
            previous_obs = np.concatenate((self._last_obs, BC_action), axis=-1)
            self._last_obs = previous_obs

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            ########################################
            clipped_actions = actions
            if use_residual_BC:
                # Clip the actions  before adding to the decision transformer actions
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(actions,
                                              self.action_space.low * self.residual_BC_augment_options.aug_action_range,
                                              self.action_space.high * self.residual_BC_augment_options.aug_action_range)

                if self.residual_BC_augment_options.add_BC_action_to_aug_policy:
                    clipped_actions = clipped_actions + BC_action
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            ########################################
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            if use_residual_BC:
                obs_th = obs_as_tensor(new_obs, device=self.device)
                BC_action, _, _ = self.residual_BC_policy(obs_th)
                BC_action = BC_action.detach().cpu().numpy()
                BC_action = BC_action.clip(self.residual_BC_policy.action_space.low,
                                           self.residual_BC_policy.action_space.high)
                new_obs = np.concatenate((new_obs, BC_action), axis=-1)

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

            if np.any(dones):
                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

