import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from .replay_buffer_task_phase import TaskPhaseRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from dataclasses import dataclass

OnPolicyAlgorithmSelf = TypeVar("OnPolicyAlgorithmSelf", bound="OnPolicyAlgorithm")

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

class OnPolicyAlgorithmAddBC(OnPolicyAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        task_phase_buffer: bool = False,
        task_phase_current_beta: float = None,
        decision_transformer_policy = None,
        decision_transformer_augment_options = None,
        residual_BC_policy=None,
        residual_BC_augment_options=None,
    ):
        self.task_phase_buffer = task_phase_buffer
        self.task_phase_current_beta = task_phase_current_beta
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

        super().__init__(
            policy = policy,
            env = env,
            learning_rate = learning_rate,
            n_steps = n_steps,
            gamma = gamma,
            gae_lambda = gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces,
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.task_phase_buffer:
            buffer_cls = TaskPhaseRolloutBuffer
        else:
            buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, infos: List[Dict[str, Any]],
                            dones: Optional[np.ndarray] = None) -> Tuple[Optional[np.array], Optional[np.array], Optional[np.ndarray]]:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        if dones is None:
            dones = np.array([False] * len(infos))

        demo_action_flag = []
        taken_actions = np.zeros((len(infos),)+self.action_space.shape)
        pid_actions = np.zeros((len(infos),)+self.action_space.shape)
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

            demo = info.get('demo_action_flag')
            act = info.get('phase_action')
            act_pid = info.get('pid_action')
            if demo is not None:
                demo_action_flag.append(demo)
            if act is not None:
                taken_actions[idx,:] = act
            if act_pid is not None:
                pid_actions[idx, :] = act_pid

        if self.task_phase_buffer:
            assert len(demo_action_flag) == len(infos)
            assert act is not None
            assert act_pid is not None
            return np.array(demo_action_flag), np.array(taken_actions), np.array(pid_actions)
        else:
            return None, None, None

    def _evaluate_DT(self, vec_env):
        from imitation.sb3.evaluation_online_dt import vec_evaluate_episode_rtg
        eval_start = time.time()
        all_returns, all_lengths = [], []
        while len(all_returns) < 20-1:
            self.decision_transformer_policy.eval()
            eval_rtg = self.decision_transformer_policy.variant['eval_rtg']
            use_mean = True
            target_return = [eval_rtg * self.decision_transformer_policy.reward_scale] * vec_env.num_envs
            returns, lengths, _ = vec_evaluate_episode_rtg(
                vec_env,
                self.decision_transformer_policy.state_dim,
                self.decision_transformer_policy.act_dim,
                self.decision_transformer_policy,
                max_ep_len=1000,
                reward_scale=self.decision_transformer_policy.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.decision_transformer_policy.state_mean,
                state_std=self.decision_transformer_policy.state_std,
                device=self.device,
                use_mean=use_mean,
            )
            all_returns.extend(returns)
            all_lengths.extend(lengths)
        suffix = "_gm" if use_mean else ""
        outputs =  {
            f"evaluation/return_mean{suffix}": np.mean(all_returns),
            f"evaluation/return_std{suffix}": np.std(all_returns),
            f"evaluation/length_mean{suffix}": np.mean(all_lengths),
            f"evaluation/length_std{suffix}": np.std(all_lengths),
        }
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if self.decision_transformer_policy is None:
            return self._collect_rollouts_orig(env, callback, rollout_buffer, n_rollout_steps)
        else:
            # print('evaluating inside')
            # with th.no_grad():
            #     outputs, rews = self._evaluate_DT(env)
            # print(outputs)
            print('collecting rollouts...')
            with th.no_grad():
                return self._collect_rollouts_with_transformer(env, callback, rollout_buffer, n_rollout_steps)

    def _collect_rollouts_with_transformer(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ):

        assert self.decision_transformer_policy is not None
        assert self.decision_transformer_augment_options.enable
        assert self._last_obs is not None, "No previous observation was provided"

        DT_eval_rtg = self.decision_transformer_policy.variant['eval_rtg']
        DT_use_mean = True
        DT_target_return = [DT_eval_rtg * self.decision_transformer_policy.reward_scale] * env.num_envs

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        self._last_obs = env.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)


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

        callback.on_rollout_start()
        t = 0
        while n_steps < n_rollout_steps:
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

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                obs_tensor = th.concat((obs_tensor, DT_action), dim = -1)
                actions, values, log_probs = self.policy(obs_tensor)
                previous_obs = obs_tensor.detach().cpu().numpy()
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions  before adding to the decision transformer actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions,
                                          self.action_space.low*self.decision_transformer_augment_options.aug_action_range,
                                          self.action_space.high*self.decision_transformer_augment_options.aug_action_range)

            if self.decision_transformer_augment_options.add_DT_action_to_aug_policy:
                DT_action_np = DT_action.detach().cpu().numpy()
                clipped_actions = clipped_actions + DT_action_np
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # new_obs, rewards, dones, infos = env.step(DT_action_np)
            DT_state = new_obs
            DT_reward = rewards
            DT_done = dones
            DT_infos = infos

            if 'phase_action' in infos[0]:
                asdf
                taken_actions = np.array([info['phase_action'] for info in infos])
                DT_action = th.from_numpy(taken_actions.reshape(DT_action_np.shape)).to(device=device)
            else: # relabe actions after being sent to environment
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

            self.num_timesteps += env.num_envs
            t += 1

            if np.any(DT_done):
                ind = np.where(DT_done)[0]
                DT_unfinished[ind] = False
                DT_episode_length[ind] = np.minimum(DT_episode_length[ind], t + 1)

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            demo_action_flags, taken_actions, pid_actions = self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                asdf
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if self.task_phase_buffer:
                assert isinstance(rollout_buffer, TaskPhaseRolloutBuffer)
                rollout_buffer.add(previous_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                                   taken_actions, pid_actions, demo_action_flags)
            else:
                assert isinstance(rollout_buffer, RolloutBuffer)
                rollout_buffer.add(previous_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # calculate next DT action so you can send to policy
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
        if self.decision_transformer_augment_options.use_mean_DT_action:
            DT_action = DT_action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        DT_action = DT_action.clamp(*self.decision_transformer_policy.action_range)
        obs_tensor = th.concat((obs_as_tensor(new_obs, self.device), DT_action), dim=-1)

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def _collect_rollouts_orig(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        use_residual_BC = False
        if self.residual_BC_augment_options.enable:
            assert self.residual_BC_policy is not None
            self.residual_BC_policy.set_training_mode(False)
            use_residual_BC = True
            # only done for off policy
            raise NotImplementedError

        n_steps = 0
        rollout_buffer.reset()
        self._last_obs = env.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            demo_action_flags, taken_actions, pid_actions = self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if self.task_phase_buffer:
                assert isinstance(rollout_buffer, TaskPhaseRolloutBuffer)
                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                                   taken_actions, pid_actions, demo_action_flags)
            else:
                assert isinstance(rollout_buffer, RolloutBuffer)
                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: OnPolicyAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        expert_batch = None,
    ) -> OnPolicyAlgorithmSelf:
        iteration = 0

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

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            # with th.autograd.set_detect_anomaly(True):
            self.train(expert_batch)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
