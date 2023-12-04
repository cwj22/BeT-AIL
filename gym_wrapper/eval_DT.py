from stable_baselines3.common.callbacks import EvalCallback, sync_envs_normalization
import os

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy_withDT(
        model: "base_class.BaseAlgorithm",
        decision_transformer_policy,
        decision_transformer_augment_options,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        laptime_override_stop=False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    assert decision_transformer_policy is not None
    assert decision_transformer_augment_options.enable

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    DT_eval_rtg = decision_transformer_policy.variant['eval_rtg']
    DT_use_mean = True
    DT_target_return = [DT_eval_rtg * decision_transformer_policy.reward_scale] * env.num_envs
    state_dim = decision_transformer_policy.state_dim
    act_dim = decision_transformer_policy.act_dim
    device = model.device
    assert len(DT_target_return) == env.num_envs

    decision_transformer_policy.eval()

    state_mean = th.from_numpy(decision_transformer_policy.state_mean).to(device=device)
    state_std = th.from_numpy(decision_transformer_policy.state_std).to(device=device)
    num_envs = env.num_envs

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    DT_states = (
        th.from_numpy(observations)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=th.float32)
    ).reshape(num_envs, -1, state_dim)
    DT_actions = th.zeros(0, device=device, dtype=th.float32)
    DT_rewards = th.zeros(0, device=device, dtype=th.float32)

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

    t = 0
    while (episode_counts < episode_count_targets).any():
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
            DT_state_pred, DT_action_dist, DT_reward_pred = decision_transformer_policy.get_predictions(
                (DT_states.to(dtype=th.float32) - state_mean) / state_std,
                DT_actions.to(dtype=th.float32),
                DT_rewards.to(dtype=th.float32),
                DT_target_return.to(dtype=th.float32),
                DT_timesteps.to(dtype=th.long),
                num_envs=num_envs,
            )
            DT_state_pred = DT_state_pred.detach().cpu().numpy().reshape(num_envs, -1)
            if decision_transformer_policy.return_embedding:
                DT_reward_pred = DT_reward_pred.detach().cpu().numpy().reshape(num_envs)

            # the return action is a SquashNormal distribution
            assert decision_transformer_augment_options.number_of_states_in_aug_policy == 1
            assert decision_transformer_augment_options.number_of_actions_in_aug_policy == 1
            DT_action = DT_action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
            # if self.decision_transformer_augment_options.use_mean_DT_action:
            if DT_use_mean:
                DT_action = DT_action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
            DT_action = DT_action.clamp(*decision_transformer_policy.action_range)

        DT_action_np = DT_action.detach().cpu().numpy()
        observations_to_model = np.concatenate((observations, DT_action_np), axis = -1)
        actions, states = model.predict(observations_to_model, state=states, episode_start=episode_starts,
                                        deterministic=deterministic)
        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions  before adding to the decision transformer actions
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions,
                                      env.action_space.low * decision_transformer_augment_options.aug_action_range,
                                      env.action_space.high * decision_transformer_augment_options.aug_action_range)

        if decision_transformer_augment_options.add_DT_action_to_aug_policy:
            clipped_actions = clipped_actions + DT_action_np
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_actions = np.clip(clipped_actions, env.action_space.low, env.action_space.high)


        observations, rewards, dones, infos = env.step(clipped_actions)
        current_rewards += rewards
        if np.count_nonzero(current_rewards) == len(current_rewards) and laptime_override_stop:
            dones = np.full_like(dones, True)
        DT_state = observations
        DT_reward = rewards
        DT_done = dones
        DT_infos = infos

        if 'phase_action' in infos[0]:
            asdf
            taken_actions = np.array([info['phase_action'] for info in infos])
            DT_action = th.from_numpy(taken_actions.reshape(DT_action_np.shape)).to(device=device)
        else:  # relabe actions after being sent to environment
            DT_action = th.from_numpy(clipped_actions.reshape(DT_action_np.shape)).to(device=device)

        DT_episode_return[DT_unfinished] += DT_reward[DT_unfinished].reshape(-1, 1)

        DT_actions[:, -1] = DT_action
        DT_state = (
            th.from_numpy(DT_state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        DT_states = th.cat([DT_states, DT_state], dim=1)
        DT_reward = th.from_numpy(DT_reward).to(device=device).reshape(num_envs, 1)
        DT_rewards[:, -1] = DT_reward

        if decision_transformer_policy.mode != "delayed":
            DT_reward_scale = decision_transformer_policy.reward_scale
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

        t += 1

        if np.any(DT_done):
            ind = np.where(DT_done)[0]
            DT_unfinished[ind] = False
            DT_episode_length[ind] = np.minimum(DT_episode_length[ind], t + 1)




        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class DecisionTransformerEvalCallback(EvalCallback):
    def __init__(
        self,
        decision_transformer_policy,
        decision_transformer_augment_options,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best = None,
        callback_after_eval = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        laptime_recorder = False,
    ):
        self.decision_transformer_policy = decision_transformer_policy
        self.decision_transformer_augment_options = decision_transformer_augment_options
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.laptime_recorder = laptime_recorder

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy_withDT(
                self.model,
                self.decision_transformer_policy,
                self.decision_transformer_augment_options,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                laptime_override_stop=self.laptime_recorder
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )


            if not self.laptime_recorder:
                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = mean_reward

                if self.verbose >= 1:
                    print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                # Add to current Logger
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record('eval/std_reward', float(std_reward))
                self.logger.record("eval/mean_ep_length", mean_ep_length)

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose >= 1:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

                if mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    self.best_mean_reward = mean_reward
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()
            else:
                mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)

                mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
                self.last_mean_reward = mean_reward

                self.logger.record("eval_laptime/finished_laps", float(np.count_nonzero(episode_rewards)))
                if float(np.count_nonzero(episode_rewards)) > 0:
                    laptimes = [r for r in episode_rewards if r>0]
                    self.logger.record("eval_laptime/average_laptime", np.mean(laptimes))
                    self.logger.record("eval_laptime/std_laptime", np.std(laptimes))
                    self.logger.record("eval_laptime/laptime_count", len(laptimes))

                if len(self._is_success_buffer) > 0:
                    success_rate = np.mean(self._is_success_buffer)
                    if self.verbose >= 1:
                        print(f"Success rate: {100 * success_rate:.2f}%")
                    self.logger.record("eval/success_rate", success_rate)

                # Dump log so the evaluation results are printed with the correct timestep
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(self.num_timesteps)

                if mean_reward > self.best_mean_reward:
                    if self.verbose >= 1:
                        print("New best mean reward!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, "best_model_laptime"))
                    self.best_mean_reward = mean_reward
                    # Trigger callback on new best model, if needed
                    if self.callback_on_new_best is not None:
                        continue_training = self.callback_on_new_best.on_step()

                # Trigger callback after every evaluation, if needed
                if self.callback is not None:
                    continue_training = continue_training and self._on_event()

        return continue_training
