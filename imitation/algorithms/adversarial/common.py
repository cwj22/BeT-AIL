"""Core code for adversarial imitation learning, shared between GAIL and AIRL."""
import abc
import dataclasses
import logging
from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload, cast, Union, Dict, List

from copy import copy
import numpy as np
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, on_policy_algorithm, policies, vec_env, off_policy_algorithm
from stable_baselines3.sac import policies as sac_policies
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from imitation.algorithms.bc import BehaviorCloningLossCalculator
from stable_baselines3.common.buffers import ReplayBuffer as SB3Buffer
from stable_baselines3.common.type_aliases import Schedule

from imitation.algorithms.adversarial.discriminator_augments import (
DiscLossAugmentationVariance,
DiscLossAugmentationEntropy,
DiscLossAugmentationGradPen,
DiscAugmentVariationalKLConstraint,
)

def compute_train_stats(
    disc_logits_expert_is_high: th.Tensor,
    labels_expert_is_one: th.Tensor, 
    disc_loss: th.Tensor,
    aug_loss: th.Tensor = None,
) -> Mapping[str, float]:
    """Train statistics for GAIL/AIRL discriminator.

    Args:
        disc_logits_expert_is_high: discriminator logits produced by
            `AdversarialTrainer.logits_expert_is_high`.
        labels_expert_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        A mapping from statistic names to float values.
    """
    with th.no_grad():
        # Logits of the discriminator output; >0 for expert samples, <0 for generator.
        bin_is_generated_pred = disc_logits_expert_is_high < 0
        # Binary label, so 1 is for expert, 0 is for generator.
        bin_is_generated_true = labels_expert_is_one == 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = float(th.sum(int_is_generated_true))
        n_labels = float(len(labels_expert_is_one))
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels) if n_labels > 0 else float("NaN")
        n_expert_pred = int(n_labels - th.sum(int_is_generated_pred))
        if n_labels > 0:
            pct_expert_pred = n_expert_pred / float(n_labels)
        else:
            pct_expert_pred = float("NaN")
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        if n_expert < 1:
            expert_acc = float("NaN")
        else:
            # float() is defensive, since we cannot divide Torch tensors by
            # Python ints
            expert_acc = _n_pred_expert.item() / float(n_expert)

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / float(_n_gen_or_1)

        label_dist = th.distributions.Bernoulli(logits=disc_logits_expert_is_high)
        entropy = th.mean(label_dist.entropy())

    return {
        "disc_loss": float(th.mean(disc_loss)),
        "disc_acc": float(acc),
        "disc_acc_expert": float(expert_acc),  # accuracy on just expert examples
        "disc_acc_gen": float(generated_acc),  # accuracy on just generated examples
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        "disc_entropy": float(entropy),
        # true number of expert demos and predicted number of expert demos
        "disc_proportion_expert_true": float(pct_expert),
        "disc_proportion_expert_pred": float(pct_expert_pred),
        "n_expert": float(n_expert),
        "n_generated": float(n_generated),
        "disc_aug_loss": float(aug_loss) if aug_loss else 0.
    }




class ReplayBufferWrapperSB3:
    def __init__(self, replay_buffer: SB3Buffer):
        self._buffer = replay_buffer

    def store(self, transitions: types.Transitions, truncate_ok: bool = True) -> None:
        self._buffer.store(trans_dict, truncate_ok=truncate_ok)


class AdversarialTrainer(base.DemonstrationAlgorithm[types.Transitions]):
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    _demo_data_loader: Optional[Iterable[base.TransitionMapping]]
    _endless_expert_iterator: Optional[Iterator[base.TransitionMapping]]

    venv_wrapped: vec_env.VecEnvWrapper

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        demo_minibatch_size: Optional[int] = None,
        n_disc_updates_per_round: int = 2,
        log_dir: types.AnyPath = "output/",
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_train_timesteps: Optional[int] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        allow_variable_horizon: bool = False,
        gen_callback: Optional[BaseCallback] = None,
        log_interval: Optional[int] = 20,
        disc_augmented_loss: Optional[Union[List, DiscLossAugmentationVariance]] = None,
        use_behavioral_cloning: bool = False,
        bc_batch_size: Optional[int] = None, # defaults to demo_batch_size,
        bc_ent_weight: float = 1e-3,
        bc_l2_weight: float = 0.0,
        bc_use_disc_demo_batch: bool = True, # whether to resample new demos or use what the disc uses
        initialize_replay_buffer_with_demos: bool = False,
        use_policy_replay_buffer_for_gen: bool = False,
        replay_buffer_expert_percentage: Optional[Union[float, Schedule]] = None,
        disc_learning_starts: int = 1,
        recompute_disc_reward: bool = False,
        gen_policy_observation_extension: int = 0,# how much the observations of the policy are extended (not to be used by disc)
        gen_augmented_lambda: float = 0.0, # augment with augment env reward
        gen_augmented_lambda_schedule: str = 'constant',
        residual_policy_scale: float = 1.0,
        task_phasing_config: None,
        disc_label_smoothing: Optional[Dict]  = None,
    ):
        """Builds AdversarialTrainer.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensors as input and computes a reward signal.
            demo_minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until the entire batch is
                processed before making an optimization step. This is
                useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `demo_batch_size`.
                Optional, defaults to `demo_batch_size`.
            n_disc_updates_per_round: The number of discriminator updates after each
                round of generator updates in AdversarialTrainer.learn().
            log_dir: Directory to store TensorBoard logs, plots, etc. in.
            disc_opt_cls: The optimizer for discriminator training.
            disc_opt_kwargs: Parameters for discriminator training.
            gen_train_timesteps: The number of steps to train the generator policy for
                each iteration. If None, then defaults to the batch size (for on-policy)
                or number of environments (for off-policy).
            gen_replay_buffer_capacity: The capacity of the
                generator replay buffer (the number of obs-action-obs samples from
                the generator that can be stored). By default this is equal to
                `gen_train_timesteps`, meaning that we sample only from the most
                recent batch of generator samples.
            custom_logger: Where to log to; if None (default), creates a new logger.
            init_tensorboard: If True, makes various discriminator
                TensorBoard summaries.
            init_tensorboard_graph: If both this and `init_tensorboard` are True,
                then write a Tensorboard graph summary to disk.
            debug_use_ground_truth: If True, use the ground truth reward for
                `self.train_env`.
                This disables the reward wrapping that would normally replace
                the environment reward with the learned reward. This is useful for
                sanity checking that the policy training is functional.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.

        Raises:
            ValueError: if the batch size is not a multiple of the minibatch size.
        """
        self.residual_policy_scale = residual_policy_scale
        self.recompute_disc_reward = recompute_disc_reward
        self.disc_learning_starts = disc_learning_starts
        self.use_behavioral_cloning = use_behavioral_cloning
        self.gen_policy_observation_extension = gen_policy_observation_extension

        self.task_phasing_config = task_phasing_config
        if self.task_phasing_config:
            assert task_phasing_config['enable']
            self.task_phase_current_beta = gen_algo.task_phase_current_beta
            self.task_phase_beta_start = task_phasing_config['beta_start']
            self.task_phase_beta_end = task_phasing_config['beta_end']
            assert task_phasing_config['type'] == 'temporal_scaling'
            assert task_phasing_config['subtype'] == 'random_selection'
            self.task_phase_alpha = task_phasing_config['alpha']
            self.task_phase_step_every = task_phasing_config['alpha_step_every_n_steps']
            if task_phasing_config['commands_as_observations']:
                assert self.gen_policy_observation_extension == len(task_phasing_config['aug_obs_with_keys'])
            else:
                assert self.gen_policy_observation_extension == 0
        else:
            self.task_phase_beta_start = None
            self.task_phase_alpha = None
            self.task_phase_current_beta = None
            self.task_phase_beta_end = None
            self.task_phase_step_every = None


        if self.use_behavioral_cloning:
            if bc_batch_size:
                self.bc_batch_size = bc_batch_size
            else:
                self.bc_batch_size = demo_batch_size
            self.bc_loss_calculator = BehaviorCloningLossCalculator(ent_weight=bc_ent_weight,
                                                                    l2_weight=bc_l2_weight)
            self.bc_use_disc_demo_batch = bc_use_disc_demo_batch
            if self.bc_use_disc_demo_batch:
                assert self.bc_batch_size <= demo_batch_size
        if disc_augmented_loss:
            if isinstance(disc_augmented_loss, List):
                self.disc_augmented_loss = disc_augmented_loss
            else:
                self.disc_augmented_loss = [disc_augmented_loss]
        else:
            self.disc_augmented_loss = None

        if disc_label_smoothing:
            self.disc_label_smoothing = True
            self.smooth_expert_high = disc_label_smoothing['smooth_expert_high']
            self.smooth_expert_low = disc_label_smoothing['smooth_expert_low']
            self.smooth_gen_high = disc_label_smoothing['smooth_gen_high']
            self.smooth_gen_low = disc_label_smoothing['smooth_gen_low']
        else:
            self.disc_label_smoothing = False
            self.smooth_expert_high, self.smooth_expert_low = None, None
            self.smooth_gen_high, self.smooth_gen_low = None, None
        self.demo_batch_size = demo_batch_size
        self.demo_minibatch_size = demo_minibatch_size or demo_batch_size
        if self.demo_batch_size % self.demo_minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self._demo_data_loader = None
        self._endless_expert_iterator = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )


        self.n_disc_updates_per_round = n_disc_updates_per_round

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo

        self._log_dir = types.parse_path(log_dir)

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph

        if isinstance(reward_net, List):
            self._reward_net = [rr.to(gen_algo.device) for rr in reward_net]
            params = []
            for rr in self._reward_net:
                params = params + list(rr.parameters())
        else:
            self._reward_net = reward_net.to(gen_algo.device)
            params = self._reward_net.parameters()
        self._disc_opt = self._disc_opt_cls(
            params,
            **self._disc_opt_kwargs,
        )

        if self._init_tensorboard:
            logging.info(f"building summary directory at {self._log_dir}")
            summary_dir = self._log_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(str(summary_dir))

        self.venv_buffering = wrappers.BufferingWrapper(self.venv, error_on_premature_reset=False)

        self.gen_callback = gen_callback if gen_callback else []
        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = self.venv_buffering
        else:
            self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                self.venv_buffering,
                reward_fn=self.reward_train.predict_processed,
                gen_policy_observation_extension=gen_policy_observation_extension,
                gen_augmented_lambda = gen_augmented_lambda,
                gen_augmented_lambda_schedule=gen_augmented_lambda_schedule,
            )
            self.gen_callback.append(self.venv_wrapped.make_log_callback())
        if 'end_at_' in gen_augmented_lambda_schedule:
            self.end_gen_augmented_lambda = int(gen_augmented_lambda_schedule[7:])
            self.current_gen_aug_lambda = gen_augmented_lambda
        else:
            self.end_gen_augmented_lambda = None
            self.current_gen_aug_lambda = gen_augmented_lambda

        self.venv_train = self.venv_wrapped

        self.gen_algo.set_env(self.venv_train)
        self.gen_algo.set_logger(self.logger)

        if gen_train_timesteps is None:
            gen_algo_env = self.gen_algo.get_env()
            assert gen_algo_env is not None
            self.gen_train_timesteps = gen_algo_env.num_envs
            if isinstance(self.gen_algo, on_policy_algorithm.OnPolicyAlgorithm):
                self.gen_train_timesteps *= self.gen_algo.n_steps
        else:
            self.gen_train_timesteps = gen_train_timesteps

        if gen_algo.num_timesteps > 1:
            steps = gen_algo.n_envs * gen_algo.num_timesteps
            self._global_step = int(gen_algo.num_timesteps/steps)
            self._disc_step = int(gen_algo.num_timesteps/steps)
        else:
            self._global_step = 0
            self._disc_step = 0

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps

        self.use_policy_replay_buffer_for_gen = use_policy_replay_buffer_for_gen
        if self.use_policy_replay_buffer_for_gen:
            raise NotImplementedError
        else:
            if self.gen_algo.decision_transformer_augment_options.enable or self.gen_algo.residual_BC_augment_options.enable:
                self._gen_replay_buffer = buffer.ReplayBuffer(
                gen_replay_buffer_capacity,
                obs_shape=tuple(self.venv.venv.env.base_observation_space.shape),
                act_shape = tuple(self.venv.venv.env.base_action_space.shape),
                obs_dtype = self.venv.venv.env.base_observation_space.dtype,
                act_dtype = self.venv.venv.env.base_action_space.dtype,
                )
            else:
                self._gen_replay_buffer = buffer.ReplayBuffer(
                gen_replay_buffer_capacity,
                self.venv,
            )
        self.initialize_replay_buffer_with_demos = initialize_replay_buffer_with_demos
        if self.initialize_replay_buffer_with_demos:
            self._store_demos_in_policy_replay(demonstrations)
        self.replay_buffer_expert_percentage = replay_buffer_expert_percentage
        if isinstance(self.replay_buffer_expert_percentage, float):
            assert 0 <= self.replay_buffer_expert_percentage <= 1
            v = copy(self.replay_buffer_expert_percentage)
            self.replay_buffer_expert_percentage = lambda x: v


        self.log_interval = log_interval

    def _update_task_beta(self):
        self.task_phase_current_beta = self.venv.env.current_task_phase_beta
        if self._global_step % self.task_phase_step_every == 0 and self.task_phasing_config and self._global_step>0:
            self.task_phase_current_beta = self.venv.env.update_task_phase_beta()
            self.gen_algo.task_phase_current_beta = self.task_phase_current_beta
        return self.task_phase_current_beta

    def _update_gen_aug_lambda(self):
        if self.end_gen_augmented_lambda is not None:
            if self._global_step >= self.end_gen_augmented_lambda:
                self.current_gen_aug_lambda = self.venv.update_gen_augmented_lambda(value = 0.0)

        return self.current_gen_aug_lambda


    def _store_demos_in_policy_replay(self, demo, num_to_store: Optional[float] = None):
        n_envs = self.gen_algo.replay_buffer.n_envs
        if isinstance(demo, Dict):
            for k in demo:
                if isinstance(demo[k], th.Tensor) and not k == 'infos':
                    demo[k] = demo[k].detach().numpy()
            if num_to_store is not None:
                inds = np.random.choice(np.arange(0, demo['obs'].shape[0]), num_to_store, replace=False)
                for k in demo:
                    if not k == 'infos':
                        demo[k] = demo[k][inds]
            for i in range(0, demo['obs'].shape[0]-n_envs, n_envs):
                self.gen_algo.replay_buffer.add(
                    obs=demo['obs'][i:n_envs+i],
                    next_obs=demo['next_obs'][i:n_envs+i],
                    action=demo['acts'][i:n_envs+i],
                    reward=demo['rews'][i:n_envs+i],
                    done=demo['dones'][i:n_envs+i],
                    infos = demo['dones'][i:n_envs+i],
                )
        else:
            if num_to_store is not None:
                inds = np.random.choice(np.arange(0, demo.obs.shape[0]), num_to_store, replace=False)
                demo = demo[inds]
            for i in range(0, demo.obs.shape[0] - n_envs, n_envs):
                self.gen_algo.replay_buffer.add(
                    obs=demo.obs[i:n_envs+i],
                    next_obs=demo.next_obs[i:n_envs+i],
                    action=demo.acts[i:n_envs+i],
                    reward=demo.rews[i:n_envs+i],
                    done=demo.dones[i:n_envs+i],
                    infos = demo.dones[i:n_envs+i],
                )


    @property
    def policy(self) -> policies.BasePolicy:
        policy = self.gen_algo.policy
        assert policy is not None
        return policy

    @abc.abstractmethod
    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting expert, and a low value corresponds to
        predicting generator.

        Args:
            state: state at time t, of shape `(batch_size,) + state_shape`.
            action: action taken at time t, of shape `(batch_size,) + action_shape`.
            next_state: state at time t+1, of shape `(batch_size,) + state_shape`.
            done: binary episode completion flag after action at time t,
                of shape `(batch_size,)`.
            log_policy_act_prob: log probability of generator policy taking
                `action` at time t.

        Returns:
            Discriminator logits of shape `(batch_size,)`. A high output indicates an
            expert-like transition.
        """  # noqa: DAR202

    @property
    @abc.abstractmethod
    def reward_train(self) -> reward_nets.RewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> reward_nets.RewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)


    def _next_expert_batch(self) -> Mapping:
        assert self._endless_expert_iterator is not None
        return next(self._endless_expert_iterator)

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % self.log_interval == 0

            # compute loss
            self._disc_opt.zero_grad()

            batch_iter = self._make_separate_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            for batch in batch_iter:
                disc_logits_gen = self.logits_expert_is_high(
                    batch["gen_state"],
                    batch["gen_action"],
                    batch["gen_next_state"],
                    batch["gen_done"],
                    batch["log_policy_act_prob"],
                )
                disc_logits_exp = self.logits_expert_is_high(
                    batch["exp_state"],
                    batch["exp_action"],
                    batch["exp_next_state"],
                    batch["exp_done"],
                    batch["log_policy_act_prob"],
                )


                loss_gen = F.binary_cross_entropy_with_logits(
                    disc_logits_gen,
                    batch["gen_labels_expert_is_one"].float(),
                    reduction='none'
                )
                loss_disc = F.binary_cross_entropy_with_logits(
                    disc_logits_exp,
                    batch["exp_labels_expert_is_one"].float(),
                    reduction='none'
                )
                loss = (loss_disc + loss_gen).mean()
                aug_losses = self.augmented_disc_loss(batch["gen_state"],
                                                      batch["gen_action"],
                                                      batch["gen_next_state"],
                                                      batch["gen_done"],
                                                      batch["gen_labels_expert_is_one"],
                                                      batch["exp_state"],
                                                      batch["exp_action"],
                                                      batch["exp_next_state"],
                                                      batch["exp_done"],
                                                      batch["exp_labels_expert_is_one"],
                                                      disc_logits_gen,
                                                      disc_logits_exp,
                                                      )
                for k in aug_losses:
                    loss += aug_losses[k]

                # Renormalise the loss to be averaged over the whole
                # batch size instead of the minibatch size.
                assert len(batch["gen_state"]) == self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()



            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1


            # compute/write stats and TensorBoard data
            disc_logits = th.cat((disc_logits_exp, disc_logits_gen))
            labels_expert_is_one = th.cat((batch["exp_labels_expert_is_one"], batch["gen_labels_expert_is_one"]))
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    labels_expert_is_one,
                    loss,
                )
            for k in aug_losses:
                train_stats.update({k: float(aug_losses[k])})

            self.logger.record("global_step", self._global_step)
            if 'disc_aug_loss_KLContsr_VAE' in aug_losses:
                for disc_aug in self.disc_augmented_loss:
                    if isinstance(disc_aug, DiscAugmentVariationalKLConstraint):
                        with th.no_grad():
                            vaeloss = aug_losses['disc_aug_loss_KLContsr_VAE'] / disc_aug.beta
                            b = disc_aug.update_beta(vaeloss, self._disc_step)
                            train_stats["vae_const_beta"] = float(b)
                            train_stats['vae_const'] = float(vaeloss.item())


            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}
        if isinstance(self.gen_algo, off_policy_algorithm.OffPolicyAlgorithm):
            learn_kwargs['recalculate_rewards'] = self.reward_train

        if isinstance(self.gen_algo, on_policy_algorithm.OnPolicyAlgorithm):
            if self.gen_algo.bc_weight > 0:
                learn_kwargs['expert_batch'] = self._next_expert_batch()
                
        with self.logger.accumulate_means("gen"):

            if self.task_phasing_config:
                self.logger.record("global_step", self._global_step)
                b = self._update_task_beta()
                self.logger.record('task_phase_beta', b)
                c = self._update_gen_aug_lambda()
                self.logger.record('gen_aug_lambda', c)
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                log_interval=self.log_interval,
                **learn_kwargs,
            )
            self._global_step += 1


        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)
        if self.replay_buffer_expert_percentage is not None:
            exp_percent = self.replay_buffer_expert_percentage(self._global_step)
            n_exp_samples = int(gen_samples.obs.shape[0] * exp_percent/(1-exp_percent))
            expert_samples = self._next_expert_batch()
            if n_exp_samples <= self.demo_batch_size:
                self._store_demos_in_policy_replay(expert_samples, num_to_store = n_exp_samples)
            else:
                count = 0
                while count < n_exp_samples - self.demo_batch_size:
                    if count > 0:
                        expert_samples = self._next_expert_batch()
                    self._store_demos_in_policy_replay(expert_samples)
                    count += self.demo_batch_size
                expert_samples = self._next_expert_batch()
                self._store_demos_in_policy_replay(expert_samples, num_to_store=n_exp_samples-count)

    def set_steps(self, global_step):
        self._global_step = global_step

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
            steps_in_rollout: Optional[int] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        n_rounds = total_timesteps // self.gen_train_timesteps
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        if steps_in_rollout:
            if self.gen_train_timesteps < steps_in_rollout and self.gen_train_timesteps != 1:
                self.gen_train_timesteps *= steps_in_rollout

        if self.use_behavioral_cloning:
            exp, _ = self._make_separate_expert_gen_samples(no_gen = True)
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            if self.use_behavioral_cloning:
                exp, _ = self._make_separate_expert_gen_samples(no_gen=True)
                self.reward_train.bc_reward = self.bc_reward(expert_demo_batch=exp)
            if r == 0:
                self.train_gen(1)
            else:
                self.train_gen(self.gen_train_timesteps)
            if self._global_step >= self.disc_learning_starts:
                for _ in range(self.n_disc_updates_per_round):
                    with networks.training(self.reward_train):
                        # switch to training mode (affects dropout, normalization)
                        exp, gen = self._make_separate_expert_gen_samples()
                        self.train_disc(expert_samples=exp, gen_samples=gen)
            if callback:
                callback(r)

            if self._global_step % self.log_interval == 0:
                self.logger.dump(self._global_step)

    @overload
    def _torchify_array(self, ndarray: np.ndarray) -> th.Tensor:
        ...

    @overload
    def _torchify_array(self, ndarray: None) -> None:
        ...

    def _torchify_array(self, ndarray: Optional[np.ndarray]) -> Optional[th.Tensor]:
        if ndarray is not None:
            return th.as_tensor(ndarray, device=self.reward_train.device)
        return None

    def _get_log_policy_act_prob(
        self,
        obs_th: th.Tensor,
        acts_th: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Evaluates the given actions on the given observations.

        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.

        Returns:
            A batch of log policy action probabilities.
        """
        if isinstance(self.policy, policies.ActorCriticPolicy):
            # policies.ActorCriticPolicy has a concrete implementation of
            # evaluate_actions to generate log_policy_act_prob given obs and actions.
            _, log_policy_act_prob_th, _ = self.policy.evaluate_actions(
                obs_th,
                acts_th,
            )
        elif isinstance(self.policy, sac_policies.SACPolicy):
            gen_algo_actor = self.policy.actor
            assert gen_algo_actor is not None
            # generate log_policy_act_prob from SAC actor.
            mean_actions, log_std, _ = gen_algo_actor.get_action_dist_params(obs_th)
            distribution = gen_algo_actor.action_dist.proba_distribution(
                mean_actions,
                log_std,
            )
            # SAC applies a squashing function to bound the actions to a finite range
            # `acts_th` need to be scaled accordingly before computing log prob.
            # Scale actions only if the policy squashes outputs.
            assert self.policy.squash_output
            # scaled_acts_th = self.policy.scale_action(acts_th)
            scaled_acts_th = self.policy.scale_action(acts_th.cpu())
            scaled_acts_th = th.as_tensor(scaled_acts_th, device=self.gen_algo.device)
            log_policy_act_prob_th = distribution.log_prob(scaled_acts_th)
        else:
            return None
        return log_policy_act_prob_th

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and "
                "generator samples, each. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}
            if self.gen_policy_observation_extension != 0:
                gen_batch['obs'] = gen_batch['obs'][:, :-self.gen_policy_observation_extension]
                gen_batch['next_obs'] = gen_batch['next_obs'][:, :-self.gen_policy_observation_extension]
                if not self.task_phasing_config:
                    gen_batch['acts'] = gen_batch['acts'] *self.residual_policy_scale + gen_batch['obs'][:, -self.gen_policy_observation_extension:]

            # Concatenate rollouts, and label each row as expert or generator.
            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.demo_minibatch_size, dtype=int),
                    np.zeros(self.demo_minibatch_size, dtype=int),
                ],
            )

            # Calculate generator-policy log probabilities.
            with th.no_grad():
                obs_th = th.as_tensor(obs, device=self.gen_algo.device)
                acts_th = th.as_tensor(acts, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape(
                        (2 * self.demo_minibatch_size,),
                    )
                del obs_th, acts_th  # unneeded

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            yield batch_dict

    def _make_separate_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and "
                "generator samples, each. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            if self.gen_policy_observation_extension != 0:
                gen_batch['obs'] = gen_batch['obs'][:, :-self.gen_policy_observation_extension]
                gen_batch['next_obs'] = gen_batch['next_obs'][:, :-self.gen_policy_observation_extension]
                if not self.task_phasing_config:
                    gen_batch['acts'] = gen_batch['acts']*self.residual_policy_scale + gen_batch['obs'][:, -self.gen_policy_observation_extension:]

            # Concatenate rollouts, and label each row as expert or generator.
            # obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            # acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            # next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            # dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # # notice that the labels use the convention that expert samples are
            # # labelled with 1 and generator samples with 0.
            # labels_expert_is_one = np.concatenate(
            #     [
            #         np.ones(self.demo_minibatch_size, dtype=int),
            #         np.zeros(self.demo_minibatch_size, dtype=int),
            #     ],
            # )

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                gen_batch['obs'],
                gen_batch['acts'],
                gen_batch['next_obs'],
                gen_batch['dones'],
            )
            obs_th_exp, acts_th_exp, next_obs_th_exp, dones_th_exp = self.reward_train.preprocess(
                expert_batch['obs'],
                expert_batch['acts'],
                expert_batch['next_obs'],
                expert_batch['dones'],
            )

            if self.disc_label_smoothing:
                exp_labels_expert_is_one_np = np.random.uniform(self.smooth_expert_low, self.smooth_expert_high,
                                                                self.demo_minibatch_size)
                exp_labels_expert_is_one = self._torchify_array(exp_labels_expert_is_one_np)
                gen_labels_expert_is_one_np = np.random.uniform(self.smooth_gen_low, self.smooth_gen_high,
                                                                self.demo_minibatch_size)
                gen_labels_expert_is_one = self._torchify_array(gen_labels_expert_is_one_np)

            else:
                exp_labels_expert_is_one = self._torchify_array(np.ones(self.demo_minibatch_size, dtype=int))
                gen_labels_expert_is_one = self._torchify_array(np.zeros(self.demo_minibatch_size, dtype=int))

            batch_dict = {
                "gen_state": obs_th,
                "gen_action": acts_th,
                "gen_next_state": next_obs_th,
                "gen_done": dones_th,
                "gen_labels_expert_is_one": gen_labels_expert_is_one,
                "exp_state": obs_th_exp,
                "exp_action": acts_th_exp,
                "exp_next_state": next_obs_th_exp,
                "exp_done": dones_th_exp,
                "exp_labels_expert_is_one": exp_labels_expert_is_one,
                "log_policy_act_prob": None,
            }

            yield batch_dict

    def augmented_disc_loss(self,
                            gen_state: th.Tensor,
                            gen_action: th.Tensor,
                            gen_next_state: th.Tensor,
                            gen_done: th.Tensor,
                            gen_labels_expert_is_one: th.Tensor,
                            exp_state: th.Tensor,
                            exp_action: th.Tensor,
                            exp_next_state: th.Tensor,
                            exp_done: th.Tensor,
                            exp_labels_expert_is_one: th.Tensor,
                            disc_logits_gen: th.Tensor,
                            disc_logits_exp: th.Tensor,
                            ):
        losses = {}
        if self.disc_augmented_loss:
            for disc_augmented_loss in self.disc_augmented_loss:
                if isinstance(disc_augmented_loss, DiscLossAugmentationVariance):
                    raise NotImplementedError
                    # fix the batches to be separate now
                    assert isinstance(self.gen_algo, SAC)
                    losses['disc_variance_loss'] = disc_augmented_loss(self.gen_algo,
                                                    self.reward_train,
                                                    state,
                                                    action,
                                                    next_state,
                                                    done)
                elif isinstance(disc_augmented_loss, DiscLossAugmentationGradPen):
                    losses['disc_aug_loss_grad_pen'] = disc_augmented_loss(gen_state, gen_action, gen_next_state, gen_done,
                                                    exp_state, exp_action, exp_next_state, exp_done)

                elif isinstance(disc_augmented_loss, DiscLossAugmentationEntropy):
                    losses['disc_aug_loss_entropy'] = disc_augmented_loss(disc_logits_gen, disc_logits_exp)

                elif isinstance(disc_augmented_loss, DiscAugmentVariationalKLConstraint):
                    s = th.concat((exp_state, gen_state))
                    a = th.concat((exp_action, gen_action))
                    ns = th.concat((exp_next_state, gen_next_state))
                    d = th.concat((exp_done, gen_done))
                    vaeloss = disc_augmented_loss(s, a, ns, d)
                    losses['disc_aug_loss_KLContsr_VAE'] = vaeloss


                else:
                    raise NotImplementedError

        else:
            losses['aug_loss'] = 0.0
        return losses

    def _make_separate_expert_gen_samples(self, no_gen: bool = False):
        batch_size = self.demo_batch_size
        expert_samples = self._next_expert_batch()

        if not no_gen:

            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)
        else:
            gen_samples = None
        return expert_samples, gen_samples

    def bc_reward(self, expert_demo_batch: Optional[Mapping] = None) -> th.Tensor:
        # based on https://arxiv.org/pdf/1910.04281.pdf

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [expert_demo_batch]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(expert_demo_batch["obs"], np.ndarray)

        assert self.use_behavioral_cloning

        # get training expert data
        if self.bc_use_disc_demo_batch:
            assert expert_demo_batch is not None
        else:
            expert_demo_batch = self._next_expert_batch()

        inds = np.random.choice(np.arange(0, self.demo_batch_size), self.bc_batch_size, replace=False)
        expert_batch = {k: v[inds] for k, v in expert_demo_batch.items() if not 'infos' == k}
        obs = th.as_tensor(expert_batch["obs"], device=self.policy.device).detach()
        acts = th.as_tensor(expert_batch["acts"], device=self.policy.device).detach()
        training_metrics = self.bc_loss_calculator(self.gen_algo.policy, obs, acts)
        for k, v in training_metrics.__dict__.items():
            self.logger.record(f"bc/{k}", float(v))
        loss = training_metrics.loss

        return -loss # return reward

