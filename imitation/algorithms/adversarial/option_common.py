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
import datetime

from imitation.algorithms import base
from imitation.data import buffer, rollout, types, wrappers
from imitation.sb3.option import option_buffer
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import logger, networks, util

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from imitation.algorithms.bc import BehaviorCloningLossCalculator
from stable_baselines3.common.buffers import ReplayBuffer as SB3Buffer
from stable_baselines3.common.type_aliases import Schedule
from imitation.algorithms.adversarial.common import DiscLossAugmentationEntropy, DiscLossAugmentationVariance, compute_train_stats
from .option_reward_nets import OptionRewardNet, OptionBaseRewardNet

def formatted_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


class OptionDiscLossAugmentationGradPen:
    def __init__(self, base_reward_net: OptionBaseRewardNet, grad_pen_scale: float = 10.,
                 grad_pen_targ: float = 1., grad_pen_type: str = 'wgan',
                 one_sided_pen: bool = True):
        self.grad_pen_scale = grad_pen_scale
        self.grad_pen_targ = grad_pen_targ
        self.grad_pen_type = grad_pen_type
        self.one_sided_pen = one_sided_pen
        self.base = base_reward_net
        num_inputs = 0
        if self.base.use_state:
            num_inputs += 1
        if self.base.use_action:
            num_inputs += 1
        if self.base.use_next_state:
            num_inputs += 1
        if self.base.use_done:
            num_inputs += 1
        # if self.base.use_option:
        #     num_inputs += 1
        # if self.base.use_next_option:
        #     num_inputs += 1

        self.num_inputs = num_inputs
        self.device = self.base.device

    # https: // github.com / lionelblonde / liayn - pytorch / blob / 39
    # a10c88c81b240e3bcf16107a21b4bbd7827751 / agents / sam_agent.py
    def __call__(self,
                 gen_state: th.Tensor,
                 gen_action: th.Tensor,
                 gen_next_state: th.Tensor,
                 gen_done: th.Tensor,
                 gen_option: th.Tensor,
                 gen_next_option: th.Tensor,
                 exp_state: th.Tensor,
                 exp_action: th.Tensor,
                 exp_next_state: th.Tensor,
                 exp_done: th.Tensor,
                 exp_option: th.Tensor,
                 exp_next_option: th.Tensor,
                 ):
        self.device = gen_state.device
        gen_done = gen_done.view(-1, 1)
        gen_option = gen_option.view(-1, 1)
        gen_next_option = gen_next_option.view(-1, 1)
        exp_done = exp_done.view(-1, 1)
        exp_option = exp_option.view(-1, 1)
        exp_next_option = exp_next_option.view(-1, 1)

        if self.grad_pen_type == 'wgan':
            eps_s = th.rand(gen_state.size(0), 1).to(self.device)
            eps_a = th.rand(gen_action.size(0), 1).to(self.device)
            eps_ns = th.rand(gen_next_state.size(0), 1).to(self.device)
            eps_d = th.rand(gen_done.size(0), 1).to(self.device)
            eps_o = th.rand(gen_option.size(0), 1).to(self.device)
            eps_no = th.rand(gen_next_option.size(0), 1).to(self.device)
            input_s = eps_s * gen_state + ((1. - eps_s) * exp_state)
            input_a = eps_a * gen_action + ((1. - eps_a) * exp_action)
            input_ns = eps_ns * gen_next_state + ((1. - eps_ns) * exp_next_state)
            input_d = eps_d * gen_done + ((1. - eps_d) * exp_done)
            input_o = eps_o * gen_option + ((1. - eps_o) * exp_option)
            input_no = eps_no * gen_next_option + ((1. - eps_no) * exp_next_option)
        elif self.grad_pen_type == 'dragan' or self.grad_pen_type == 'nagard':
            eps_s = gen_state.clone().detach().data.normal_(0, 10)
            eps_a = gen_action.clone().detach().data.normal_(0, 10)
            eps_ns = gen_next_state.clone().detach().data.normal_(0, 10)
            eps_d = gen_done.clone().detach().data.normal_(0, 10)
            eps_o = gen_option.clone().detach().data.normal_(0, 10)
            eps_no = gen_next_option.clone().detach().data.normal_(0, 10)
            if self.grad_pen_type == 'dragan':
                input_s = eps_s + exp_state
                input_a = eps_a + exp_action
                input_ns = eps_ns + exp_next_state
                input_d = eps_d + exp_done
                input_o = eps_o + exp_option
                input_no = eps_ns + exp_next_option
            else:
                input_s = eps_s + gen_state
                input_a = eps_a + gen_action
                input_ns = eps_ns + gen_next_state
                input_d = eps_d + gen_done
                input_o = eps_o + gen_option
                input_no = eps_ns + gen_next_option
        elif self.grad_pen_type == 'bare':
            input_s = Variable(gen_state, requires_grad=True)
            input_a = Variable(gen_action, requires_grad=True)
            input_ns = Variable(gen_next_state, requires_grad=True)
            input_d = Variable(gen_done, requires_grad=True)
            input_o = Variable(gen_option, requires_grad=True)
            input_no = Variable(gen_next_option, requires_grad=True)


        else:
            raise NotImplementedError("invalid gradient penalty type")
        input_s.requires_grad = True
        input_a.requires_grad = True
        input_ns.requires_grad = True
        input_d.requires_grad = True
        input_o.requires_grad = True
        input_no.requires_grad = True

        score = self.base(input_s, input_a, input_ns, input_d, input_o, input_no)
        inputs = self._get_inputs(input_s, input_a, input_ns, input_d, input_o, input_no)
        # Get the gradient of this operation with respect to its inputs
        grads = th.autograd.grad(
            outputs=score,
            inputs=inputs,
            only_inputs=True,
            grad_outputs=[th.ones_like(score)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert len(list(grads)) == self.num_inputs
        grads = th.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)

        if self.grad_pen_type == 'bare':
            # Penalize the gradient for having a norm GREATER than k
            _grad_pen = th.max(th.zeros_like(grads_norm), grads_norm - self.grad_pen_targ).pow(2)
            return _grad_pen * self.grad_pen_scale
        else:
            if self.one_sided_pen:
                # Penalize the gradient for having a norm GREATER than k
                _grad_pen = th.max(th.zeros_like(grads_norm), grads_norm - self.grad_pen_targ).pow(2)
            else:
                # Penalize the gradient for having a norm LOWER OR GREATER than k
                _grad_pen = (grads_norm - self.grad_pen_targ).pow(2)
            grad_pen = _grad_pen.mean()
            return grad_pen * self.grad_pen_scale

    def _get_inputs(self, input_s, input_a, input_ns, input_d, input_o, input_no):
        inputs = []
        if self.base.use_state:
            inputs.append(input_s)
        if self.base.use_action:
            inputs.append(input_a)
        if self.base.use_next_state:
            inputs.append(input_ns)
        if self.base.use_done:
            inputs.append(input_d)
        # do not append options since they are not part of the graph - these are indexes
        # if self.base.use_option:
        #     inputs.append(input_o)
        # if self.base.use_next_option:
        #     inputs.append(input_no)
        return inputs




class ReplayBufferWrapperSB3:
    def __init__(self, replay_buffer: SB3Buffer):
        self._buffer = replay_buffer

    def store(self, transitions: types.Transitions, truncate_ok: bool = True) -> None:
        self._buffer.store(trans_dict, truncate_ok=truncate_ok)



class OptionAdversarialTrainer(base.DemonstrationAlgorithm[types.Transitions]):
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
        reward_net: OptionRewardNet,
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
        disc_augmented_loss: Optional[Union[List, DiscLossAugmentationEntropy]] = None,
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
        gen_policy_observation_extension: int = 0,
        # how much the observations of the policy are extended (not to be used by disc)
        wasserstein_gail: bool = False,
        gen_augmented_lambda: float = 0.0,  # augment with augment env reward
        gen_augmented_lambda_schedule: str = 'constant',
        residual_policy_scale: float = 1.0,
        replace_viterbi_with_segments: bool = False,
        expert_demo_data_loader_kwargs: Dict = None,
        task_phasing_config: None
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
        assert gen_policy_observation_extension == 0
        self.gen_policy_observation_extension = gen_policy_observation_extension
        assert isinstance(reward_net, OptionRewardNet)
        assert not wasserstein_gail
        assert gen_augmented_lambda == 0.0
        self.recompute_disc_reward = recompute_disc_reward
        self.replace_viterbi_with_segments = replace_viterbi_with_segments
        self.expert_demo_data_loader_kwargs = expert_demo_data_loader_kwargs

        self.disc_learning_starts = disc_learning_starts
        self.use_behavioral_cloning = use_behavioral_cloning

        self.task_phasing_config = task_phasing_config
        if self.task_phasing_config:
            assert task_phasing_config['enable']
            self.task_phase_current_beta = task_phasing_config['beta_start']
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

        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._reward_net = reward_net.to(gen_algo.device)
        self._log_dir = types.parse_path(log_dir)

        # Create graph for optimising/recording stats on discriminator
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self._reward_net.parameters(),
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
            self.venv_wrapped = reward_wrapper.OptionsVecEnvWrapper(
                self.venv_buffering,
                use_options=True,
                latent_option_dim=self.gen_algo.policy.latent_option_dim
            )
        else:
            self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                self.venv_buffering,
                reward_fn=self.reward_train.predict_processed,
                use_options=True,
                latent_option_dim=self.gen_algo.policy.latent_option_dim
            )
            self.gen_callback.append(self.venv_wrapped.make_log_callback())
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

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps

        self.use_policy_replay_buffer_for_gen = use_policy_replay_buffer_for_gen
        if self.use_policy_replay_buffer_for_gen:
            raise NotImplementedError
        else:
            # self._gen_replay_buffer = buffer.ReplayBuffer(
            #     gen_replay_buffer_capacity,
            #     self.venv,
            # )
            self._gen_replay_buffer = option_buffer.OptionReplayBuffer(
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

        if self._global_step % self.task_phase_step_every == 0 and self.task_phasing_config and self._global_step>0:
            self.task_phase_current_beta = self.venv.env.update_task_phase_beta()
            self.gen_algo.task_phase_current_beta = self.task_phase_current_beta
        return self.task_phase_current_beta


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
        option: th.Tensor,
        next_option: th.Tensor,
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
    def reward_train(self) -> OptionRewardNet:
        """Reward used to train generator policy."""

    @property
    @abc.abstractmethod
    def reward_test(self) -> OptionRewardNet:
        """Reward used to train policy at "test" time after adversarial training."""

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        if self.expert_demo_data_loader_kwargs is None:
            self.expert_demo_data_loader_kwargs = {}
        self._demo_data_loader = base.make_data_loader(
            demonstrations,
            self.demo_batch_size,
            data_loader_kwargs=self.expert_demo_data_loader_kwargs
        )
        self._endless_expert_iterator = util.endless_iter(self._demo_data_loader)

    def _next_expert_batch(self) -> Mapping:
        assert self._endless_expert_iterator is not None
        expert_batch = next(self._endless_expert_iterator)
        if not self.replace_viterbi_with_segments:
            expert_batch = self.add_option_to_expert_demo(expert_batch)
        return expert_batch

    def add_option_to_expert_demo(self, expert_batch, raise_flag = False):
        traj_len = self.expert_demo_data_loader_kwargs['expert_mini_batch']
        self.logger.info(formatted_datetime(), 'Starting Option Viterbi on expert demos', )

        options = []
        next_options = []
        log_probs = []
        if raise_flag:
            log_probs = []
        for i in range(0,expert_batch['obs'].shape[0], traj_len):
            driver_num = expert_batch['driver_num'][i:i+traj_len]
            if len(th.unique(driver_num)) == 1:
                option, log_prob= self.policy.option_viterbi(
                expert_batch['obs'][i:i+traj_len].to(self.gen_algo.device),
                expert_batch['acts'][i:i+traj_len].to(self.gen_algo.device))

                options.append(option[:-1].squeeze())
                next_options.append(option[1:].squeeze())
                log_probs.append(log_prob.broadcast_to(option[:-1].squeeze().shape))
            else:
                obs = expert_batch['obs'][i:i + traj_len].to(self.gen_algo.device)
                acts = expert_batch['acts'][i:i + traj_len].to(self.gen_algo.device)
                driver_inds = th.unique(driver_num).tolist()
                for d_i in driver_inds:
                    inds = th.where(driver_num == d_i)
                    option, log_prob = self.policy.option_viterbi(obs[inds], acts[inds])

                    options.append(option[:-1].squeeze(-1))
                    next_options.append(option[1:].squeeze(-1))
                    log_probs.append(log_prob.broadcast_to(option[:-1].squeeze(-1).shape))




        expert_batch['opts'] = th.concat(options)
        expert_batch['next_opts'] = th.concat(next_options)
        expert_batch['log_prob_option'] = th.concat(log_probs)
        self.logger.info(formatted_datetime(), 'Finished Option Viterbi on expert demos!')
        assert expert_batch['opts'].shape[0] == expert_batch['obs'].shape[0]
        return expert_batch


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
                    batch['gen_option'],
                    batch['gen_next_option'],
                    batch["log_policy_act_prob"],
                )
                disc_logits_exp = self.logits_expert_is_high(
                    batch["exp_state"],
                    batch["exp_action"],
                    batch["exp_next_state"],
                    batch["exp_done"],
                    batch['exp_option'],
                    batch['exp_next_option'],
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
                                                      batch['gen_option'],
                                                      batch['gen_next_option'],
                                                      batch["gen_labels_expert_is_one"],
                                                      batch["exp_state"],
                                                      batch["exp_action"],
                                                      batch["exp_next_state"],
                                                      batch["exp_done"],
                                                      batch['exp_option'],
                                                      batch['exp_next_option'],
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
            disc_logits = th.concat((disc_logits_exp, disc_logits_gen))
            labels_expert_is_one = th.concat((batch["exp_labels_expert_is_one"], batch["gen_labels_expert_is_one"]))
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    labels_expert_is_one,
                    loss,
                )
            for k in aug_losses:
                train_stats.update({k: float(aug_losses[k])})
            if not self.replace_viterbi_with_segments:
                train_stats.update({'expert_viterbi_log_prob': float(expert_samples['log_prob_option'].mean())})
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def _get_opts_from_buffer(self):
        next_opts = self.gen_algo.rollout_buffer.options
        options = np.zeros_like(next_opts)
        options[0] = self.gen_algo.rollout_buffer.latent_option_dim
        options[1:] = next_opts[:-1]
        return options, next_opts

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
        options, next_opts = self._get_opts_from_buffer()
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples, options, next_opts)

        if self.replay_buffer_expert_percentage is not None:
            raise NotImplementedError
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

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
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
        if self.use_behavioral_cloning:
            exp, _ = self._make_separate_expert_gen_samples(no_gen = True)
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            if self.use_behavioral_cloning:
                exp, _ = self._make_separate_expert_gen_samples(no_gen=True)
                self.reward_train.bc_reward = self.bc_reward(expert_demo_batch=exp)
            self.logger.info('\n\n', formatted_datetime(), ' Starting Gen Training... ')
            self.train_gen(self.gen_train_timesteps)
            self.logger.info(formatted_datetime(), ' Finished Gen Training... ')
            if self._global_step >= self.disc_learning_starts:
                self.logger.info(formatted_datetime(), ' Starting Disc Training... ')
                for i_d in range(self.n_disc_updates_per_round):
                    self.logger.info(formatted_datetime(),
                                     f' Disc Training round {i_d+1} of {self.n_disc_updates_per_round}')
                    with networks.training(self.reward_train):
                        # switch to training mode (affects dropout, normalization)
                        exp, gen = self._make_separate_expert_gen_samples()
                        self.train_disc(expert_samples=exp, gen_samples=gen)
                self.logger.info(formatted_datetime(), ' Finished Disc Training')

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
        option_th: th.Tensor,
        next_option_th: th.Tensor,
    ) -> Optional[th.Tensor]:
        """Evaluates the given actions on the given observations.

        Args:
            obs_th: A batch of observations.
            acts_th: A batch of actions.

        Returns:
            A batch of log policy action probabilities.
        """
        if isinstance(self.policy, policies.ActorCriticPolicy):
            raise NotImplemented # not implemented for option
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

        # expert_log_prob_option = expert_samples.pop('log_prob_option')

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

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
                option_th = th.as_tensor(option, device=self.gen_algo.device)
                next_option_th = th.as_tensor(next_option, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th, option_th, next_option_th)
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
                option,
                next_option,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
                "option": option_th,
                "next_option": next_option_th,
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

        # expert_log_prob_option = expert_samples.pop('log_prob_option')

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

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

            obs_th, acts_th, next_obs_th, dones_th, opts_th, next_opts_th = self.reward_train.preprocess(
                gen_batch['obs'],
                gen_batch['acts'],
                gen_batch['next_obs'],
                gen_batch['dones'],
                gen_batch['opts'],
                gen_batch['next_opts'],
            )
            obs_th_exp, acts_th_exp, next_obs_th_exp, dones_th_exp, opts_th_exp, next_opts_th_exp = self.reward_train.preprocess(
                expert_batch['obs'],
                expert_batch['acts'],
                expert_batch['next_obs'],
                expert_batch['dones'],
                expert_batch['opts'],
                expert_batch['next_opts']
            )
            batch_dict = {
                "gen_state": obs_th,
                "gen_action": acts_th,
                "gen_next_state": next_obs_th,
                "gen_done": dones_th,
                "gen_option": opts_th,
                "gen_next_option": next_opts_th,
                "gen_labels_expert_is_one": self._torchify_array(np.zeros(self.demo_minibatch_size, dtype=int)),
                "exp_state": obs_th_exp,
                "exp_action": acts_th_exp,
                "exp_next_state": next_obs_th_exp,
                "exp_done": dones_th_exp,
                "exp_option": opts_th_exp,
                "exp_next_option": next_opts_th_exp,
                "exp_labels_expert_is_one": self._torchify_array(np.ones(self.demo_minibatch_size, dtype=int)),
                "log_policy_act_prob": None,
            }

            yield batch_dict

    def augmented_disc_loss(self,
                            gen_state: th.Tensor,
                            gen_action: th.Tensor,
                            gen_next_state: th.Tensor,
                            gen_done: th.Tensor,
                            gen_option: th.Tensor,
                            gen_next_option: th.Tensor,
                            gen_labels_expert_is_one: th.Tensor,
                            exp_state: th.Tensor,
                            exp_action: th.Tensor,
                            exp_next_state: th.Tensor,
                            exp_done: th.Tensor,
                            exp_option: th.Tensor,
                            exp_next_option: th.Tensor,
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
                elif isinstance(disc_augmented_loss, OptionDiscLossAugmentationGradPen):
                    losses['disc_aug_loss_grad_pen'] = disc_augmented_loss(gen_state, gen_action, gen_next_state, gen_done,
                                                                           gen_option, gen_next_option,
                                                                           exp_state, exp_action, exp_next_state, exp_done,
                                                                           exp_option, exp_next_option)

                elif isinstance(disc_augmented_loss, DiscLossAugmentationEntropy):
                    losses['disc_aug_loss_entropy'] = disc_augmented_loss(disc_logits_gen, disc_logits_exp)

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
            # gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            # gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)
            gen_samples = self._gen_replay_buffer.sample(batch_size)
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

