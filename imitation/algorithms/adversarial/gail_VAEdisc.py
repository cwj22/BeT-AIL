# from .gail import GAIL, RewardNetFromDiscriminatorLogit, RewardNetSum
# from .common import AdversarialTrainer
# from imitation.rewards import reward_nets
# import torch as th
# from typing import Optional, List
#
# import torch as th
# from stable_baselines3.common import base_class, vec_env
# from torch.nn import functional as F
#
# from imitation.algorithms import base
# from imitation.algorithms.adversarial import common
# from imitation.rewards import reward_nets
# import torch as th
#
#
# class GAILVAE(AdversarialTrainer):
#     def __init__(
#         self,
#         *,
#         demonstrations: base.AnyTransitions,
#         demo_batch_size: int,
#         venv: vec_env.VecEnv,
#         gen_algo: base_class.BaseAlgorithm,
#         reward_net: reward_nets.RewardNet,
#         wasserstein_gail: bool,
#         wasserstein_gail_reward: str = 'exp',
#         **kwargs,
#     ):
#         """Generative Adversarial Imitation Learning.
#
#         Args:
#             demonstrations: Demonstrations from an expert (optional). Transitions
#                 expressed directly as a `types.TransitionsMinimal` object, a sequence
#                 of trajectories, or an iterable of transition batches (mappings from
#                 keywords to arrays containing observations, etc).
#             demo_batch_size: The number of samples in each batch of expert data. The
#                 discriminator batch size is twice this number because each discriminator
#                 batch contains a generator sample for every expert sample.
#             venv: The vectorized environment to train in.
#             gen_algo: The generator RL algorithm that is trained to maximize
#                 discriminator confusion. Environment and logger will be set to
#                 `venv` and `custom_logger`.
#             reward_net: a Torch module that takes an observation, action and
#                 next observation tensor as input, then computes the logits.
#                 Used as the GAIL discriminator.
#             **kwargs: Passed through to `AdversarialTrainer.__init__`.
#         """
#         # Raw self._reward_net is discriminator logits
#         self.wasserstein_gail = wasserstein_gail
#
#         if isinstance(reward_net, List):
#             reward_net = [rr.to(gen_algo.device) for rr in reward_net]
#             self._processed_reward = RewardNetSum(reward_net)
#             assert not wasserstein_gail
#         elif wasserstein_gail:
#             reward_net = reward_net.to(gen_algo.device)
#             self._processed_reward = reward_net
#         else:
#
#             reward_net = reward_net.to(gen_algo.device)
#             # Process it to produce output suitable for RL training
#             # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
#             self._processed_reward = RewardNetFromDiscriminatorLogit(reward_net)
#
#
#         super().__init__(
#             demonstrations=demonstrations,
#             demo_batch_size=demo_batch_size,
#             venv=venv,
#             gen_algo=gen_algo,
#             reward_net=th.nn.Sequential(encoder_net, reward_net),
#             wasserstein_gail = wasserstein_gail,
#             **kwargs,
#         )
#
#     def logits_expert_is_high(
#         self,
#         state: th.Tensor,
#         action: th.Tensor,
#         next_state: th.Tensor,
#         done: th.Tensor,
#         log_policy_act_prob: Optional[th.Tensor] = None,
#     ) -> th.Tensor:
#         r"""Compute the discriminator's logits for each state-action sample.
#
#         Args:
#             state: The state of the environment at the time of the action.
#             action: The action taken by the expert or generator.
#             next_state: The state of the environment after the action.
#             done: whether a `terminal state` (as defined under the MDP of the task) has
#                 been reached.
#             log_policy_act_prob: The log probability of the action taken by the
#                 generator, :math:`\log{P(a|s)}`.
#
#         Returns:
#             The logits of the discriminator for each state-action sample.
#         """
#         del log_policy_act_prob
#         if isinstance(self._reward_net, List):
#             logits = 0.0
#             for reward_net in self._reward_net:
#                 logits = logits + reward_net(state, action, next_state, done)
#         else:
#             logits = self._reward_net(state, action, next_state, done)
#         assert logits.shape == state.shape[:1]
#         return logits
#
#
#
#     def set_reward_train(self, reward_train):
#         self._processed_reward = reward_train
#
#     def set_reward_test(self, reward_test):
#         self._processed_reward = reward_test
#
#     @property
#     def reward_train(self) -> reward_nets.RewardNet:
#         return self._processed_reward
#
#     @property
#     def reward_test(self) -> reward_nets.RewardNet:
#         return self._processed_reward
#
#     @property
#     def reward_net_decoder(self) -> reward_nets.RewardNet:
#         return self._reward_net_decoder
#
#     @property
#     def reward_net_decoder_processed(self) -> reward_nets.RewardNet:
#         return self._reward_net_decoder_processed
#
#     @property
#     def encoder_net(self) -> reward_nets.RewardNet:
#         return self._encoder_net
#
