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

class DiscLossAugmentationVariance:
    def __init__(self, augmented_lambda: float = 0.1, base_reward_net = None):
        self.augmented_lambda = augmented_lambda


    def __call__(self, gen_algo: SAC, train_reward: reward_nets.RewardNet,
                 state: th.Tensor,
                 action: th.Tensor,
                 next_state: th.Tensor,
                 done: th.Tensor,
                 ):
        # Advantage is r(s,a) + gamma*V(s') - V(s)
        # advantage is rewards + gamma*target_q_next - q_current
        # we want to minimize [Rhat(s,a) - V(s)]**2
        # Rhat = sum_t=l^T [gamma**(t-l) log(1-D(s_t, a_t))]

        # calculate V(s)
        with th.no_grad():
            # in SAC two q networks are used for critic and target critic
            # as in the optimization set of sac, we take the min over all critics
            current_q_values = th.cat(gen_algo.critic(state, action), dim=1)
            current_q_values, _ = th.min(current_q_values, dim=1, keepdim=False) # dont keep dim to match GAIL implementaiton

        # calculate Rhat(s,a) for one step
        generator_reward = train_reward(state, action, next_state, done)
        return F.mse_loss(generator_reward, current_q_values)

class DiscLossAugmentationEntropy:
    def __init__(self, base_reward_net: reward_nets.BasicRewardNet, ent_reg_scale = 0.001):
        self.base = base_reward_net
        self.ent_reg_scale = ent_reg_scale

    def __call__(self, disc_logits_gen, disc_logits_exp):
        disc_logits = th.cat([disc_logits_gen, disc_logits_exp], dim=0)
        entropy = F.binary_cross_entropy_with_logits(input=disc_logits, target=th.sigmoid(disc_logits))
        entropy_loss = -self.ent_reg_scale * entropy
        return entropy_loss

class DiscLossAugmentationGradPen:
    def __init__(self, base_reward_net: reward_nets.BasicRewardNet, grad_pen_scale: float = 10.,
                 grad_pen_targ: float = 1., grad_pen_type: str = 'wgan',
                 one_sided_pen: bool = True):
        self.grad_pen_scale = grad_pen_scale
        self.grad_pen_targ = grad_pen_targ
        self.grad_pen_type = grad_pen_type
        self.one_sided_pen = one_sided_pen
        self.base = base_reward_net
        if isinstance(self.base, List):
            num_inputs_list = []
            for base in self.base:
                num_inputs = 0
                if base.use_state:
                    num_inputs += 1
                if base.use_action:
                    num_inputs += 1
                if base.use_next_state:
                    num_inputs += 1
                if base.use_done:
                    num_inputs += 1
                num_inputs_list.append(num_inputs)
            self.num_inputs = num_inputs_list
            self.device = self.base[0].device

        else:

            num_inputs = 0
            if self.base.use_state:
                num_inputs += 1
            if self.base.use_action:
                num_inputs += 1
            if self.base.use_next_state:
                num_inputs += 1
            if self.base.use_done:
                num_inputs += 1
            self.num_inputs = num_inputs
            self.device = self.base.device

    # https: // github.com / lionelblonde / liayn - pytorch / blob / 39
    # a10c88c81b240e3bcf16107a21b4bbd7827751 / agents / sam_agent.py
    def __call__(self,
                 gen_state: th.Tensor,
                 gen_action: th.Tensor,
                 gen_next_state: th.Tensor,
                 gen_done: th.Tensor,
                 exp_state: th.Tensor,
                 exp_action: th.Tensor,
                 exp_next_state: th.Tensor,
                 exp_done: th.Tensor,
                 ):
        self.device = gen_state.device
        if self.grad_pen_type == 'wgan':
            eps_s = th.rand(gen_state.size(0), 1).to(self.device)
            eps_a = th.rand(gen_action.size(0), 1).to(self.device)
            eps_ns = th.rand(gen_next_state.size(0), 1).to(self.device)
            eps_d = th.rand(gen_done.size(0), 1).to(self.device)
            input_s = eps_s * gen_state + ((1. - eps_s) * exp_state)
            input_a = eps_a * gen_action + ((1. - eps_a) * exp_action)
            input_ns = eps_ns * gen_next_state + ((1. - eps_ns) * exp_next_state)
            input_d = eps_d * gen_done + ((1. - eps_d) * exp_done)
        elif self.grad_pen_type == 'dragan' or self.grad_pen_type == 'nagard':
            eps_s = gen_state.clone().detach().data.normal_(0, 10)
            eps_a = gen_action.clone().detach().data.normal_(0, 10)
            eps_ns = gen_next_state.clone().detach().data.normal_(0, 10)
            eps_d = gen_done.clone().detach().data.normal_(0, 10)
            if self.grad_pen_type == 'dragan':
                input_s = eps_s + exp_state
                input_a = eps_a + exp_action
                input_ns = eps_ns + exp_next_state
                input_d = eps_d + exp_done
            else:
                input_s = eps_s + gen_state
                input_a = eps_a + gen_action
                input_ns = eps_ns + gen_next_state
                input_d = eps_d + gen_done
        elif self.grad_pen_type == 'bare':
            input_s = Variable(gen_state, requires_grad=True)
            input_a = Variable(gen_action, requires_grad=True)
            input_ns = Variable(gen_next_state, requires_grad=True)
            input_d = Variable(gen_done, requires_grad=True)


        else:
            raise NotImplementedError("invalid gradient penalty type")
        input_s.requires_grad = True
        input_a.requires_grad = True
        input_ns.requires_grad = True
        input_d.requires_grad = True

        grad_pen = 0.0
        if isinstance(self.base, List):
            for base, num_inputs in zip(self.base, self.num_inputs):
                gp = self._calc_grad_norm(base, input_s, input_a, input_ns, input_d, num_inputs)
                grad_pen += gp
        else:
            grad_pen = self._calc_grad_norm(self.base, input_s, input_a, input_ns, input_d, self.num_inputs)

        return grad_pen

    def _calc_grad_norm(self, base, input_s, input_a, input_ns, input_d, num_inputs):
        score = base(input_s, input_a, input_ns, input_d)
        inputs = self._get_inputs(base, input_s, input_a, input_ns, input_d)

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
        assert len(list(grads)) == num_inputs
        grads = th.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)

        if self.grad_pen_type  == 'bare':
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

    def _get_inputs(self, base, input_s, input_a, input_ns, input_d):
        inputs = []
        if base.use_state:
            inputs.append(input_s)
        if base.use_action:
            inputs.append(input_a)
        if base.use_next_state:
            inputs.append(input_ns)
        if base.use_done:
            inputs.append(input_d)
        return inputs


class DiscAugmentVariationalKLConstraint:
    def __init__(self, base_reward_net: reward_nets.BasicRewardNet,
                 information_flow: float = 0.5,
                 beta: float = 0.1,
                 dual_descent_on_beta: bool = False,
                 beta_step_size: float = 1e-6,
                 update_beta_every_nstep: int = 1,
    ):
        self.base = base_reward_net
        self.update_beta_every_nstep = update_beta_every_nstep
        self.information_KL_target = information_flow
        self.dual_descent_on_beta = dual_descent_on_beta
        self.initial_beta = beta
        self.beta_step_size = beta_step_size
        self.beta = beta

    def __call__(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, dones: th.Tensor):
        mean, logstd = self.base.forward_encoder(state, action, next_state, dones, return_distribution = True)
        self.kl_loss = self._kl_loss(mean, logstd)
        return self.kl_loss * self.beta

    def update_beta(self, current_KL_value: th.Tensor, global_step: int):
        if global_step % self.update_beta_every_nstep == 0 and self.dual_descent_on_beta:
            with th.no_grad():
                new_beta = self.beta + self.beta_step_size * (current_KL_value - self.information_KL_target)
                self.beta = th.clamp(new_beta, min = 0.0)
        return self.beta

    def _kl_loss(self, mean, logstd):
        # formula here is (10) from AEVB paper. https://arxiv.org/pdf/1312.6114.pdf
        # std = tf.exp(logstd)
        std = th.exp(logstd)
        loss = 0.5 * th.sum(-1 - 2 * logstd + std ** 2 + mean**2, dim = -1)
        return th.mean(loss)



