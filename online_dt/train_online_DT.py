"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from torch.utils.tensorboard import SummaryWriter
# import pickle
import random
import time
# import gym
# import d4rl
import torch
import numpy as np
from copy import copy, deepcopy

import online_dt.utils_onlinedt as utils_onlinedt
from online_dt.replay_buffer import ReplayBuffer
from online_dt.lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from online_dt.data import create_dataloader
from online_dt.decision_transformer.models.decision_transformer import DecisionTransformer
from online_dt.evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from online_dt.trainer import SequenceTrainer
from online_dt.logger import Logger
from copy import deepcopy

MAX_EPISODE_LEN = 1000

class Experiment:
    def __init__(self, variant, env, eval_env, transitions,):
        self.env = env
        self.eval_env = eval_env
        self.variant = variant
        self.task_phase_current_beta = 0.0
        self.pretrain_outputs=[]

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(env)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(transitions)
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], deepcopy(self.offline_trajs))


        self.aug_trajs = []

        if torch.cuda.is_available():
            self.device = variant.get("device", "cuda")
        else:
            self.device = 'cpu'
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
            return_embedding=variant['return_embedding'],
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, env):
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")
            return f"{path_prefix}/pretrain_model.pt"
        else:
            return f"{path_prefix}/model.pt"

    def _load_model(self, path_prefix, device = None, set_rng = True):

        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f, map_location=device)
        elif Path(f"{path_prefix}").exists():
            with open(f"{path_prefix}", "rb") as f:
                checkpoint = torch.load(f, map_location=device)
        else:
            return False

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.log_temperature_optimizer.load_state_dict(
            checkpoint["log_temperature_optimizer_state_dict"]
        )
        self.pretrain_iter = checkpoint["pretrain_iter"]
        self.online_iter = checkpoint["online_iter"]
        self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
        if set_rng:
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
        print(f"Model loaded at {path_prefix}/model.pt")
        return True

    def _filter_reward(self, reward, filter_reward_value, filter_reward_steps = 2):
        max_reward= filter_reward_value
        for step in range(filter_reward_steps):
            inds = np.where(reward >= max_reward)[0]
            for ind in inds:
                if ind > 0:
                    reward[ind] = reward[ind - 1]
                else:
                    reward[ind] = reward[ind + 1]
        return reward


    def _load_dataset(self, transitions):
        trajectories = []
        for traj in transitions:
            path = {}
            path['observations'] = traj.obs
            path['rewards'] = traj.rews
            path['actions'] = traj.acts
            dones = np.zeros_like(traj.rews)
            dones[-1] = float(traj.terminal)
            path['dones'] = dones
            trajectories.append(copy(path))


        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)

        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        num_online_rollouts,
        randomized=False,
        override_use_mean_action=False,
    ):

        max_ep_len = MAX_EPISODE_LEN
        all_returns, all_lengths, all_trajs = [], [], []
        for n in range(num_online_rollouts):
            with torch.no_grad():
                # generate init state
                target_return = [target_explore * self.reward_scale] * online_envs.num_envs

                returns, lengths, trajs = vec_evaluate_episode_rtg(
                    online_envs,
                    self.state_dim,
                    self.act_dim,
                    self.model,
                    max_ep_len=max_ep_len,
                    reward_scale=self.reward_scale,
                    target_return=target_return,
                    mode="normal",
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    device=self.device,
                    use_mean=override_use_mean_action,
                )
                all_returns.extend(returns)
                all_lengths.extend(lengths)
                all_trajs.extend(trajs)

            self.replay_buffer.add_new_trajs(all_trajs)
            self.aug_trajs += all_trajs
            self.total_transitions_sampled += np.sum(all_lengths)

        return {
            "aug_traj/return_mean": np.mean(all_returns),
            "aug_traj/return_std": np.std(all_returns),
            "aug_traj/length": np.mean(all_lengths),
            "aug_traj/n_episodes": len(all_returns),
        }

    def pretrain(self, loss_fn):
        print("\n\n\n*** Pretrain ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            with torch.no_grad():
                eval_outputs, eval_reward = self.evaluate()
            outputs = {"time/total": time.time() - self.start_time}
            self.pretrain_outputs.append(copy(eval_outputs))

            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            save_path = self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1
        return save_path

    def evaluate(self,):
        eval_start = time.time()
        all_returns, all_lengths = [], []
        while len(all_returns) < self.variant['num_eval_episodes']-1:
            self.model.eval()
            vec_env = self.eval_env
            eval_rtg = self.variant["eval_rtg"]
            use_mean = True
            target_return = [eval_rtg * self.reward_scale] * vec_env.num_envs
            returns, lengths, _ = vec_evaluate_episode_rtg(
                vec_env,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=MAX_EPISODE_LEN,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
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

    def _update_task_phase_env(self, online_envs):
        if self.task_phasing_config.enable:
            self.task_phase_current_beta = online_envs.env.current_task_phase_beta
            if self.online_iter >= self.task_phasing_config.alpha_step_after_n_steps:
                if self.online_iter % self.task_phasing_config.alpha_step_every_n_steps == 0:
                    self.task_phase_current_beta = online_envs.env.update_task_phase_beta()
        return self.task_phase_current_beta

    def online_tuning(self, online_envs, loss_fn):
        online_envs.reset()
        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.online_iter < self.variant["max_online_iters"]:


            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                num_online_rollouts=self.variant["num_online_rollouts"],
                override_use_mean_action=self.variant['override_use_mean_action']
            )
            outputs.update(augment_outputs)

            if self.variant['percent_demo_in_online']>0:
                assert self.variant['percent_demo_in_online']<=1.0
                n_online_trajs = len(self.replay_buffer.trajectories)
                n_offline_trajs = n_online_trajs*self.variant['percent_demo_in_online']/(1-self.variant['percent_demo_in_online'])
                trajectories = deepcopy(self.replay_buffer.trajectories)
                if int(n_offline_trajs) >= len(self.offline_trajs):
                    trajectories.extend(self.offline_trajs)
                    outputs['aug_traj/percent_demo_in_online'] = len(self.offline_trajs)/(len(self.offline_trajs)+n_online_trajs)
                else:
                    trajectories.extend(random.sample(self.offline_trajs, int(n_offline_trajs)))
                    outputs['aug_traj/percent_demo_in_online'] = self.variant['percent_demo_in_online']

            else:
                trajectories = self.replay_buffer.trajectories

            dataloader = create_dataloader(
                trajectories=trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            if self.task_phasing_config.enable and self.variant['stop_entropy_update']:
                update_entropy = self.task_phase_current_beta >= 1.0
            else:
                update_entropy = True
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                update_entropy=update_entropy,
            )
            outputs.update(train_outputs)

            outputs['aug_traj/task_phase_beta'] = self._update_task_phase_env(online_envs)

            if evaluation:
                with torch.no_grad():
                    eval_outputs, eval_reward = self.evaluate()
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1

    def __call__(self, args):

        utils_onlinedt.set_seed_everywhere(args.seed)



        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        print("\n\nMaking Eval Env.....")
        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            save_path = self.pretrain(loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = self.env
            self.online_tuning(online_envs, loss_fn)

        return save_path

