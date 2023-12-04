import torch as th
import json
import pathlib
from dataclasses import asdict

from stable_baselines3.common.callbacks import CheckpointCallback
from utils.config import  SAC_Config
from sb3.sac_recalc_rew import SAC
from imitation.policies import serialize
from utils.config import ImitationExpConfig
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.common import (DiscLossAugmentationVariance,
                                                     DiscLossAugmentationEntropy,
                                                     DiscLossAugmentationGradPen)
from utils.discriminator_augment import DISC_AUGMENT_CONFIGS
def checkpoint_and_eval_callbacks(exp_config: ImitationExpConfig, venv_eval, gen_learner = None,
                                  remove_eval_callback=False):
    n_eval_episodes = exp_config.n_eval_episodes
    save_freq = exp_config.save_every_n_steps
    checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                             save_path=exp_config.logging_directory + "/checkpoints/",
                                             name_prefix= "model",
                                             save_replay_buffer=True)

    if exp_config.decision_transformer_config.enable:
        from gym_wrapper.eval_DT import DecisionTransformerEvalCallback
        eval_callback = DecisionTransformerEvalCallback(decision_transformer_policy=gen_learner.decision_transformer_policy,
                                                    decision_transformer_augment_options=gen_learner.decision_transformer_augment_options,
                                                    eval_env=venv_eval, n_eval_episodes=n_eval_episodes,
                             best_model_save_path=exp_config.logging_directory + '/models',
                             log_path=exp_config.logging_directory + '/logs',
                             eval_freq=int(exp_config.save_every_n_steps / 10),
                             deterministic=True, render=False)

    elif exp_config.residual_BC_config.enable:
        from gym_wrapper.eval_BC_callback import ResidualBCEvalCallback
        eval_callback = ResidualBCEvalCallback(residual_BC_policy=gen_learner.residual_BC_policy,
                                                    residual_BC_augment_options=gen_learner.residual_BC_augment_options,
                                                    eval_env=venv_eval, n_eval_episodes=n_eval_episodes,
                             best_model_save_path=exp_config.logging_directory + '/models',
                             log_path=exp_config.logging_directory + '/logs',
                             eval_freq=int(exp_config.save_every_n_steps / 10),
                             deterministic=True, render=False)

    else:
        from gym_wrapper.eval_callback import NewEvalCallback
        eval_callback = NewEvalCallback(venv_eval, n_eval_episodes=n_eval_episodes,
                             best_model_save_path=exp_config.logging_directory + '/models',
                             log_path=exp_config.logging_directory + '/logs',
                             eval_freq=int(exp_config.save_every_n_steps / 10),
                             deterministic=True, render=False)

    callbacks = [checkpoint_callback]
    callbacks.append(eval_callback)
    return callbacks



def create_policy_learner(exp_config: ImitationExpConfig, venv, expert_transitions=None):
    if not exp_config.gen_learner_config.policy_kwargs:
        policy_kwargs = {}
    else:
        policy_kwargs = exp_config.gen_learner_config.policy_kwargs
    if exp_config.gen_learner_config.policy_net_arch is not None:
        policy_kwargs['net_arch'] = [exp_config.gen_learner_config.policy_net_arch]

    if isinstance(exp_config.gen_learner_config, SAC_Config):
        if exp_config.decision_transformer_config.enable:
            assert not exp_config.iq_learn_config.enable
            from online_dt.train_online_DT import GTSExperiment

            filename = exp_config.decision_transformer_config.path_to_exp_config
            with open(filename, 'rb') as f:
                args = json.load(f)
            offline_trajs = load_trajectories_from_processed(TRAJ_FILES[exp_config.which_preprocessed_trajs])
            venv.venv.env.base_env.name = 'gym-gts'
            venv.venv.env.base_env.num_envs = venv.num_envs
            experiment = GTSExperiment(args, env=venv.venv.env.base_env, eval_env=venv.venv.env.base_env,
                                       transitions=offline_trajs,
                                       task_phasing_config=exp_config.task_phasing_config)
            experiment._load_model(path_prefix=exp_config.decision_transformer_config.path_to_model, device = experiment.device, set_rng = False)
            DT_policy = experiment.model
            DT_policy.state_mean = experiment.state_mean
            DT_policy.state_std = experiment.state_std
            DT_policy.variant = experiment.variant
            DT_policy.reward_scale = experiment.reward_scale
            DT_policy.mode = 'normal'

        else:
            DT_policy= None

        if exp_config.residual_BC_config.enable:
            from stable_baselines3.common.utils import get_device
            bc_policy_path = exp_config.residual_BC_config.path_to_model
            BC_policy = th.load(bc_policy_path, map_location=get_device("auto"))

        else:
            BC_policy = None

        if BC_policy is not None:
            assert DT_policy is None

        method = SAC
        kwargs = {}
        t_f = exp_config.gen_learner_config.train_freq
        exp_config.gen_learner_config.train_freq = (t_f, exp_config.gen_learner_config.train_freq_unit)
        gen_learner = method(
            **kwargs,
            env=venv,
            policy="MlpPolicy",
            tensorboard_log=exp_config.logging_directory,
            **exp_config.gen_learner_config.todict(remove_keys=['learning_rate_schedule',
                                                                'learning_rate_start', 'train_freq_unit',
                                                                'use_n_step_return',
                                                                'n_steps_return', 'policy_net_arch', ]),
            verbose=0,
            optimize_memory_usage=True,
            replay_buffer_kwargs=dict(
                                      handle_timeout_termination=False),
            decision_transformer_augment_options=exp_config.decision_transformer_config,
            decision_transformer_policy=DT_policy,
            residual_BC_policy=BC_policy,
            residual_BC_augment_options=exp_config.residual_BC_config,
        )
    else:
        raise NotImplementedError

    return gen_learner
def save(trainer: common.AdversarialTrainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.reward_train, save_path / "reward_train.pt")
    th.save(trainer.reward_test, save_path / "reward_test.pt")
    serialize.save_stable_model(
        save_path / "gen_policy",
        trainer.gen_algo,
    )


def gail_checkpoint_callback(exp_config, trainer):
    assert trainer is not None
    checkpoint_interval = exp_config.save_every_n_steps
    def callback(round_num: int, /) -> None:
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(trainer, pathlib.Path(exp_config.logging_directory + "/checkpoints/" + f"{round_num:05d}"))
    return callback



DISC_AUGMENT_LOSSES = {'DiscAugmentVarianceConfig': DiscLossAugmentationVariance,
                        'DiscAugmentGradientPenaltyConfig': DiscLossAugmentationGradPen,
                        'DiscAugmentEntropyConfig': DiscLossAugmentationEntropy,}


def create_disc_aug_loss(config, base_reward_net):
    aug_loss = None
    for k in DISC_AUGMENT_CONFIGS:
        if isinstance(config, DISC_AUGMENT_CONFIGS[k]):
            if k == 'DiscAugmentListConfig':
                aug_loss = []
                for item in config.config_list:
                    aug_loss.append(create_disc_aug_loss(item, base_reward_net))
            else:
                aug_loss = DISC_AUGMENT_LOSSES[k](base_reward_net = base_reward_net, **asdict(config))
    return aug_loss
