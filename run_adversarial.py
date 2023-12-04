from utils.config import ImitationExpConfig, SAC_Config
from utils.utils import checkpoint_and_eval_callbacks, create_policy_learner, gail_checkpoint_callback, create_disc_aug_loss
from imitation.data import types
from typing import Optional, List
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms.adversarial.gail import GAIL
from stable_baselines3.common.vec_env import DummyVecEnv


def run_adversarial(venv: DummyVecEnv,
             exp_config: ImitationExpConfig,
             trajs: Optional[List[types.TrajectoryWithRew]] = None,
             venv_eval = None,
             ):
    if venv_eval is None:
        venv_eval = venv
    gen_learner = create_policy_learner(exp_config, venv, trajs)

    callbacks = checkpoint_and_eval_callbacks(exp_config, venv_eval, gen_learner,)

    reward_net_kwargs = exp_config.reward_net_kwargs.to_dict()

    residual_policy_scale = 0
    if exp_config.decision_transformer_config.enable or exp_config.residual_BC_config.enable:
        obs = venv.venv.env.base_observation_space
    else:
        obs = venv.observation_space

    kw = {}

    reward_net = BasicRewardNet(
        obs, venv.action_space, normalize_input_layer=RunningNorm,
        **reward_net_kwargs)

    if exp_config.disc_augment_reward:
        kw['disc_augmented_loss'] = create_disc_aug_loss(exp_config.disc_augment_reward,
                                                         base_reward_net=reward_net,)

    log_interval = 1
    task_phasing_config = None
    gen_policy_observation_extension = 0

    ## Pick adversarial trainer and initialize
    adversarial_trainer = GAIL(
        demonstrations=trajs,
        demo_batch_size=exp_config.demo_batch_size,
        gen_replay_buffer_capacity=exp_config.gen_replay_buffer_capacity,
        n_disc_updates_per_round=exp_config.n_disc_updates_per_round,
        venv=venv,
        gen_algo=gen_learner,
        reward_net=reward_net,
        custom_logger=exp_config.logger,
        log_dir = exp_config.logging_directory,
        debug_use_ground_truth=exp_config.debug_use_ground_truth,
        gen_callback=callbacks,
        allow_variable_horizon=False,
        gen_train_timesteps=exp_config.gen_train_timesteps,
        disc_learning_starts=exp_config.disc_learning_starts,
        recompute_disc_reward=exp_config.recompute_disc_reward,
        disc_opt_kwargs = exp_config.disc_opt_kwargs,
        log_interval = log_interval,
        gen_policy_observation_extension = gen_policy_observation_extension,
        residual_policy_scale = residual_policy_scale,
        task_phasing_config=task_phasing_config,
        **kw,

    )
    gail_callback = gail_checkpoint_callback(exp_config, adversarial_trainer)


    ## Train
    env = gen_learner.get_env()
    env.reset()
    if isinstance(exp_config.gen_learner_config, SAC_Config):
        adversarial_trainer.train(exp_config.training_steps, gail_callback)
    else:
        steps_in_rollout = int(exp_config.env_config.num_cars*exp_config.env_config.episode_duration_step)
        adversarial_trainer.train(int(exp_config.training_steps/steps_in_rollout),
                                  gail_callback, steps_in_rollout)