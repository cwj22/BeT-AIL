
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.policies import MlpPolicy


from imitation.data import rollout

def train_expert(env, dir = None, timesteps = 50000):
    print("Training a expert.")
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log = dir,
                batch_size= 512,
                buffer_size= 50000,
                ent_coef =0.1,
                gamma = 0.9999,
                gradient_steps = 32,
                learning_rate = 0.0003,
                learning_starts = 0,
                # n_timesteps = 50000.0,
                policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]),
                tau = 0.01,
                train_freq = 32,
                use_sde = True,
                # normalize = False,
                )
    # parameters from https://huggingface.co/sb3/sac-MountainCarContinuous-v0
    model.learn(total_timesteps=timesteps,eval_env=env, eval_freq=1000)
    return model


def sample_expert_transitions(venv, rng, dir = None):
    expert = train_expert(venv, dir)

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)


def sample_expert_trajectories(venv, rng, dir = None, expert_timesteps = 50000):
    expert = train_expert(venv, dir, expert_timesteps)

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    return rollouts

