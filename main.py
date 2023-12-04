import gym
import numpy as np
import os
import json
from dataclasses import asdict
import argparse

from utils.gym_utils import sample_expert_transitions, sample_expert_trajectories
from run_adversarial import run_adversarial
from utils.config import ImitationExpConfig, DecisionTransformerConfig
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper

from utils.discriminator_augment import DiscAugmentListConfig, DiscAugmentGradientPenaltyConfig, DiscAugmentEntropyConfig

from online_dt.train_online_DT import Experiment
from online_dt.bet_config import BeTConfig

def run_AIL():
    env = gym.make('MountainCarContinuous-v0')
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    rng = np.random.default_rng(0)
    disc_aug = DiscAugmentListConfig(config_list=[DiscAugmentGradientPenaltyConfig(), DiscAugmentEntropyConfig()])
    exp_config = ImitationExpConfig(disc_augment_reward=disc_aug)

    transitions = sample_expert_transitions(venv, rng, dir=exp_config.logging_directory + '/rl')
    run_adversarial(venv, exp_config, transitions, )



def run_BeT():
    env = gym.make('MountainCarContinuous-v0')
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    rng = np.random.default_rng(0)
    disc_aug = DiscAugmentListConfig(config_list=[DiscAugmentGradientPenaltyConfig(), DiscAugmentEntropyConfig()])
    exp_config = ImitationExpConfig(disc_augment_reward=disc_aug)

    bet_config = BeTConfig()
    transitions = sample_expert_trajectories(venv, rng, dir=exp_config.logging_directory + '/rl',
                                             expert_timesteps=22_000)
    experiment = Experiment(asdict(bet_config), env = venv, eval_env=venv, transitions=transitions)
    onlinedt_config = os.path.join(experiment.logger.log_path, 'online_dt_config.json')
    with open(onlinedt_config, 'w') as f:
       f.write(json.dumps(asdict(bet_config)))

    print("=" * 50)
    bet_path = experiment(bet_config)
    return onlinedt_config, bet_path, transitions


def run_BeT_AIL(onlinedt_config: str, bet_path: str, transitions = None):
    env = gym.make('MountainCarContinuous-v0')
    venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    rng = np.random.default_rng(0)
    disc_aug = DiscAugmentListConfig(config_list=[DiscAugmentGradientPenaltyConfig(), DiscAugmentEntropyConfig()])
    bet_aug_config = DecisionTransformerConfig(enable = True, path_to_model = bet_path, path_to_exp_config=onlinedt_config,
                                               aug_action_range=0.1)# aug_action_range is alpha in the paper
    exp_config = ImitationExpConfig(disc_augment_reward=disc_aug, decision_transformer_config=bet_aug_config)

    transitions = sample_expert_transitions(venv, rng, dir=exp_config.logging_directory + '/rl', )
    run_adversarial(venv, exp_config, transitions, )




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default='BeT-AIL')
    args = parser.parse_args()

    if args.algorithm == 'BeT-AIL':
        onlinedt_config, bet_path, transitions = run_BeT()
        run_BeT_AIL(onlinedt_config, bet_path, transitions=transitions)
    elif args.algorithm == 'AIL':
        run_AIL()
    elif args.algorithm == 'BeT':
        run_BeT()
    else:
        raise NotImplementedError