from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
from copy import copy
import os
import datetime

from stable_baselines3.common.type_aliases import Schedule

from utils.RL_utils import linear_schedule
from imitation.util import logger as imit_logger
@dataclass
class LearnerConfig:
    learning_rate: Union[float, Schedule] = 3e-4 # learning rate or initial learning rate
    learning_rate_schedule: Optional[str] = None
    learning_rate_start: Optional[float] = 3e-4
    policy_kwargs: Optional[Dict] = None
    policy_net_arch: Optional[List] = None
    use_n_step_return: bool = False
    n_steps_return: int = 5
    seed: int = 0
    device: str = 'auto'


    def todict(self, remove_keys=None):
        if not self.learning_rate_schedule is None:
            if self.learning_rate_schedule == 'linear':
                self.learning_rate_start = copy(self.learning_rate)
                self.learning_rate = linear_schedule(self.learning_rate)
            elif self.learning_rate_schedule == 'constant':
                self.learning_rate_start = copy(self.learning_rate)
                pass
            else:
                raise NotImplementedError
        return_dict = asdict(self)
        for k in remove_keys:
            return_dict.pop(k)
        return return_dict


@dataclass
class SAC_Config(LearnerConfig):
    buffer_size: int = 1_000_000  # 1e6
    learning_starts: int = 100
    batch_size: int = 4096
    tau: float = 0.002
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = 1
    train_freq_unit: str = 'step'
    gradient_steps: int = 2500
    ent_coef: Union[str, float] = "auto"
    target_update_interval: int = 1
    target_entropy: Union[str, float] = "auto"
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False
    seed: int = 0
    policy_kwargs: Optional[Dict] = None


@dataclass
class RewardNetConfig:
    use_state: bool = True
    use_action: bool = True
    use_next_state: bool = False
    use_done: bool = False
    use_spectral_norm: bool = False
    spectral_norm_kwargs: Optional[Dict] = None
    net_arch: List = (32, 32)

    def to_dict(self,):
        return_dict = asdict(self)
        return return_dict


@dataclass
class DecisionTransformerConfig:
    enable: bool = False
    path_to_model: Optional[str] = None
    path_to_exp_config: Optional[str] = None
    number_of_states_in_aug_policy: int = 1 # how far forward to look. One is only the current obs
    number_of_actions_in_aug_policy: int = 1 # how far forward to look. One is the current action sent
    add_DT_action_to_aug_policy: bool = True
    aug_action_range: float = .2
    use_mean_DT_action: bool = True

@dataclass
class ResidualBCPolicyConfig:
    enable: bool = False
    path_to_model: Optional[str] = None
    path_to_exp_config: Optional[str] = None
    number_of_states_in_aug_policy: int = 1  # how far forward to look. One is only the current obs
    number_of_actions_in_aug_policy: int = 1  # how far forward to look. One is the current action sent
    add_BC_action_to_aug_policy: bool = True
    aug_action_range: float = .2
    use_mean_BC_action: bool = True


def configure_logger(log_folder, exp_name, verbose=2,) -> Tuple[imit_logger.HierarchicalLogger, str]:
    if log_folder is None:
        folder = None
    else:
        folder = os.path.join(log_folder, datetime.datetime.now().strftime(exp_name + "SB3-%Y-%m-%d-%H-%M-%S-%f"))



    if verbose >= 1:
        format_strs = ["stdout", "tensorboard"]
    else:
        format_strs = ["tensorboard"]
    return imit_logger.configure(folder=folder, format_strs = format_strs), folder

@dataclass
class ImitationExpConfig:
    exp_name: str = 'BeT-AIL'
    gen_learner_config: LearnerConfig = SAC_Config()
    reward_net_kwargs: Optional[RewardNetConfig] = RewardNetConfig()
    decision_transformer_config: DecisionTransformerConfig = DecisionTransformerConfig()
    residual_BC_config: ResidualBCPolicyConfig = ResidualBCPolicyConfig()
    logging_directory: Optional[Path] = 'logs'
    ###########################################  Adversarial HParams    ##############################################
    n_eval_episodes: int = 5
    training_steps: int = 10000
    save_every_n_steps: int = 1000
    debug_use_ground_truth: bool = False
    demo_batch_size: int = 1024
    gen_replay_buffer_capacity: Optional[int] = None
    n_disc_updates_per_round: int = 4
    gen_train_timesteps: Optional[int] = None
    disc_learning_starts: int = 0
    recompute_disc_reward: bool = False
    disc_opt_kwargs: Optional[Dict] = None
    disc_augment_reward: Optional = None


    def __post_init__(self):
        self.logger, self.logging_directory = configure_logger(self.logging_directory,
                                                               self.exp_name, verbose=0,
)

