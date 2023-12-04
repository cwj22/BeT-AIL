from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DiscAugmentVarianceConfig:
    augmented_lambda: float = 0.1



@dataclass
class DiscAugmentGradientPenaltyConfig:
    grad_pen_scale: float = 10.
    grad_pen_targ: float = 1.
    grad_pen_type: str = 'wgan'
    one_sided_pen: bool = True

@dataclass
class DiscAugmentEntropyConfig:
    ent_reg_scale: float = 0.001


@dataclass
class DiscAugmentVariationalKLConstraintConfig:
    information_flow: float = 0.5
    beta: float = 0.1
    dual_descent_on_beta: bool = False
    beta_step_size: float = 1e-6
    update_beta_every_nstep: int = 1


@dataclass
class DiscAugmentListConfig:
    config_list: List = None



DISC_AUGMENT_CONFIGS = {'DiscAugmentVarianceConfig': DiscAugmentVarianceConfig,
                        'DiscAugmentGradientPenaltyConfig': DiscAugmentGradientPenaltyConfig,
                        'DiscAugmentEntropyConfig': DiscAugmentEntropyConfig,
                        'DiscAugmentListConfig': DiscAugmentListConfig,
                        'DiscAugmentVariationalKLConstraintConfig': DiscAugmentVariationalKLConstraintConfig}