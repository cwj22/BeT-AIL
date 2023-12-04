from dataclasses import dataclass

@dataclass
class BeTConfig:
    env: str = 'MountainCarContinuous-v0'
    seed: int = 0
    K: int = 20
    embed_dim: int = 512
    n_layer: int = 4
    n_head: int = 4
    activation_function: str = 'relu'
    dropout: float = 0.1
    eval_context_length: int = 5
    ordering: int = 0
    eval_rtg: int = 3000
    num_eval_episodes: int = 20
    init_temperature: float = 0.1
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    warmup_steps: int = 10000
    max_pretrain_iters: int = 100
    num_updates_per_pretrain_iter: int = 5000
    max_online_iters: int = 0
    online_rtg: int = 7200
    num_online_rollouts: int = 1
    replay_size: int = 1000
    eval_interval: int = 1
    device: str = 'cuda'
    log_to_tb: bool = True
    save_dir: str = './bet'
    exp_name: str = 'default'
    return_embedding: bool = True