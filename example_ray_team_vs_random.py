from ray import tune
from soccer_twos import EnvType
import os

from utils import create_rllib_env, init_ray


NUM_ENVS_PER_WORKER = 5
NUM_WORKERS = 8
BASE_PORT = 55000
TRAIN_BATCH_SIZE = 4000

if __name__ == "__main__":
    init_ray()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name=(
            "PPO_baseline_team_vs_random"
        ),
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
            },
            "train_batch_size": TRAIN_BATCH_SIZE,
        },
        stop={
            "timesteps_total": 2000000,  # 2M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    if best_trial is None and analysis.trials:
        best_trial = analysis.trials[0]

    print(best_trial)

    best_checkpoint = None
    if best_trial is not None:
        try:
            best_checkpoint = analysis.get_best_checkpoint(
                trial=best_trial, metric="episode_reward_mean", mode="max"
            )
        except ValueError:
            best_checkpoint = None

    print(best_checkpoint)
    print("Done training")
