from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env, init_ray, RewardShapingMetricsCallbacks


NUM_ENVS_PER_WORKER = 5
NUM_WORKERS = 8
BASE_PORT = 55000
TRAIN_BATCH_SIZE = 4000
BALL_PROGRESS_WEIGHT = 0.05
DEFENSIVE_CLEAR_WEIGHT = 0.0
DEFENSIVE_HALF_THRESHOLD = -4.0

if __name__ == "__main__":
    init_ray()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_reward_exp_prog005_clear0",
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
                "reward_shaping": "custom",
                "ball_progress_weight": BALL_PROGRESS_WEIGHT,
                "defensive_clear_weight": DEFENSIVE_CLEAR_WEIGHT,
                "defensive_half_threshold": DEFENSIVE_HALF_THRESHOLD,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
            },
            "train_batch_size": TRAIN_BATCH_SIZE,
            "callbacks": RewardShapingMetricsCallbacks,
            "evaluation_interval": 50,
            "evaluation_num_workers": 1,
            "evaluation_num_episodes": 10,
            "evaluation_config": {
                "explore": False,
                "env_config": {
                    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                    "base_port": BASE_PORT,
                    "variation": EnvType.team_vs_policy,
                    "multiagent": False,
                },
            },
        },
        stop={
            "timesteps_total": 5000000,  # 5M
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
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
