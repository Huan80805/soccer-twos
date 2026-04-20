from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env, init_ray


NUM_ENVS_PER_WORKER = 3
NUM_WORKERS = 8
BASE_PORT = 56000
TRAIN_BATCH_SIZE = 4000
TIMESTEPS_TOTAL = 5000000
LR = 2.5e-5
NUM_SGD_ITER = 10


if __name__ == "__main__":
    init_ray()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({
        "base_port": BASE_PORT,
        "variation": EnvType.multiagent_team,
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_baseline_teams_lr25e6_sgd10",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(lambda _: "default"),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
                "variation": EnvType.multiagent_team,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
            },
            "train_batch_size": TRAIN_BATCH_SIZE,
            # Slower PPO updates: the previous default did 30 passes over each
            # batch, which pushed KL above target and forced a high KL penalty.
            "lr": LR,
            "num_sgd_iter": NUM_SGD_ITER,
            "evaluation_interval": 50,
            "evaluation_num_workers": 1,
            "evaluation_num_episodes": 10,
            "evaluation_config": {
                "explore": False,
                "env_config": {
                    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                    "base_port": BASE_PORT,
                    "variation": EnvType.multiagent_team,
                },
            },
        },
        stop={
            "timesteps_total": TIMESTEPS_TOTAL,  # 5M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_teams_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
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
