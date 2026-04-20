"""Unified reward-shaping PPO experiments.

Legacy team-vs-random shaped runs:
    python exp_reward_shaping.py --mode team_vs_random --ball-progress-weight 0.05 --defensive-clear-weight 0.0 --port 55000 --exp-name PPO_reward_exp_prog005_clear0
    python exp_reward_shaping.py --mode team_vs_random --ball-progress-weight 0.05 --defensive-clear-weight 0.1 --port 55000 --exp-name PPO_reward_exp_prog005_clear01
    python exp_reward_shaping.py --mode team_vs_random --ball-progress-weight 0.1 --defensive-clear-weight 0.0 --port 55000 --exp-name PPO_reward_exp_prog01_clear00
    python exp_reward_shaping.py --mode team_vs_random --ball-progress-weight 0.1 --defensive-clear-weight 0.1 --port 55000 --exp-name PPO_reward_exp_prog01_clear01

Legacy self-play shaped runs:
    python exp_reward_shaping.py --mode selfplay --ball-progress-weight 0.05 --defensive-clear-weight 0.0 --port 57000 --exp-name PPO_selfplay_reward_prog005_clear0
    python exp_reward_shaping.py --mode selfplay --ball-progress-weight 0.05 --defensive-clear-weight 0.1 --port 50000 --exp-name PPO_selfplay_reward_prog005_clear01

Tuned self-play shaped run:
    python exp_reward_shaping.py --mode selfplay --ball-progress-weight 0.05 --defensive-clear-weight 0.1 --lr 2.5e-5 --num-sgd-iter 10 --sgd-batch-size 128 --port 50000 --exp-name PPO_selfplay_reward_prog005_clear01_lr25e6_sgd10

If --exp-name is omitted, the run name is generated from mode, reward weights,
lr, num_sgd_iter, and sgd_batch_size.
"""

import argparse

from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env, init_ray, RewardShapingMetricsCallbacks


DEFAULT_NUM_WORKERS = 8
DEFAULT_TRAIN_BATCH_SIZE = 4000
DEFAULT_TIMESTEPS_TOTAL = 5000000
DEFAULT_LR = 5e-5
DEFAULT_NUM_SGD_ITER = 30
DEFAULT_SGD_BATCH_SIZE = 128
DEFAULT_DEFENSIVE_HALF_THRESHOLD = -4.0


def parse_args(default_mode="team_vs_random"):
    parser = argparse.ArgumentParser(
        description="Train PPO with custom Soccer-Twos reward shaping."
    )
    parser.add_argument(
        "--mode",
        choices=["team_vs_random", "team", "selfplay", "multiagent_team"],
        default=default_mode,
        help="Training setup: team_vs_random for TeamVsPolicyWrapper, selfplay for multiagent teams.",
    )
    parser.add_argument(
        "--ball-progress-weight", "-WBP", type=float, default=0.05,
        help="Reward added for moving the ball toward the opponent goal.",
    )
    parser.add_argument(
        "--defensive-clear-weight", "-WD", type=float, default=0.0,
        help="Reward added for moving the ball out of the defensive half.",
    )
    parser.add_argument( 
        "--defensive-half-threshold", "-TD", type=float, default=DEFAULT_DEFENSIVE_HALF_THRESHOLD,
        help="Ball x-position threshold used by the defensive-clear shaping term.",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR, help="PPO learning rate.",
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Base Unity worker port."
    )
    parser.add_argument(
        "--num-sgd-iter", type=int, default=DEFAULT_NUM_SGD_ITER, help="Number of SGD passes over each PPO train batch.",
    )
    parser.add_argument(
        "--sgd-batch-size", type=int, default=DEFAULT_SGD_BATCH_SIZE, help="RLlib PPO sgd_minibatch_size.",
    )

    parser.add_argument(
        "--exp-name", default=None, help="Tune experiment name. If omitted, generated from the hparam combination.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="RLlib rollout workers.",
    )
    parser.add_argument(
        "--num-envs-per-worker", type=int, default=5, help="Unity envs per worker",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help="PPO train_batch_size.",
    )
    parser.add_argument(
        "--timesteps-total", type=int, default=DEFAULT_TIMESTEPS_TOTAL, help="Total environment timesteps before stopping.",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=100, help="Tune checkpoint frequency in training iterations.",
    )
    parser.add_argument(
        "--local-dir", default="./ray_results", help="Tune output directory.",
    )
    return parser.parse_args()


def normalized_mode(mode):
    if mode in ("team", "team_vs_random"):
        return "team_vs_random"
    if mode in ("selfplay", "multiagent_team"):
        return "selfplay"
    raise ValueError(f"Unsupported mode: {mode}")

def weight_token(value):
    if value == 0:
        return "0"
    return f"{value:.6g}".replace(".", "").replace("-", "m")


def hparam_token(value):
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def default_exp_name(args, mode):
    prefix = "PPO_reward_exp" if mode == "team_vs_random" else "PPO_selfplay_reward"
    return (
        f"{prefix}_prog{weight_token(args.ball_progress_weight)}"
        f"_clear{weight_token(args.defensive_clear_weight)}"
        f"_lr{hparam_token(args.lr)}"
        f"_sgd{args.num_sgd_iter}"
        f"_mb{args.sgd_batch_size}"
    )


def reward_env_config(args, mode, include_shaping=True):
    variation = (
        EnvType.team_vs_policy
        if mode == "team_vs_random"
        else EnvType.multiagent_team
    )
    env_config = {
        "num_envs_per_worker": args.num_envs_per_worker,
        "base_port": args.port,
        "variation": variation,
    }
    if mode == "team_vs_random":
        env_config["multiagent"] = False
    if include_shaping:
        env_config.update(
            {
                "reward_shaping": "custom",
                "ball_progress_weight": args.ball_progress_weight,
                "defensive_clear_weight": args.defensive_clear_weight,
                "defensive_half_threshold": args.defensive_half_threshold,
            }
        )
    return env_config


def selfplay_multiagent_config(args):
    temp_env = create_rllib_env(reward_env_config(args, "selfplay"))
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()
    return {
        "policies": {
            "default": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": tune.function(lambda _: "default"),
        "policies_to_train": ["default"],
    }


def build_config(args, mode):
    config = {
        # system settings
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.num_envs_per_worker,
        "log_level": "INFO",
        "framework": "torch",
        # RL setup
        "env": "Soccer",
        "env_config": reward_env_config(args, mode),
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": [256, 256],
        },
        "train_batch_size": args.train_batch_size,
        "lr": args.lr,
        "num_sgd_iter": args.num_sgd_iter,
        "sgd_minibatch_size": args.sgd_batch_size,
        "callbacks": RewardShapingMetricsCallbacks,
        "evaluation_interval": 50,
        "evaluation_num_workers": 1,
        "evaluation_num_episodes": 10,
        "evaluation_config": {
            "explore": False,
            # Evaluate the learned policy on the base environment reward. This
            # keeps training reward shaping separate from the win-rate objective.
            "env_config": reward_env_config(args, mode, include_shaping=False),
        },
    }
    if mode == "selfplay":
        config["multiagent"] = selfplay_multiagent_config(args)
    return config


def train(args):
    mode = normalized_mode(args.mode)

    exp_name = args.exp_name or default_exp_name(args, mode)

    init_ray()
    tune.registry.register_env("Soccer", create_rllib_env)

    print(f"Starting reward-shaped PPO run: {exp_name}")
    print(f"mode={mode}")
    print(f"ball_progress_weight={args.ball_progress_weight}")
    print(f"defensive_clear_weight={args.defensive_clear_weight}")
    print(f"lr={args.lr}")
    print(f"num_sgd_iter={args.num_sgd_iter}")
    print(f"sgd_minibatch_size={args.sgd_batch_size}")
    print(f"base_port={args.port}")

    analysis = tune.run(
        "PPO",
        name=exp_name,
        config=build_config(args, mode),
        stop={
            "timesteps_total": args.timesteps_total,
        },
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
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


def main(default_mode="team_vs_random"):
    train(parse_args(default_mode=default_mode))


if __name__ == "__main__":
    main()
