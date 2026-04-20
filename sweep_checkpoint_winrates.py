import argparse
import json
import os
from pathlib import Path

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import BaseEnv

import soccer_twos
from soccer_twos import evaluate as soccer_evaluate


ALGORITHM = "PPO"
DEFAULT_OUTPUT = "checkpoint_winrates.jsonl"
DEFAULT_EPISODES = 10
DEFAULT_BASE_PORT = 55000

RAY_RESULTS_DIR = Path("ray_results")
BASELINE_RUN_NAME = "PPO_baseline_team_vs_random"
OPPONENT_MODULES = {
    "random": "example_player_agent",
    "ceia": "ceia_baseline_agent",
}


class DummyEnv(BaseEnv):
    """
    Minimal env used only to provide the restored PPO policy with the team-level
    observation and action spaces it was trained on.
    """

    def __init__(self, env_config):
        self.observation_space = env_config["observation_space"]
        self.action_space = env_config["action_space"]


class CheckpointTeamPPOAgent:
    """
    Generic evaluator-side PPO agent that restores any team_vs_policy checkpoint
    and exposes the same two-player team interface expected by soccer_twos.
    """

    def __init__(self, env, checkpoint_path: Path):
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )

        self.checkpoint_path = str(checkpoint_path)
        self.player_action_space = env.action_space
        self.player_action_dim = len(env.action_space.nvec)
        self.team_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            dtype=np.float32,
            shape=(env.observation_space.shape[0] * 2,),
        )
        self.team_action_space = gym.spaces.MultiDiscrete(
            np.repeat(env.action_space.nvec, 2)
        )

        config = self._load_config(checkpoint_path)
        config.pop("callbacks", None)
        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["evaluation_interval"] = None
        config["evaluation_num_workers"] = 0
        config["evaluation_num_episodes"] = 0
        config["evaluation_config"] = {}
        config["env"] = "CheckpointSweepDummyEnv"
        config["env_config"] = {
            "observation_space": self.team_observation_space,
            "action_space": self.team_action_space,
        }

        tune.registry.register_env(
            "CheckpointSweepDummyEnv",
            lambda env_config: DummyEnv(env_config),
        )

        trainer_cls = get_trainable_cls(ALGORITHM)
        self.agent = trainer_cls(env=config["env"], config=config)
        self.agent.restore(self.checkpoint_path)
        self.policy = self.agent.get_policy()

        if self.policy is None:
            raise RuntimeError(
                f"Failed to restore a default policy from {self.checkpoint_path}."
            )

    @staticmethod
    def _load_config(checkpoint_path: Path):
        checkpoint_dir = checkpoint_path.parent
        config_candidates = [
            checkpoint_dir / "params.pkl",
            checkpoint_dir.parent / "params.pkl",
            checkpoint_dir.parent.parent / "params.pkl",
            checkpoint_dir / "params.json",
            checkpoint_dir.parent / "params.json",
            checkpoint_dir.parent.parent / "params.json",
        ]

        for config_path in config_candidates:
            if not config_path.exists():
                continue

            if config_path.suffix == ".pkl":
                import pickle

                with open(config_path, "rb") as config_file:
                    return pickle.load(config_file)

            with open(config_path) as config_file:
                return json.load(config_file)

        raise ValueError(f"Could not find params.pkl or params.json for {checkpoint_path}")

    def act(self, observation):
        ordered_player_ids = sorted(observation.keys())
        team_observation = np.concatenate(
            [observation[player_id] for player_id in ordered_player_ids]
        ).astype(np.float32)
        team_action, *_ = self.policy.compute_single_action(
            team_observation,
            explore=False,
        )
        team_action = np.asarray(team_action, dtype=self.player_action_space.dtype)

        return {
            ordered_player_ids[0]: team_action[: self.player_action_dim],
            ordered_player_ids[1]: team_action[self.player_action_dim :],
        }

    def close(self):
        self.agent.stop()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep PPO checkpoints against a baseline agent and log side-specific win rates."
    )
    parser.add_argument(
        "--opponent",
        default="random",
        choices=sorted(OPPONENT_MODULES.keys()),
        help="Baseline opponent to evaluate against.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSONL output path. One result object is written per checkpoint.",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=DEFAULT_BASE_PORT,
        help="Base communication port used for Unity env creation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of checkpoints to evaluate.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run-name filter, e.g. PPO_reward_exp_prog005_clear0.",
    )
    return parser.parse_args()


def iter_checkpoints(run_name_filter=None):
    checkpoint_paths = sorted(
        path
        for path in RAY_RESULTS_DIR.glob("**/checkpoint-*")
        if path.is_file() and not path.name.endswith(".tune_metadata")
    )

    for checkpoint_path in checkpoint_paths:
        run_name = checkpoint_path.parents[2].name
        if run_name_filter and run_name != run_name_filter:
            continue
        yield checkpoint_path


def build_checkpoint_record(checkpoint_path: Path):
    run_name = checkpoint_path.parents[2].name
    trial_dir = checkpoint_path.parents[1]
    checkpoint_dir = checkpoint_path.parent
    checkpoint_iteration = int(checkpoint_path.name.split("-")[-1])

    return {
        "run_name": run_name,
        "trial_dir": str(trial_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iteration": checkpoint_iteration,
        "agent_kind": "ppo_baseline" if run_name == BASELINE_RUN_NAME else "reward_shaping",
    }


def summarize_policy_winrates(summary, policy_name):
    policy_summary = summary["policies"][policy_name]
    return {
        "blue": policy_summary["blue_team"]["policy_blue_team_win_rate"],
        "orange": policy_summary["orange_team"]["policy_orange_team_win_rate"],
        "overall": policy_summary["policy_win_rate"],
        "wins": policy_summary["policy_wins"],
        "losses": policy_summary["policy_losses"],
        "draws": policy_summary["policy_draws"],
        "games": policy_summary["policy_total_games"],
    }


def load_team_agent_for_checkpoint(checkpoint_path: Path, base_port: int):
    env = soccer_twos.make(base_port=base_port)
    agent = CheckpointTeamPPOAgent(env, checkpoint_path)
    env.close()
    return agent


def close_agent(agent):
    trainer = getattr(agent, "agent", None)
    if trainer is not None and hasattr(trainer, "stop"):
        trainer.stop()


def evaluate_checkpoint(checkpoint_path: Path, opponent_agent, opponent_module: str, episodes: int, base_port: int):
    checkpoint_agent_name = str(checkpoint_path)
    checkpoint_agent = load_team_agent_for_checkpoint(checkpoint_path, base_port=base_port + 2)
    env = soccer_twos.make(base_port=base_port + 3)

    try:
        episodes_data = soccer_evaluate.collect_episodes(
            env,
            checkpoint_agent,
            opponent_agent,
            episodes,
        )
        summary = soccer_evaluate.summarize_episodes(
            episodes_data,
            checkpoint_agent_name,
            opponent_module,
        )
    finally:
        env.close()
        checkpoint_agent.close()

    return {
        checkpoint_agent_name: summarize_policy_winrates(summary, checkpoint_agent_name),
        opponent_module: summarize_policy_winrates(summary, opponent_module),
    }


def main():
    args = parse_args()
    opponent_module = OPPONENT_MODULES[args.opponent]
    checkpoint_paths = list(iter_checkpoints(args.run_name))

    if args.limit is not None:
        checkpoint_paths = checkpoint_paths[: args.limit]

    if not checkpoint_paths:
        raise ValueError("No checkpoints matched the requested filters.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating {len(checkpoint_paths)} checkpoints against {opponent_module} "
        f"for {args.episodes} episodes each."
    )

    opponent_agent = soccer_evaluate.load_agent(opponent_module, base_port=args.base_port + 1)

    try:
        with open(output_path, "a") as output_file:
            for index, checkpoint_path in enumerate(checkpoint_paths):
                port_seed = args.base_port + index * 10
                record = build_checkpoint_record(checkpoint_path)
                record["opponent"] = opponent_module
                record["episodes"] = args.episodes
                record["winrates"] = evaluate_checkpoint(
                    checkpoint_path,
                    opponent_agent,
                    opponent_module,
                    args.episodes,
                    port_seed,
                )

                line = json.dumps(record, sort_keys=True)
                print(line)
                output_file.write(line + "\n")
                output_file.flush()
    finally:
        close_agent(opponent_agent)

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
