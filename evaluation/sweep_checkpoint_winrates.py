import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
DEFAULT_EPISODES = 20
DEFAULT_BASE_PORT = 65000
DEFAULT_CHECKPOINT_INTERVAL = 100

RAY_RESULTS_DIR = REPO_ROOT / "ray_results"
OPPONENT_MODULES = {
    "random": "example_player_agent",
    "ceia": "ceia_baseline_agent",
}


def get_agent_kind(run_name):
    if run_name.startswith("PPO_vs_random"):
        # PPO with reward shaping / baseline alg against random team agent
        return "ppo_vs_random"
    elif run_name.startswith("PPO_selfplay"):
        # PPO self-play trained team agent
        return "selfplay"
    elif run_name.startswith("PPO_historical_selfplay"):
        return "historical_selfplay"
    elif run_name.startswith("PPO_seeded_historical_selfplay"):
        # PPO trained against sampled frozen historical self-play opponents
        return "seededrun_selfplay"
    raise ValueError(f"Unrecognized run name: {run_name}")


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
        self.policy = self._get_restored_policy()

        if self.policy is None:
            raise RuntimeError(
                f"Failed to restore an inference policy from {self.checkpoint_path}."
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

    def _get_restored_policy(self):
        """
        Team-vs-random checkpoints are single-agent RLlib checkpoints and use
        RLlib's default policy id. Self-play checkpoints are multiagent and use
        the explicit policy id `default`.
        """

        policy = self.agent.get_policy("current_team")
        if policy is not None:
            return policy

        policy = self.agent.get_policy("default")
        if policy is not None:
            return policy

        policy = self.agent.get_policy()
        if policy is not None:
            return policy

        policy_map = self.agent.workers.local_worker().policy_map
        if len(policy_map) == 1:
            return next(iter(policy_map.values()))

        return None


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
    parser.add_argument(
        "--run-kind",
        default=None,
        choices=(
            "ppo_vs_random",
            "selfplay",
            "historical_selfplay",
            "seededrun_selfplay",
        ),
        help="Optional run-kind filter.",
    )
    parser.add_argument(
        "--trial-dir",
        default=None,
        help=(
            "Optional exact trial directory filter. This is useful when multiple "
            "trials share the same run name."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate checkpoints even if matching records already exist in the output JSONL.",
    )
    return parser.parse_args()


def iter_checkpoints(run_name_filter=None, run_kind_filter=None, trial_dir_filter=None):
    trial_dir_path = Path(trial_dir_filter).resolve() if trial_dir_filter else None
    checkpoint_paths = sorted(
        path
        for path in RAY_RESULTS_DIR.glob("**/checkpoint-*")
        if path.is_file() and not path.name.endswith(".tune_metadata")
    )

    for checkpoint_path in checkpoint_paths:
        checkpoint_iteration = int(checkpoint_path.name.split("-")[-1])
        if checkpoint_iteration % DEFAULT_CHECKPOINT_INTERVAL != 0:
            continue
        run_name = checkpoint_path.parents[2].name
        if run_name_filter and run_name != run_name_filter:
            continue
        if run_kind_filter and get_agent_kind(run_name) != run_kind_filter:
            continue
        if trial_dir_path and checkpoint_path.parents[1].resolve() != trial_dir_path:
            continue
        yield checkpoint_path


def checkpoint_path_for_record(run_name: str, trial_dir: str, checkpoint_iteration: int) -> Path:
    return (
        RAY_RESULTS_DIR
        / run_name
        / trial_dir
        / f"checkpoint_{checkpoint_iteration:06d}"
        / f"checkpoint-{checkpoint_iteration}"
    )


def build_checkpoint_record(checkpoint_path: Path):
    run_name = checkpoint_path.parents[2].name
    trial_dir = checkpoint_path.parents[1]
    checkpoint_iteration = int(checkpoint_path.name.split("-")[-1])
    agent_kind = get_agent_kind(run_name)

    return {
        "run_name": run_name,
        "trial_dir": trial_dir.name,
        "checkpoint_iteration": checkpoint_iteration,
        "agent_kind": agent_kind,
    }


def sweep_key(run_name, trial_dir, checkpoint_iteration, opponent_module, episodes):
    return (
        run_name,
        trial_dir,
        int(checkpoint_iteration),
        opponent_module,
        int(episodes),
    )


def load_completed_sweeps(output_path: Path):
    completed = set()
    if not output_path.exists():
        return completed

    with open(output_path) as output_file:
        for line_number, line in enumerate(output_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping malformed JSONL record in {output_path} "
                    f"at line {line_number}."
                )
                continue

            run_name = record.get("run_name")
            trial_dir = record.get("trial_dir")
            checkpoint_iteration = record.get("checkpoint_iteration")
            opponent = record.get("opponent")
            episodes = record.get("episodes")
            if (
                run_name is None
                or trial_dir is None
                or checkpoint_iteration is None
                or opponent is None
                or episodes is None
            ):
                continue

            completed.add(
                sweep_key(
                    run_name,
                    trial_dir,
                    int(checkpoint_iteration),
                    opponent,
                    int(episodes),
                )
            )

    return completed


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


def evaluate_checkpoint(
    checkpoint_path: Path,
    agent_kind: str,
    opponent_agent,
    opponent_module: str,
    episodes: int,
    base_port: int,
):
    checkpoint_agent_name = agent_kind
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = list(iter_checkpoints(args.run_name, args.run_kind, args.trial_dir))

    completed_sweeps = set() if args.force else load_completed_sweeps(output_path)
    total_checkpoint_count = len(checkpoint_paths)
    if completed_sweeps:
        filtered_checkpoint_paths = []
        for checkpoint_path in checkpoint_paths:
            record = build_checkpoint_record(checkpoint_path)
            if (
                sweep_key(
                    record["run_name"],
                    record["trial_dir"],
                    record["checkpoint_iteration"],
                    opponent_module,
                    args.episodes,
                )
                not in completed_sweeps
            ):
                filtered_checkpoint_paths.append(checkpoint_path)
        checkpoint_paths = filtered_checkpoint_paths
        skipped_count = total_checkpoint_count - len(checkpoint_paths)
    else:
        skipped_count = 0

    if args.limit is not None:
        checkpoint_paths = checkpoint_paths[: args.limit]

    if not checkpoint_paths:
        print(
            "No remaining checkpoints to evaluate. "
            f"Matched {total_checkpoint_count}; skipped {skipped_count} already-swept records."
        )
        return

    print(
        f"Evaluating {len(checkpoint_paths)} checkpoints against {opponent_module} "
        f"for {args.episodes} episodes each. "
        f"Skipped {skipped_count} already-swept checkpoints."
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
                    record["agent_kind"],
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
