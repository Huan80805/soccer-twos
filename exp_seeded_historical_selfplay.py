"""Seeded historical-opponent self-play PPO experiments.

This variant bootstraps historical self-play with high-performing checkpoints
selected from prior runs. The seed list is driven by CEIA sweep results because
the TA confirmed CEIA-based model selection is allowed, but CEIA is not used as
a direct training opponent.

Typical run:
    python exp_seeded_historical_selfplay.py --port 59000 --yes

To add new seed-generation runs, edit SEED_RUN_NAMES below, rerun the CEIA
checkpoint sweep for those runs, then start this script.
"""

import argparse
import copy
import json
import pickle
import random
from pathlib import Path

import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
from soccer_twos import EnvType

from utils import create_rllib_env, init_ray, RewardShapingMetricsCallbacks


CURRENT_POLICY = "current_team"
OPPONENT_POLICY = "opponent_team"
ALGORITHM = "PPO"

DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_NUM_WORKERS = 8
DEFAULT_BASE_PORT = 59000
DEFAULT_TRAIN_BATCH_SIZE = 4000
DEFAULT_TIMESTEPS_TOTAL = 10000000
DEFAULT_LR = 2e-5
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_SGD_BATCH_SIZE = 128
DEFAULT_HISTORY_INTERVAL = 10
DEFAULT_HISTORY_MAX_SIZE = 0

DEFAULT_SEED_WEIGHTED_PROB = 0.4
DEFAULT_HISTORY_PROB = 0.3
DEFAULT_CURRENT_PROB = 0.2
DEFAULT_SEED_UNIFORM_PROB = 0.1
DEFAULT_SEED_WEIGHT_ALPHA = 2.0

DEFAULT_MODEL_HIDDENS = [256, 256]
DEFAULT_VF_SHARE_LAYERS = True

REPO_ROOT = Path(__file__).resolve().parent

# ROUND 3 archive seeds
# SEED_RUN_NAMES = [
#     "PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA",
#     "PPO_seeded_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_runC",
#     "PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr2p5em05_sgd10_runD",
#     "PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr1em04_sgd6_runE",
#     "PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runF" 
# ]

# ROUND 4 archive seeds
SEED_RUN_NAMES = [
    "PPO_seeded_historical_selfplay_reward_goal200_retreat250_scale6_lr5em05_sgd6_runH",
    "PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG",
    "PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA"
]

SEEDED_SELFPLAY_SETTINGS = {}


class DummyPolicyEnv(BaseEnv):
    """Minimal env used only while restoring seed checkpoints."""

    def __init__(self, env_config):
        self.observation_space = env_config["observation_space"]
        self.action_space = env_config["action_space"]


class TeamMatchMaker:
    """Randomly assign the trainable policy to blue or orange each episode."""

    def __init__(self, debug=False):
        self.debug = debug
        self._fallback_current_side = random.choice([0, 1])
        self._fallback_seen_agent_ids = set()

    def policy_mapping_fn(self, agent_id, episode=None, **kwargs):
        if episode is not None:
            current_side = episode.user_data.get("current_team_side")
            if current_side is None:
                current_side = random.choice([0, 1])
                episode.user_data["current_team_side"] = current_side
            return CURRENT_POLICY if int(agent_id) == current_side else OPPONENT_POLICY

        policy_id = (
            CURRENT_POLICY
            if int(agent_id) == self._fallback_current_side
            else OPPONENT_POLICY
        )
        self._fallback_seen_agent_ids.add(int(agent_id))
        if len(self._fallback_seen_agent_ids) >= 2:
            self._fallback_seen_agent_ids.clear()
            self._fallback_current_side = random.choice([0, 1])
        return policy_id


class SeededHistoricalSelfPlayCallbacks(RewardShapingMetricsCallbacks):
    """Sample frozen opponents from seeds, current-run history, or current policy."""

    def __init__(self):
        super().__init__()
        settings = SEEDED_SELFPLAY_SETTINGS
        self.save_interval = settings.get("history_save_interval", DEFAULT_HISTORY_INTERVAL)
        self.sample_interval = settings.get(
            "opponent_update_interval",
            DEFAULT_HISTORY_INTERVAL,
        )
        self.max_history_size = settings.get("history_max_size", DEFAULT_HISTORY_MAX_SIZE)
        self.seed_pool = list(settings.get("seed_pool", []))
        self.seed_weighted_prob = settings.get(
            "seed_weighted_prob",
            DEFAULT_SEED_WEIGHTED_PROB,
        )
        self.history_prob = settings.get("history_prob", DEFAULT_HISTORY_PROB)
        self.current_prob = settings.get("current_prob", DEFAULT_CURRENT_PROB)
        self.seed_uniform_prob = settings.get(
            "seed_uniform_prob",
            DEFAULT_SEED_UNIFORM_PROB,
        )
        self.seed_weight_alpha = settings.get(
            "seed_weight_alpha",
            DEFAULT_SEED_WEIGHT_ALPHA,
        )
        self.debug = settings.get("debug", False)
        self.init_weights = settings.get("init_weights", None)

        self.policy_history = []
        self.counter = 0
        self.saved_policy_count = 0
        self.evicted_policy_count = 0
        self.current_opponent_source = "initial"
        self.current_opponent_source_type = "initial"
        self.current_seed_score = 0.0
        self._init_weights_applied = False

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        super().on_episode_start(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
        current_side = episode.user_data.get("current_team_side")
        if current_side is None:
            current_side = random.choice([0, 1])
            episode.user_data["current_team_side"] = current_side
        episode.user_data["seeded_current_side"] = current_side

        # Ray 1.4 calls policy_mapping_fn(agent_id) without episode context, so
        # pre-fill the mapping here to keep vectorized envs from interleaving
        # fallback matchmaker calls.
        episode._agent_to_policy[0] = (
            CURRENT_POLICY if current_side == 0 else OPPONENT_POLICY
        )
        episode._agent_to_policy[1] = (
            CURRENT_POLICY if current_side == 1 else OPPONENT_POLICY
        )

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs,
        )
        current_side = episode.user_data.get("seeded_current_side")
        if current_side is not None:
            episode.custom_metrics["seeded_current_team_blue"] = (
                1.0 if current_side == 0 else 0.0
            )
            episode.custom_metrics["seeded_current_team_orange"] = (
                1.0 if current_side == 1 else 0.0
            )

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        if not self._init_weights_applied:
            self._init_weights_applied = True
            if self.init_weights is not None:
                # TODO: This runs after the first Tune training iteration because
                # Ray 1.4 callbacks do not expose an on-trainer-created hook.
                # For long runs this is acceptable, but a cleaner warm start
                # would apply these weights before the first rollout.
                trainer.set_weights(
                    {
                        CURRENT_POLICY: self.init_weights,
                        OPPONENT_POLICY: self.init_weights,
                    }
                )
                trainer.workers.sync_weights()
                print("[seeded_selfplay] applied --init-checkpoint weights to current_team and opponent_team")

        self.counter += 1
        current_weights = trainer.get_weights([CURRENT_POLICY])[CURRENT_POLICY]

        if self.counter % self.save_interval == 0:
            history_entry = self._save_current_policy(trainer, result, current_weights)
            self.policy_history.append(history_entry)
            self.saved_policy_count += 1
            if self.max_history_size > 0 and len(self.policy_history) > self.max_history_size:
                evicted_policy = self.policy_history.pop(0)
                self.evicted_policy_count += 1
                print(f"[seeded_selfplay] evicted={evicted_policy}")

        if self.counter % self.sample_interval == 0:
            opponent_weights, source, source_type, seed_score = self._sample_opponent_weights(
                current_weights
            )
            trainer.set_weights(
                {
                    CURRENT_POLICY: current_weights,
                    OPPONENT_POLICY: opponent_weights,
                }
            )
            trainer.workers.sync_weights()
            self.current_opponent_source = source
            self.current_opponent_source_type = source_type
            self.current_seed_score = seed_score
            print(f"[seeded_selfplay] opponent={source_type}:{source}")

        custom_metrics = result.setdefault("custom_metrics", {})
        custom_metrics["seeded_selfplay_seed_pool_size"] = len(self.seed_pool)
        custom_metrics["seeded_selfplay_history_size"] = len(self.policy_history)
        custom_metrics["seeded_selfplay_counter"] = self.counter
        custom_metrics["seeded_selfplay_saved_policy_count"] = self.saved_policy_count
        custom_metrics["seeded_selfplay_evicted_policy_count"] = self.evicted_policy_count
        custom_metrics["seeded_selfplay_selected_seed_score"] = self.current_seed_score
        for source_type in ("current", "history", "seed_weighted", "seed_uniform"):
            custom_metrics[f"seeded_selfplay_opponent_is_{source_type}"] = (
                1.0 if self.current_opponent_source_type == source_type else 0.0
            )

    def _save_current_policy(self, trainer, result, current_weights):
        checkpoint_dir = Path(trainer.logdir) / "historical_opponent_weights"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        iteration = getattr(trainer, "iteration", self.counter)
        weights_path = checkpoint_dir / f"current_team_iter_{iteration:06d}.pkl"

        with open(weights_path, "wb") as weights_file:
            pickle.dump(current_weights, weights_file)

        policy_rewards = result.get("policy_reward_mean", {})
        history_entry = {
            "iteration": iteration,
            "weights_path": str(weights_path),
            "current_team_reward_mean": policy_rewards.get(CURRENT_POLICY),
        }
        print(f"[seeded_selfplay] saved={history_entry}")
        return history_entry

    def _sample_opponent_weights(self, current_weights):
        available_sources = [("current", self.current_prob)]
        if self.policy_history:
            available_sources.append(("history", self.history_prob))
        if self.seed_pool:
            available_sources.append(("seed_weighted", self.seed_weighted_prob))
            available_sources.append(("seed_uniform", self.seed_uniform_prob))

        names = [name for name, _ in available_sources]
        source_probs = [max(0.0, source_prob) for _, source_prob in available_sources]
        if sum(source_probs) <= 0.0:
            return current_weights, "current_team", "current", 0.0

        source_type = random.choices(names, weights=source_probs, k=1)[0]
        if source_type == "current":
            return current_weights, "current_team", "current", 0.0
        if source_type == "history":
            entry = self._sample_history_entry()
            weights = self._load_weights(entry)
            return weights, f"history_iter_{entry['iteration']}", "history", 0.0
        if source_type == "seed_uniform":
            entry = random.choice(self.seed_pool)
            weights = self._load_weights(entry)
            return weights, self._seed_source(entry), "seed_uniform", entry["ceia_overall"]

        entry = self._sample_seed_entry_weighted()
        weights = self._load_weights(entry)
        return weights, self._seed_source(entry), "seed_weighted", entry["ceia_overall"]

    def _sample_history_entry(self):
        if len(self.policy_history) == 1:
            return self.policy_history[0]
        max_index = len(self.policy_history) - 1
        sampled_index = int(random.triangular(0, max_index, max_index))
        sampled_index = max(0, min(max_index, sampled_index))
        return self.policy_history[sampled_index]

    def _sample_seed_entry_weighted(self):
        weights = [
            (0.05 + max(0.0, entry["selection_score"])) ** self.seed_weight_alpha
            for entry in self.seed_pool
        ]
        return random.choices(self.seed_pool, weights=weights, k=1)[0]

    def _load_weights(self, entry):
        with open(entry["weights_path"], "rb") as weights_file:
            return pickle.load(weights_file)

    def _seed_source(self, entry):
        return (
            f"{entry['run_name']}:"
            f"checkpoint_{entry['checkpoint_iteration']}:"
            f"ceia_{entry['ceia_overall']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with seeded historical-opponent self-play."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        default=DEFAULT_NUM_ENVS_PER_WORKER,
    )
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--timesteps-total", type=int, default=DEFAULT_TIMESTEPS_TOTAL)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num-sgd-iter", type=int, default=DEFAULT_NUM_SGD_ITER)
    parser.add_argument("--sgd-batch-size", type=int, default=DEFAULT_SGD_BATCH_SIZE)
    parser.add_argument(
        "--history-save-interval",
        type=int,
        default=DEFAULT_HISTORY_INTERVAL,
        help="Save current_team into current-run history every N iterations.",
    )
    parser.add_argument(
        "--opponent-update-interval",
        type=int,
        default=DEFAULT_HISTORY_INTERVAL,
        help="Sample opponent_team every N training iterations.",
    )
    parser.add_argument(
        "--history-max-size",
        type=int,
        default=DEFAULT_HISTORY_MAX_SIZE,
        help="Maximum current-run history size. Use 0 for unbounded history.",
    )
    parser.add_argument(
        "--reward-shaping",
        choices=["none", "custom"],
        default="custom",
        help="Enable symmetric multiagent-team reward shaping.",
    )
    parser.add_argument(
        "--goal-progress-weight",
        "-WGP",
        type=float,
        default=0.75,
        help="Reward weight for increasing exponential goal-proximity potential.",
    )
    parser.add_argument(
        "--retreat-penalty-weight",
        "-WR",
        type=float,
        default=1.25,
        help="Penalty weight for decreases in exponential goal-proximity potential.",
    )
    parser.add_argument(
        "--goal-potential-scale",
        "-WGS",
        type=float,
        default=6.0,
        help="Distance scale for exp(-distance_to_goal / scale) reward shaping.",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--local-dir", default="./ray_results")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--model-hiddens",
        default="256,256",
        help="Comma-separated hidden layer sizes for the current/opponent policies.",
    )
    vf_group = parser.add_mutually_exclusive_group()
    vf_group.add_argument(
        "--vf-share-layers",
        dest="vf_share_layers",
        action="store_true",
        default=DEFAULT_VF_SHARE_LAYERS,
    )
    vf_group.add_argument(
        "--no-vf-share-layers",
        dest="vf_share_layers",
        action="store_false",
    )
    parser.add_argument(
        "--ceia-eval-path",
        default="evaluation/ceia_eval.jsonl",
        help="JSONL checkpoint sweep file used for seed ranking.",
    )
    parser.add_argument(
        "--seed-output-dir",
        default="ray_results/seeded_opponent_weights",
        help="Directory for extracted seed policy-weight pkl files.",
    )
    parser.add_argument("--max-seeds-total", type=int, default=12)
    parser.add_argument("--max-seeds-per-run", type=int, default=3)
    parser.add_argument("--min-seed-ceia", type=float, default=0.0)
    parser.add_argument(
        "--side-balance-penalty",
        type=float,
        default=0.1,
        help="Penalty multiplier for abs(blue_win_rate - orange_win_rate).",
    )
    parser.add_argument("--seed-weight-alpha", type=float, default=DEFAULT_SEED_WEIGHT_ALPHA)
    parser.add_argument("--seed-weighted-prob", type=float, default=DEFAULT_SEED_WEIGHTED_PROB)
    parser.add_argument("--history-prob", type=float, default=DEFAULT_HISTORY_PROB)
    parser.add_argument("--current-prob", type=float, default=DEFAULT_CURRENT_PROB)
    parser.add_argument("--seed-uniform-prob", type=float, default=DEFAULT_SEED_UNIFORM_PROB)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive seed confirmation. Useful for batch jobs.",
    )
    parser.add_argument(
        "--dry-run-seeds",
        action="store_true",
        help="Print selected compatible seeds and exit before Ray/training startup.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra matchmaker, opponent, and reward-shaping diagnostics.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="ray_results/.../checkpoint_004400/checkpoint-4400",
        help=(
            "Checkpoint path to warm-start current_team weights from. "
            "Starts a fresh experiment (new timesteps, fresh optimizer) but "
            "skips random policy initialization after the first Tune iteration. "
            "Example: ray_results/.../checkpoint_004400/checkpoint-4400"
        ),
    )
    parser.add_argument(
        "--restore",
        default=None,
        help=(
            "Checkpoint path for a full Tune restore (resumes optimizer state "
            "and timestep count). Raise --timesteps-total beyond the checkpoint's "
            "original limit. TODO: reload callback-side current-run history from "
            "historical_opponent_weights before using this as a faithful seeded "
            "self-play resume. "
            "Example: ray_results/.../checkpoint_004400/checkpoint-4400"
        ),
    )
    return parser.parse_args()


def weight_token(value):
    if value == 0:
        return "0"
    return f"{value:.6g}".replace(".", "").replace("-", "m")


def hparam_token(value):
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def default_exp_name(args):
    return (
        "PPO_seeded_historical_selfplay_reward"
        f"_goal{weight_token(args.goal_progress_weight)}"
        f"_retreat{weight_token(args.retreat_penalty_weight)}"
        f"_scale{weight_token(args.goal_potential_scale)}"
        f"_lr{hparam_token(args.lr)}"
        f"_sgd{args.num_sgd_iter}"
    )


def parse_model_hiddens(value):
    hiddens = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not hiddens:
        raise ValueError("--model-hiddens must contain at least one hidden layer size.")
    return hiddens


def env_config(args, include_shaping=True):
    config = {
        "num_envs_per_worker": args.num_envs_per_worker,
        "base_port": args.port,
        "variation": EnvType.multiagent_team,
    }
    if include_shaping and args.reward_shaping == "custom":
        config.update(
            {
                "reward_shaping": "custom",
                "goal_progress_weight": args.goal_progress_weight,
                "retreat_penalty_weight": args.retreat_penalty_weight,
                "goal_potential_scale": args.goal_potential_scale,
                "reward_shaping_debug": args.debug,
            }
        )
    return config


def build_multiagent_config(args, obs_space, act_space):
    policies = {
        CURRENT_POLICY: (None, obs_space, act_space, {}),
        OPPONENT_POLICY: (None, obs_space, act_space, {}),
    }
    matchmaker = TeamMatchMaker(debug=args.debug)
    return {
        "policies": policies,
        "policy_mapping_fn": tune.function(matchmaker.policy_mapping_fn),
        "policies_to_train": [CURRENT_POLICY],
    }


def configure_callback_settings(args, seed_pool, init_weights=None):
    SEEDED_SELFPLAY_SETTINGS.clear()
    SEEDED_SELFPLAY_SETTINGS.update(
        {
            "history_save_interval": args.history_save_interval,
            "opponent_update_interval": args.opponent_update_interval,
            "history_max_size": args.history_max_size,
            "seed_pool": seed_pool,
            "seed_weighted_prob": args.seed_weighted_prob,
            "history_prob": args.history_prob,
            "current_prob": args.current_prob,
            "seed_uniform_prob": args.seed_uniform_prob,
            "seed_weight_alpha": args.seed_weight_alpha,
            "debug": args.debug,
            "init_weights": init_weights,
        }
    )


def find_config_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    candidates = [
        checkpoint_dir / "params.pkl",
        checkpoint_dir.parent / "params.pkl",
        checkpoint_dir.parent.parent / "params.pkl",
        checkpoint_dir / "params.json",
        checkpoint_dir.parent / "params.json",
        checkpoint_dir.parent.parent / "params.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def checkpoint_path_for_record(record):
    checkpoint_iteration = int(record["checkpoint_iteration"])
    return (
        REPO_ROOT
        / "ray_results"
        / record["run_name"]
        / record["trial_dir"]
        / f"checkpoint_{checkpoint_iteration:06d}"
        / f"checkpoint-{checkpoint_iteration}"
    )


def load_checkpoint_config(checkpoint_path):
    config_path = find_config_path(checkpoint_path)
    if config_path is None:
        raise ValueError(f"Could not find params.pkl or params.json for {checkpoint_path}")
    if config_path.suffix == ".pkl":
        with open(config_path, "rb") as config_file:
            return pickle.load(config_file)
    with open(config_path) as config_file:
        return json.load(config_file)


def get_checkpoint_winrate(record):
    return record.get("winrates", {}).get(record.get("agent_kind"))


def is_model_compatible(config, target_hiddens, target_vf_share_layers):
    model = config.get("model", {})
    return (
        model.get("fcnet_hiddens") == target_hiddens
        and model.get("vf_share_layers", True) == target_vf_share_layers
    )


def select_seed_candidates(args, target_hiddens):
    eval_path = Path(args.ceia_eval_path)
    if not eval_path.is_absolute():
        eval_path = REPO_ROOT / eval_path
    if not eval_path.exists():
        raise FileNotFoundError(f"CEIA eval file not found: {eval_path}")

    candidates_by_run = {}
    skipped = {
        "run_name": 0,
        "missing_checkpoint": 0,
        "missing_config": 0,
        "incompatible_model": 0,
        "low_score": 0,
        "bad_record": 0,
    }

    with open(eval_path) as eval_file:
        for line_number, line in enumerate(eval_file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            run_name = record.get("run_name")
            if run_name not in SEED_RUN_NAMES:
                skipped["run_name"] += 1
                continue

            checkpoint_path = checkpoint_path_for_record(record)
            if not checkpoint_path.exists():
                skipped["missing_checkpoint"] += 1
                continue

            winrate = get_checkpoint_winrate(record)
            if winrate is None:
                skipped["bad_record"] += 1
                continue
            ceia_overall = float(winrate.get("overall", 0.0))
            if ceia_overall < args.min_seed_ceia:
                skipped["low_score"] += 1
                continue

            try:
                config = load_checkpoint_config(checkpoint_path)
            except ValueError:
                skipped["missing_config"] += 1
                continue

            if not is_model_compatible(config, target_hiddens, args.vf_share_layers):
                skipped["incompatible_model"] += 1
                continue

            blue = float(winrate.get("blue", 0.0))
            orange = float(winrate.get("orange", 0.0))
            side_gap = abs(blue - orange)
            selection_score = ceia_overall - args.side_balance_penalty * side_gap
            candidate = {
                "run_name": run_name,
                "trial_dir": record.get("trial_dir"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_iteration": int(record.get("checkpoint_iteration", 0)),
                "ceia_overall": ceia_overall,
                "ceia_blue": blue,
                "ceia_orange": orange,
                "side_gap": side_gap,
                "selection_score": selection_score,
                "config_path": str(find_config_path(checkpoint_path)),
                "line_number": line_number,
            }
            candidates_by_run.setdefault(run_name, []).append(candidate)

    selected = []
    for run_name in SEED_RUN_NAMES:
        run_candidates = candidates_by_run.get(run_name, [])
        run_candidates = sorted(
            run_candidates,
            key=lambda item: (
                item["selection_score"],
                item["ceia_overall"],
                -item["side_gap"],
                item["checkpoint_iteration"],
            ),
            reverse=True,
        )
        selected.extend(run_candidates[: args.max_seeds_per_run])

    selected = sorted(
        selected,
        key=lambda item: (
            item["selection_score"],
            item["ceia_overall"],
            -item["side_gap"],
            item["checkpoint_iteration"],
        ),
        reverse=True,
    )
    return selected[: args.max_seeds_total], skipped


def print_seed_summary(selected, skipped):
    print("\nSelected compatible seed checkpoints:")
    if not selected:
        print("  none")
    for index, seed in enumerate(selected, start=1):
        print(
            f"{index:02d}. run={seed['run_name']} "
            f"iter={seed['checkpoint_iteration']} "
            f"ceia={seed['ceia_overall']:.2f} "
            f"blue={seed['ceia_blue']:.2f} "
            f"orange={seed['ceia_orange']:.2f} "
            f"score={seed['selection_score']:.3f}"
        )
        print(f"    checkpoint={seed['checkpoint_path']}")
    print("\nSeed selection skipped counts:")
    for reason, count in skipped.items():
        print(f"  {reason}: {count}")


def confirm_seed_selection(args, selected):
    if args.yes:
        return True
    if not selected:
        return False
    response = input("\nContinue training with these seed checkpoints? [y/N] ")
    return response.strip().lower() in ("y", "yes")


def prepare_restore_config(config, obs_space, act_space):
    restore_config = copy.deepcopy(config)
    restore_config.pop("callbacks", None)
    restore_config["num_workers"] = 0
    restore_config["num_gpus"] = 0
    restore_config["evaluation_interval"] = None
    restore_config["evaluation_num_workers"] = 0
    restore_config["evaluation_num_episodes"] = 0
    restore_config["evaluation_config"] = {}
    restore_config["env"] = "SeedPolicyExtractDummyEnv"
    restore_config["env_config"] = {
        "observation_space": obs_space,
        "action_space": act_space,
    }

    multiagent = restore_config.get("multiagent")
    policies = (multiagent or {}).get("policies", {})
    if policies:
        policy_ids = list(policies.keys())
        restore_config["multiagent"] = {
            "policies": {
                policy_id: (None, obs_space, act_space, {}) for policy_id in policy_ids
            },
            "policy_mapping_fn": tune.function(lambda agent_id: policy_ids[0]),
            "policies_to_train": policy_ids,
        }

    return restore_config


def get_restored_policy(trainer):
    for policy_id in (CURRENT_POLICY, "default"):
        policy = trainer.get_policy(policy_id)
        if policy is not None:
            return policy, policy_id

    policy = trainer.get_policy()
    if policy is not None:
        return policy, "default_policy"

    policy_map = trainer.workers.local_worker().policy_map
    if len(policy_map) == 1:
        policy_id, policy = next(iter(policy_map.items()))
        return policy, policy_id

    return None, None


def extract_seed_policy_weights(selected, obs_space, act_space, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    tune.registry.register_env(
        "SeedPolicyExtractDummyEnv",
        lambda env_config: DummyPolicyEnv(env_config),
    )

    extracted = []
    trainer_cls = get_trainable_cls(ALGORITHM)
    for seed in selected:
        config = load_checkpoint_config(seed["checkpoint_path"])
        restore_config = prepare_restore_config(config, obs_space, act_space)
        trainer = trainer_cls(env=restore_config["env"], config=restore_config)
        try:
            trainer.restore(seed["checkpoint_path"])
            policy, policy_id = get_restored_policy(trainer)
            if policy is None:
                raise RuntimeError(
                    f"Could not find restorable policy in {seed['checkpoint_path']}"
                )
            weights = policy.get_weights()
        finally:
            trainer.stop()

        trial_token = Path(seed.get("trial_dir") or "unknown_trial").name
        weights_path = output_dir / (
            f"{seed['run_name']}_{trial_token}_iter_"
            f"{seed['checkpoint_iteration']:06d}.pkl"
        )
        with open(weights_path, "wb") as weights_file:
            pickle.dump(weights, weights_file)

        entry = dict(seed)
        entry["weights_path"] = str(weights_path)
        entry["policy_id"] = policy_id
        extracted.append(entry)
        print(
            "[seeded_selfplay] extracted "
            f"run={entry['run_name']} "
            f"iter={entry['checkpoint_iteration']} "
            f"policy={policy_id} "
            f"weights={weights_path}"
        )

    return extracted


def extract_init_weights(checkpoint_path, obs_space, act_space):
    """Load policy weights from a checkpoint for warm-starting current_team.

    Reuses the same DummyPolicyEnv / restore machinery as seed extraction but
    returns the weights dict directly without writing a pkl file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"--init-checkpoint not found: {checkpoint_path}")

    tune.registry.register_env(
        "SeedPolicyExtractDummyEnv",
        lambda env_config: DummyPolicyEnv(env_config),
    )
    config = load_checkpoint_config(str(checkpoint_path))
    restore_config = prepare_restore_config(config, obs_space, act_space)
    trainer_cls = get_trainable_cls(ALGORITHM)
    trainer = trainer_cls(env=restore_config["env"], config=restore_config)
    try:
        trainer.restore(str(checkpoint_path))
        policy, policy_id = get_restored_policy(trainer)
        if policy is None:
            raise RuntimeError(
                f"Could not find a usable policy in --init-checkpoint {checkpoint_path}"
            )
        weights = policy.get_weights()
    finally:
        trainer.stop()

    print(
        f"[seeded_selfplay] init_weights loaded "
        f"policy_id={policy_id} checkpoint={checkpoint_path}"
    )
    return weights


def main():
    args = parse_args()
    target_hiddens = parse_model_hiddens(args.model_hiddens)
    exp_name = args.exp_name or default_exp_name(args)

    selected, skipped = select_seed_candidates(args, target_hiddens)
    print_seed_summary(selected, skipped)
    if args.dry_run_seeds:
        return
    if not confirm_seed_selection(args, selected):
        print("Aborted before Ray startup/training.")
        return

    init_ray()
    tune.registry.register_env("Soccer", create_rllib_env)

    temp_env = create_rllib_env(env_config(args))
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    seed_output_dir = Path(args.seed_output_dir)
    if not seed_output_dir.is_absolute():
        seed_output_dir = REPO_ROOT / seed_output_dir
    seed_pool = extract_seed_policy_weights(
        selected,
        obs_space,
        act_space,
        seed_output_dir / exp_name,
    )

    init_weights = None
    if args.init_checkpoint:
        init_weights = extract_init_weights(args.init_checkpoint, obs_space, act_space)

    configure_callback_settings(args, seed_pool, init_weights)

    print(f"\nStarting seeded historical self-play run: {exp_name}")
    print(f"reward_shaping={args.reward_shaping}")
    print(f"goal_progress_weight={args.goal_progress_weight}")
    print(f"retreat_penalty_weight={args.retreat_penalty_weight}")
    print(f"goal_potential_scale={args.goal_potential_scale}")
    print(f"model_hiddens={target_hiddens}")
    print(f"vf_share_layers={args.vf_share_layers}")
    print(f"seed_pool_size={len(seed_pool)}")
    print(
        "opponent_mix="
        f"seed_weighted:{args.seed_weighted_prob}, "
        f"history:{args.history_prob}, "
        f"current:{args.current_prob}, "
        f"seed_uniform:{args.seed_uniform_prob}"
    )

    analysis = tune.run(
        ALGORITHM,
        name=exp_name,
        restore=args.restore,
        config={
            "num_gpus": 0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "multiagent": build_multiagent_config(args, obs_space, act_space),
            "env": "Soccer",
            "env_config": env_config(args),
            "model": {
                "vf_share_layers": args.vf_share_layers,
                "fcnet_hiddens": target_hiddens,
                "fcnet_activation": "relu",
            },
            "train_batch_size": args.train_batch_size,
            "lr": args.lr,
            "num_sgd_iter": args.num_sgd_iter,
            "sgd_minibatch_size": args.sgd_batch_size,
            "callbacks": SeededHistoricalSelfPlayCallbacks,
            "evaluation_interval": 50,
            "evaluation_num_workers": 1,
            "evaluation_num_episodes": 10,
            "evaluation_config": {
                "explore": False,
                "env_config": env_config(args, include_shaping=False),
            },
        },
        stop={"timesteps_total": args.timesteps_total},
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
                trial=best_trial,
                metric="episode_reward_mean",
                mode="max",
            )
        except ValueError:
            best_checkpoint = None
    print(best_checkpoint)
    print("Done training")

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
