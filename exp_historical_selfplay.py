"""Historical-opponent self-play PPO experiments.

Baseline historical self-play:
    python exp_historical_selfplay.py --port 58000

Reward-shaped historical self-play, implemented but off by default:
    python exp_historical_selfplay.py --port 58000 --reward-shaping custom --goal-progress-weight 0.75 --retreat-penalty-weight 1.25 --goal-potential-scale 6.0

This follows the same high-level pattern as Team Pequi's
`ppo_deepmind_selfplay_v4.py`: train a `current_team` policy, periodically save
its weights, and load sampled historical weights into a frozen `opponent_team`.
"""

import argparse
import pickle
import random
from pathlib import Path

from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env, init_ray, RewardShapingMetricsCallbacks


CURRENT_POLICY = "current_team"
OPPONENT_POLICY = "opponent_team"
DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_NUM_WORKERS = 8
DEFAULT_BASE_PORT = 58000
DEFAULT_TRAIN_BATCH_SIZE = 4000
DEFAULT_TIMESTEPS_TOTAL = 20000000
DEFAULT_LR = 5e-5
DEFAULT_NUM_SGD_ITER = 30
DEFAULT_SGD_BATCH_SIZE = 128
DEFAULT_HISTORY_INTERVAL = 10
DEFAULT_HISTORY_MAX_SIZE = 0
DEFAULT_OPPONENT_CURRENT_PROB = 0.1

HISTORICAL_SELFPLAY_SETTINGS = {}


class TeamMatchMaker:
    """Randomly assign the trainable policy to blue or orange each episode.

    Older RLlib versions call policy_mapping_fn(agent_id) without an Episode
    object. In that mode we mirror Team Pequi's stateful matchmaker: assign both
    team agents for one environment episode, then sample a new side assignment
    for the next two team-agent mapping calls.
    """

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
            policy_id = (
                CURRENT_POLICY if int(agent_id) == current_side else OPPONENT_POLICY
            )
            self._debug_mapping(agent_id, current_side, policy_id, episode)
            return policy_id

        policy_id = (
            CURRENT_POLICY
            if int(agent_id) == self._fallback_current_side
            else OPPONENT_POLICY
        )
        if self.debug:
            self._debug_mapping(
                agent_id=agent_id,
                current_side=self._fallback_current_side,
                policy_id=policy_id,
                episode=None,
            )
        self._fallback_seen_agent_ids.add(int(agent_id))
        if len(self._fallback_seen_agent_ids) >= 2:
            self._fallback_seen_agent_ids.clear()
            self._fallback_current_side = random.choice([0, 1])
        return policy_id

    def _debug_mapping(self, agent_id, current_side, policy_id, episode):
        episode_id = getattr(episode, "episode_id", "none")
        print(
            "[historical_matchmaker_debug] "
            f"episode={episode_id} "
            f"agent_id={agent_id} "
            f"current_side={current_side} "
            f"policy={policy_id}"
        )


class HistoricalOpponentSelfPlayCallbacks(RewardShapingMetricsCallbacks):
    """Maintain and sample a history pool for the frozen opponent policy."""

    def __init__(self):
        super().__init__()
        settings = HISTORICAL_SELFPLAY_SETTINGS
        self.save_interval = settings.get(
            "history_save_interval",
            DEFAULT_HISTORY_INTERVAL,
        )
        self.sample_interval = settings.get(
            "opponent_update_interval",
            DEFAULT_HISTORY_INTERVAL,
        )
        self.max_history_size = settings.get(
            "history_max_size",
            DEFAULT_HISTORY_MAX_SIZE,
        )
        self.opponent_current_prob = settings.get(
            "opponent_current_prob",
            DEFAULT_OPPONENT_CURRENT_PROB,
        )
        self.policy_history = []
        self.counter = 0
        self.current_opponent_source = "initial"
        self.saved_policy_count = 0
        self.evicted_policy_count = 0
        self.debug = settings.get("debug", False)

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
        episode.user_data["historical_current_side"] = current_side

        # Ray 1.4 policy_mapping_fn only receives agent_id, not episode. Set the
        # per-episode mapping here so vectorized envs cannot interleave fallback
        # matchmaker calls and accidentally map both teams to the same policy.
        episode._agent_to_policy[0] = (
            CURRENT_POLICY if current_side == 0 else OPPONENT_POLICY
        )
        episode._agent_to_policy[1] = (
            CURRENT_POLICY if current_side == 1 else OPPONENT_POLICY
        )
        if self.debug:
            print(
                "[historical_matchmaker_debug] "
                f"episode={episode.episode_id} "
                f"source=callback "
                f"current_side={current_side} "
                f"team0_policy={episode._agent_to_policy[0]} "
                f"team1_policy={episode._agent_to_policy[1]}"
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
        current_side = episode.user_data.get("historical_current_side")
        if current_side is not None:
            episode.custom_metrics["historical_current_team_blue"] = (
                1.0 if current_side == 0 else 0.0
            )
            episode.custom_metrics["historical_current_team_orange"] = (
                1.0 if current_side == 1 else 0.0
            )

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        self.counter += 1
        current_weights = trainer.get_weights([CURRENT_POLICY])[CURRENT_POLICY]

        if self.counter % self.save_interval == 0:
            history_entry = self._save_current_policy(trainer, result, current_weights)
            self.policy_history.append(history_entry)
            self.saved_policy_count += 1
            if self.max_history_size > 0 and len(self.policy_history) > self.max_history_size:
                evicted_policy = self.policy_history.pop(0)
                self.evicted_policy_count += 1
                print(f"[historical_selfplay] evicted={evicted_policy}")

        if self.counter % self.sample_interval == 0:
            opponent_weights, source = self._sample_opponent_weights(current_weights)
            trainer.set_weights(
                {
                    CURRENT_POLICY: current_weights,
                    OPPONENT_POLICY: opponent_weights,
                }
            )
            # Ray 1.4 Trainer.set_weights only updates the local worker. PPO
            # normally syncs trainable policies after learning, but
            # opponent_team is frozen, so explicitly broadcast the sampled
            # historical opponent to rollout workers.
            trainer.workers.sync_weights()
            self.current_opponent_source = source
            print(f"[historical_selfplay] opponent={source}")

        custom_metrics = result.setdefault("custom_metrics", {})
        custom_metrics["historical_selfplay_history_size"] = len(self.policy_history)
        custom_metrics["historical_selfplay_counter"] = self.counter
        custom_metrics["historical_selfplay_saved_policy_count"] = (
            self.saved_policy_count
        )
        custom_metrics["historical_selfplay_evicted_policy_count"] = (
            self.evicted_policy_count
        )
        custom_metrics["historical_selfplay_current_opponent_is_current"] = (
            1.0 if self.current_opponent_source == "current_team" else 0.0
        )
        if self.debug:
            print(
                "[historical_selfplay_debug] "
                f"iter={self.counter} "
                f"history_size={len(self.policy_history)} "
                f"saved={self.saved_policy_count} "
                f"evicted={self.evicted_policy_count} "
                f"opponent={self.current_opponent_source}"
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
        print(f"[historical_selfplay] saved={history_entry}")
        return history_entry

    def _sample_opponent_weights(self, current_weights):
        if not self.policy_history or random.random() < self.opponent_current_prob:
            return current_weights, "current_team"

        history_entry = self._sample_history_entry()
        with open(history_entry["weights_path"], "rb") as weights_file:
            opponent_weights = pickle.load(weights_file)
        return opponent_weights, f"history_iter_{history_entry['iteration']}"

    def _sample_history_entry(self):
        if len(self.policy_history) == 1:
            return self.policy_history[0]
        max_index = len(self.policy_history) - 1
        sampled_index = int(random.triangular(0, max_index, max_index))
        sampled_index = max(0, min(max_index, sampled_index))
        return self.policy_history[sampled_index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with historical-opponent self-play."
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
        help="Save current_team into history every N training iterations.",
    )
    parser.add_argument(
        "--opponent-update-interval",
        type=int,
        default=DEFAULT_HISTORY_INTERVAL,
        help="Sample opponent_team from history every N training iterations.",
    )
    parser.add_argument(
        "--history-max-size",
        type=int,
        default=DEFAULT_HISTORY_MAX_SIZE,
        help="Maximum saved opponents to keep. Use 0 for unbounded history.",
    )
    parser.add_argument(
        "--opponent-current-prob", "-ocp",
        type=float,
        default=DEFAULT_OPPONENT_CURRENT_PROB,
        help=(
            "Probability of using current_team weights as the frozen opponent. "
            "The remaining probability samples from historical checkpoints."
        ),
    )
    parser.add_argument(
        "--reward-shaping",
        choices=["none", "custom"],
        default="none",
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
        "--debug",
        action="store_true",
        help="Print matchmaker, historical-opponent, and reward-shaping debug information.",
    )
    parser.add_argument(
        "--restore",
        default=None,
        help="Path to a Ray checkpoint to resume from, e.g. ray_results/.../checkpoint_004400/checkpoint-4400",
    )
    return parser.parse_args()


def weight_token(value):
    if value == 0:
        return "0"
    return f"{value:.6g}".replace(".", "").replace("-", "m")


def hparam_token(value):
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def default_exp_name(args):
    if args.reward_shaping == "custom":
        return (
            "PPO_historical_selfplay_reward"
            f"_goal{weight_token(args.goal_progress_weight)}"
            f"_retreat{weight_token(args.retreat_penalty_weight)}"
            f"_scale{weight_token(args.goal_potential_scale)}"
            f"_lr{hparam_token(args.lr)}"
            f"_sgd{args.num_sgd_iter}"
            f"_updateInterval{args.opponent_update_interval}"
        )
    return (
        "PPO_historical_selfplay_baseline"
        f"_lr{hparam_token(args.lr)}"
        f"_sgd{args.num_sgd_iter}"
        f"_updateInterval{args.opponent_update_interval}"
    )


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


def configure_callback_settings(args):
    HISTORICAL_SELFPLAY_SETTINGS.clear()
    HISTORICAL_SELFPLAY_SETTINGS.update(
        {
            "history_save_interval": args.history_save_interval,
            "opponent_update_interval": args.opponent_update_interval,
            "history_max_size": args.history_max_size,
            "opponent_current_prob": args.opponent_current_prob,
            "debug": args.debug,
        }
    )


def main():
    args = parse_args()
    exp_name = args.exp_name or default_exp_name(args)
    configure_callback_settings(args)

    init_ray()
    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env(env_config(args))
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    print(f"Starting historical self-play run: {exp_name}")
    print(f"reward_shaping={args.reward_shaping}")
    print(f"goal_progress_weight={args.goal_progress_weight}")
    print(f"retreat_penalty_weight={args.retreat_penalty_weight}")
    print(f"goal_potential_scale={args.goal_potential_scale}")
    print(f"history_save_interval={args.history_save_interval}")
    print(f"opponent_update_interval={args.opponent_update_interval}")
    print(f"history_max_size={args.history_max_size}")
    print(f"opponent_current_prob={args.opponent_current_prob}")
    print(f"debug={args.debug}")

    analysis = tune.run(
        "PPO",
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
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",

            },
            "train_batch_size": args.train_batch_size,
            "lr": args.lr,
            "num_sgd_iter": args.num_sgd_iter,
            "sgd_minibatch_size": args.sgd_batch_size,
            "callbacks": HistoricalOpponentSelfPlayCallbacks,
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


if __name__ == "__main__":
    main()
