from collections import deque

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from soccer_twos import EnvType
import yaml

from utils import create_rllib_env, sample_player, sample_pos_vel


NUM_GPUS = 0
NUM_WORKERS = 4
NUM_ENVS_PER_WORKER = 2
ROLLOUT_FRAGMENT_LENGTH = 500
TRAIN_BATCH_SIZE = 8000
SGD_MINIBATCH_SIZE = 1024
NUM_SGD_ITER = 6
REWARD_WINDOW = 25
STOP_TIMESTEPS_TOTAL = 10000000
STOP_TIME_TOTAL_S = 14400
SELFPLAY_POOL_SIZE = 8
SELFPLAY_SNAPSHOT_INTERVAL = 50
SELFPLAY_CURRENT_POLICY_PROB = 0.2

with open("curriculum.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)

tasks = curriculum["tasks"]

# Slow down promotion in the late curriculum without changing curriculum.yaml.
MIN_ITERATION_OVERRIDES = {
    "medium_random_opponent": 35,
    "wide_random_opponent": 40,
    "fullfield_random_opponent": 45,
    "hard_random_opponent": 50,
}
for task in tasks:
    override = MIN_ITERATION_OVERRIDES.get(task["name"])
    if override is not None:
        task["min_iterations"] = override


def make_random_policy(env):
    return lambda *_: env.action_space.sample()


def make_frozen_policy(_env):
    return lambda *_: 0


def make_current_policy(policy):
    def opponent(obs):
        action, *_ = policy.compute_single_action(obs, explore=False)
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.item())
            return action
        if isinstance(action, np.generic):
            return int(action.item())
        return action

    return opponent


def clone_policy_with_state(policy, state):
    clone = policy.__class__(
        policy.observation_space,
        policy.action_space,
        policy.config,
    )
    clone.set_state(state)
    return clone


def select_selfplay_opponent(worker, default_policy):
    pool_states = getattr(worker, "selfplay_pool_states", [])
    if not pool_states or np.random.random() < SELFPLAY_CURRENT_POLICY_PROB:
        return default_policy

    snapshot_state = pool_states[np.random.randint(len(pool_states))]
    return clone_policy_with_state(default_policy, snapshot_state)


def apply_stage(env, task, default_policy=None, selfplay_opponent=None):
    config_fn = task["config_fn"]

    if config_fn in ("none", "frozen"):
        env.set_policies(make_frozen_policy(env))
    elif config_fn == "random_players":
        env.set_policies(make_random_policy(env))
    elif config_fn == "self_play":
        if selfplay_opponent is not None:
            env.set_policies(make_current_policy(selfplay_opponent))
        elif default_policy is None:
            env.set_policies(make_frozen_policy(env))
        else:
            env.set_policies(make_current_policy(default_policy))
    else:
        raise ValueError(f"Unknown config_fn: {config_fn}")

    env.env_channel.set_parameters(
        ball_state=sample_pos_vel(task["ranges"]["ball"]),
        players_states={
            player: sample_player(task["ranges"]["players"][player])
            for player in task["ranges"]["players"]
        },
    )


class CurriculumSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.current_stage = 0
        self.stage_iterations = 0
        self.stage_rewards = deque(maxlen=REWARD_WINDOW)
        self.selfplay_pool_states = []

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        if not hasattr(worker, "curriculum_stage"):
            worker.curriculum_stage = 0

        task = tasks[worker.curriculum_stage]
        default_policy = policies.get("default_policy")
        selfplay_opponent = None
        if task["config_fn"] == "self_play" and default_policy is not None:
            selfplay_opponent = select_selfplay_opponent(worker, default_policy)

        for env in base_env.get_unwrapped():
            apply_stage(
                env,
                task,
                default_policy=default_policy,
                selfplay_opponent=selfplay_opponent,
            )

    def on_train_result(self, *, trainer, result, **kwargs):
        task = tasks[self.current_stage]
        reward_mean = result["episode_reward_mean"]

        self.stage_iterations += 1
        self.stage_rewards.append(reward_mean)
        reward_window_mean = sum(self.stage_rewards) / len(self.stage_rewards)

        result["curriculum_stage"] = self.current_stage
        result["curriculum_stage_name"] = task["name"]
        result["curriculum_stage_iterations"] = self.stage_iterations
        result["curriculum_stage_reward_window_mean"] = reward_window_mean
        result["selfplay_pool_size"] = len(self.selfplay_pool_states)

        if result["training_iteration"] % SELFPLAY_SNAPSHOT_INTERVAL == 0:
            default_policy = trainer.workers.local_worker().get_policy("default_policy")
            self.selfplay_pool_states.append(default_policy.get_state())
            self.selfplay_pool_states = self.selfplay_pool_states[-SELFPLAY_POOL_SIZE:]

        min_iterations = task.get("min_iterations", 3)
        advance_threshold = task.get("advance_threshold", 1.0)

        if (
            self.stage_iterations >= min_iterations
            and reward_window_mean >= advance_threshold
            and self.current_stage < len(tasks) - 1
        ):
            previous_task = task["name"]
            self.current_stage += 1
            self.stage_iterations = 0
            self.stage_rewards.clear()
            next_task = tasks[self.current_stage]["name"]

            print(
                "---- Advancing curriculum ---- "
                f"{previous_task} -> {next_task} "
                f"(window reward {reward_window_mean:.3f})"
            )

        def sync_worker(w):
            w.curriculum_stage = self.current_stage
            w.selfplay_pool_states = list(self.selfplay_pool_states)

        trainer.workers.foreach_worker(sync_worker)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_curriculum_selfplay_hyp",
        config={
            "num_gpus": NUM_GPUS,
            "num_workers": NUM_WORKERS,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumSelfPlayCallback,
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
                "opponent_policy": lambda *_: 0,
                "base_port": 7005,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
            },
            "lr": 1e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.003,
            "vf_loss_coeff": 1.0,
            "grad_clip": 0.5,
            "rollout_fragment_length": ROLLOUT_FRAGMENT_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "sgd_minibatch_size": SGD_MINIBATCH_SIZE,
            "num_sgd_iter": NUM_SGD_ITER,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": STOP_TIMESTEPS_TOTAL,
            "time_total_s": STOP_TIME_TOTAL_S,
        },
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
