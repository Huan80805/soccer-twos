import os
import pickle
import json
from typing import Dict

import gym
from gym_unity.envs import ActionFlattener
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
from soccer_twos import AgentInterface


ALGORITHM = "PPO"
POLICY_NAME = "default_policy"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PACKAGE_DIR, "checkpoint_000779")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint-779")
CONFIG_JSON_PATH = os.path.join(PACKAGE_DIR, "params.json")
CONFIG_PKL_PATH = os.path.join(PACKAGE_DIR, "params.pkl")
LEGACY_RUN_DIR = os.path.join(
    os.path.dirname(PACKAGE_DIR),
    "ray_results",
    "PPO_curriculum_selfplay_hyp",
    "PPO_Soccer_b2847_00000_0_2026-04-23_00-55-49",
)
INFERENCE_OBSERVATION_SPACE = None
INFERENCE_ACTION_SPACE = None


class _InferenceEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, {}


def _create_inference_env(_env_config):
    return _InferenceEnv(INFERENCE_OBSERVATION_SPACE, INFERENCE_ACTION_SPACE)


def _load_config(env_name):
    json_candidate_paths = [
        CONFIG_JSON_PATH,
        os.path.join(LEGACY_RUN_DIR, "params.json"),
        os.path.join(CHECKPOINT_DIR, "..", "params.json"),
        os.path.join(CHECKPOINT_DIR, "params.json"),
    ]
    json_config_path = next(
        (path for path in json_candidate_paths if os.path.exists(path)), None
    )
    if json_config_path is not None:
        with open(json_config_path, "r") as config_file:
            config = json.load(config_file)
    else:
        candidate_paths = [
            CONFIG_PKL_PATH,
            os.path.join(LEGACY_RUN_DIR, "params.pkl"),
            os.path.join(CHECKPOINT_DIR, "..", "params.pkl"),
            os.path.join(CHECKPOINT_DIR, "params.pkl"),
        ]
        config_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if config_path is None:
            raise ValueError("Could not find params.json or params.pkl for the PPO checkpoint.")

        with open(config_path, "rb") as config_file:
            config = pickle.load(config_file)

    # These training-time fields are not needed for inference restore and can
    # cause compatibility issues across environments.
    config.pop("callbacks", None)
    config.pop("env_config", None)
    config.pop("num_envs_per_worker", None)

    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["env"] = env_name
    return config


class CurriculumTeamAgent(AgentInterface):
    """
    PPO agent for policies trained on `team_vs_policy` with `single_player=True`.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.name = "PPOCurriculumTeam"
        self.flattener = ActionFlattener(env.action_space.nvec)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        if not os.path.isfile(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        global INFERENCE_OBSERVATION_SPACE, INFERENCE_ACTION_SPACE
        INFERENCE_OBSERVATION_SPACE = env.observation_space
        INFERENCE_ACTION_SPACE = self.flattener.action_space

        env_name = "CurriculumInferenceEnv"
        tune.registry.register_env(env_name, _create_inference_env)

        config = _load_config(env_name)
        trainer_cls = get_trainable_cls(ALGORITHM)
        self.trainer = trainer_cls(env=config["env"], config=config)
        self.trainer.restore(CHECKPOINT_PATH)
        self.policy = self.trainer.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, player_obs in observation.items():
            action_index, *_ = self.policy.compute_single_action(
                np.asarray(player_obs),
                explore=False,
            )
            actions[player_id] = self.flattener.lookup_action(int(action_index))
        return actions
