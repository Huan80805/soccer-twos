import json
import os
import pickle

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import BaseEnv
from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoint",
    "checkpoint-1250",
)


class DummyEnv(BaseEnv):
    '''
    A dummy env for team agent. The observation and action spaces needed to be defined first.
    '''

    def __init__(self, env_config):
        self.observation_space = env_config["observation_space"]
        self.action_space = env_config["action_space"]


class RewardShapingPPOAgent(AgentInterface):
    """
    PPO agent trained with reward shaping and packaged for evaluation/submission.
    """

    def __init__(self, env):
        super().__init__()

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise ValueError(
                "Could not find params.json in the checkpoint directory!"
            )

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["env"] = "DummyEnv"

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

        # The checkpoint was trained on a concatenated team observation, so the
        # dummy env must expose the team-level observation and action spaces.
        config["env_config"] = {
            "observation_space": self.team_observation_space,
            "action_space": self.team_action_space,
        }
        config.pop("callbacks", None)
        tune.registry.register_env("DummyEnv", lambda env_config: DummyEnv(env_config))

        cls = get_trainable_cls(ALGORITHM)
        self.agent = cls(env=config["env"], config=config)
        self.agent.restore(CHECKPOINT_PATH)
        self.policy = self.agent.get_policy()

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
