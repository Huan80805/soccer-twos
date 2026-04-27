import json
import os

import gym
import numpy as np
import ray
from ray import tune
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import BaseEnv
from soccer_twos import AgentInterface


ALGORITHM = "PPO"
# Copy the chosen checkpoint directory contents into selfplay_ppo_agent/checkpoint/
# and update the filename below (e.g. "checkpoint-4400").
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoint",
    "checkpoint-4400",
)


class DummyEnv(BaseEnv):
    """Minimal env used only while restoring the checkpoint — never stepped."""

    def __init__(self, env_config):
        self.observation_space = env_config["observation_space"]
        self.action_space = env_config["action_space"]


class SelfPlayPPOAgent(AgentInterface):
    """
    PPO agent trained with multiagent-team self-play (shared or historical-opponent
    variant) and packaged for evaluation/submission.

    Works with checkpoints that expose either a 'current_team' policy
    (historical-opponent self-play) or a 'default' policy (shared self-play).
    """

    def __init__(self, env):
        super().__init__()

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                log_to_driver=False,
            )

        config_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), "params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json next to the checkpoint. "
                f"Expected: {config_path}"
            )
        with open(config_path) as f:
            config = json.load(f)

        self.player_action_space = env.action_space
        self.player_action_dim = len(env.action_space.nvec)
        # The self-play policy observes the concatenated observations of both
        # team members (same layout as the team_vs_policy baseline).
        self.team_observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            dtype=np.float32,
            shape=(env.observation_space.shape[0] * 2,),
        )
        self.team_action_space = gym.spaces.MultiDiscrete(
            np.repeat(env.action_space.nvec, 2)
        )

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["env"] = "SelfPlayDummyEnv"
        config["env_config"] = {
            "observation_space": self.team_observation_space,
            "action_space": self.team_action_space,
        }
        config.pop("callbacks", None)

        # The serialized policy_mapping_fn closure in params.json cannot be
        # reconstructed at load time, so replace the entire multiagent section
        # with a trivial equivalent that uses the correct spaces.
        existing_policies = config.get("multiagent", {}).get("policies", {})
        policy_ids = list(existing_policies.keys()) if existing_policies else ["default"]
        first_pid = policy_ids[0]
        config["multiagent"] = {
            "policies": {
                pid: (None, self.team_observation_space, self.team_action_space, {})
                for pid in policy_ids
            },
            "policy_mapping_fn": tune.function(lambda agent_id, pid=first_pid: pid),
            "policies_to_train": policy_ids,
        }

        tune.registry.register_env(
            "SelfPlayDummyEnv",
            lambda env_config: DummyEnv(env_config),
        )

        cls = get_trainable_cls(ALGORITHM)
        self.agent = cls(env=config["env"], config=config)
        self.agent.restore(CHECKPOINT_PATH)

        # historical-opponent self-play uses 'current_team'; shared self-play
        # uses 'default'. Try both so this wrapper handles either checkpoint.
        policy = self.agent.get_policy("current_team")
        if policy is None:
            policy = self.agent.get_policy("default")
        if policy is None:
            policy = self.agent.get_policy()
        if policy is None:
            raise RuntimeError(
                "Could not locate a usable policy in the restored checkpoint."
            )
        self.policy = policy

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
