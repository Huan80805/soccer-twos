from random import uniform as randfloat
import fcntl
import os
import socket

import gym
import ray
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass

class TeamRewardShapingWrapper(gym.core.Wrapper):
    """
    Reward shaping for team-vs-policy training.

    This wrapper expects a single-agent, team-level environment such as the
    output of soccer_twos.make(... variation=team_vs_policy, multiagent=False).

    A generic "custom" mode is supported so experiments can be controlled by
    turning individual weights on/off without changing the wrapper logic:

    - set `ball_progress_weight > 0` to reward moving the ball toward the
      opponent goal along the x-axis
    - set `defensive_clear_weight > 0` to additionally reward moving the ball
      out of our defensive half

    Important assumption:
    We treat increasing ball x as progress toward the attacking goal for the
    controlled team. That matches the current curriculum/setup in this repo,
    but should be re-checked if team orientation changes.
    """

    def __init__(
        self,
        env,
        shaping_mode: str,
        ball_progress_weight: float = 0.05,
        defensive_clear_weight: float = 0.1,
        defensive_half_threshold: float = -6.0,
        debug: bool = False,
    ):
        super().__init__(env)
        self.shaping_mode = shaping_mode
        self.ball_progress_weight = ball_progress_weight
        self.defensive_clear_weight = defensive_clear_weight
        self.defensive_half_threshold = defensive_half_threshold
        self.debug = debug
        self._previous_ball_x = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # We shape from step-to-step ball movement, so each episode starts with
        # no previous position reference.
        self._previous_ball_x = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Start from the environment's original team reward so shaping remains
        # additive and easy to disable for baseline comparisons.
        shaped_reward = reward
        current_ball_x = self._extract_ball_x(info)
        progress_bonus = 0.0
        clear_bonus = 0.0

        if current_ball_x is not None and self._previous_ball_x is not None:
            ball_delta_x = current_ball_x - self._previous_ball_x

            if self.shaping_mode == "custom":
                if self.ball_progress_weight > 0.0:
                    # Dense progress shaping: reward moving the ball toward the
                    # opponent goal and penalize drift back toward our own goal.
                    progress_bonus = self.ball_progress_weight * ball_delta_x
                    shaped_reward += progress_bonus
                if self.defensive_clear_weight > 0.0 and (
                    # Only count a "clear" if the ball was previously in our
                    # defensive half and the new movement sends it outward.
                    self._previous_ball_x <= self.defensive_half_threshold
                    and ball_delta_x > 0.0
                ):
                    clear_bonus = self.defensive_clear_weight * ball_delta_x
                    shaped_reward += clear_bonus

        # Update the reference after shaping so the next step uses the current
        # ball location as the new baseline.
        self._previous_ball_x = current_ball_x

        info = dict(info) if isinstance(info, dict) else {}
        # Expose the decomposition for debugging and later analysis. This makes
        # it easier to verify that shaping is active and to inspect whether the
        # coefficient magnitudes are reasonable.
        info["reward_shaping"] = {
            "mode": self.shaping_mode,
            "progress_bonus": progress_bonus,
            "clear_bonus": clear_bonus,
            "base_reward": reward,
            "shaped_reward": shaped_reward,
            "ball_x": current_ball_x,
        }

        if self.debug and current_ball_x is not None:
            print(
                "[reward_shaping_debug] "
                f"ball_x={current_ball_x:.3f} "
                f"base={reward:.4f} "
                f"progress_bonus={progress_bonus:.4f} "
                f"clear_bonus={clear_bonus:.4f} "
                f"shaped={shaped_reward:.4f}"
            )

        if done:
            # Avoid carrying ball state across episode boundaries.
            self._previous_ball_x = None

        return obs, shaped_reward, done, info

    @staticmethod
    def _extract_ball_x(info):
        # The installed soccer_twos wrappers attach auxiliary simulator state to
        # `info["ball_info"]["position"]` when available. We only need the x
        # coordinate here because both shaping rules are defined along the goal
        # direction of the field.
        if not isinstance(info, dict):
            return None
        ball_info = info.get("ball_info")
        if not isinstance(ball_info, dict):
            return None
        position = ball_info.get("position")
        if position is None or len(position) < 1:
            return None
        return float(position[0])


class RewardShapingMetricsCallbacks(DefaultCallbacks):
    """
    Logs reward-shaping components into RLlib custom metrics so they appear in
    Tune results and TensorBoard.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.user_data["reward_shaping_progress_bonus_total"] = 0.0
        episode.user_data["reward_shaping_clear_bonus_total"] = 0.0
        episode.user_data["reward_shaping_delta_total"] = 0.0

    def on_episode_step(self, *, worker, base_env, episode, env_index=None, **kwargs):
        info = episode.last_info_for()
        if info is None:
            info = episode.last_info_for(0)
        if not isinstance(info, dict):
            return

        shaping = info.get("reward_shaping")
        if not isinstance(shaping, dict):
            return

        progress_bonus = float(shaping.get("progress_bonus", 0.0))
        clear_bonus = float(shaping.get("clear_bonus", 0.0))
        base_reward = float(shaping.get("base_reward", 0.0))
        shaped_reward = float(shaping.get("shaped_reward", base_reward))

        episode.user_data["reward_shaping_progress_bonus_total"] += progress_bonus
        episode.user_data["reward_shaping_clear_bonus_total"] += clear_bonus
        episode.user_data["reward_shaping_delta_total"] += shaped_reward - base_reward

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.custom_metrics["reward_shaping_progress_bonus_total"] = episode.user_data[
            "reward_shaping_progress_bonus_total"
        ]
        episode.custom_metrics["reward_shaping_clear_bonus_total"] = episode.user_data[
            "reward_shaping_clear_bonus_total"
        ]
        episode.custom_metrics["reward_shaping_delta_total"] = episode.user_data[
            "reward_shaping_delta_total"
        ]

_UNITY_PORT_LOCK = "/tmp/soccer_twos_unity_port.lock"
_UNITY_PORT_STATE = "/tmp/soccer_twos_unity_port.state"


def _port_is_available(port: int) -> bool:
    for family, bind_addr in (
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::", port)),
    ):
        sock = None
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.bind(bind_addr)
        except OSError:
            return False
        finally:
            if sock is not None:
                sock.close()
    return True


def _reserve_unity_base_port(start_port: int) -> int:
    os.makedirs(os.path.dirname(_UNITY_PORT_LOCK), exist_ok=True)
    with open(_UNITY_PORT_LOCK, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        try:
            with open(_UNITY_PORT_STATE, "r") as state_file:
                next_port = int(state_file.read().strip())
        except (FileNotFoundError, ValueError):
            next_port = start_port

        next_port = max(next_port, start_port)
        while next_port < 65535:
            if _port_is_available(next_port):
                with open(_UNITY_PORT_STATE, "w") as state_file:
                    state_file.write(str(next_port + 1))
                return next_port
            next_port += 1

    raise RuntimeError("Could not reserve a free Unity base port.")


def init_ray():
    mode = os.environ.get("SOCCER_TWOS_RAY_INIT", "default").lower()
    if mode == "default":
        return ray.init()
    if mode == "auto":
        return ray.init(address="auto")
    if mode == "no-dashboard":
        return ray.init(
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            log_to_driver=os.environ.get("SOCCER_TWOS_RAY_LOG_TO_DRIVER", "0")
            == "1",
        )
    raise ValueError(
        "Unsupported SOCCER_TWOS_RAY_INIT value. "
        "Use one of: default, auto, no-dashboard"
    )


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    env_config = dict(env_config)
    requested_base_port = env_config.get("base_port", 50039)
    env_config["base_port"] = _reserve_unity_base_port(requested_base_port)
    env_config["worker_id"] = 0
    env = soccer_twos.make(**env_config)

    shaping_mode = env_config.get("reward_shaping")
    if shaping_mode == "custom":
        env = TeamRewardShapingWrapper(
            env,
            shaping_mode=shaping_mode,
            ball_progress_weight=env_config.get("ball_progress_weight", 0.05),
            defensive_clear_weight=env_config.get("defensive_clear_weight", 0.1),
            defensive_half_threshold=env_config.get(
                "defensive_half_threshold", -6.0
            ),
            debug=env_config.get("reward_shaping_debug", False)
        )
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
