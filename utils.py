from random import uniform as randfloat
import fcntl
import os
import socket
import sys

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


def is_episode_done(done):
    """Handle both Gym bool done and RLlib multiagent done dicts."""
    if isinstance(done, dict):
        return bool(done.get("__all__", False))
    return bool(done)


def extract_ball_x(info):
    """Extract ball x-position from either single-team or multiagent-team info."""
    if not isinstance(info, dict):
        return None

    player_ball_x = extract_ball_x_from_player_info(info)
    if player_ball_x is not None:
        return player_ball_x

    for team_info in info.values():
        if not isinstance(team_info, dict):
            continue
        for player_info in team_info.values():
            player_ball_x = extract_ball_x_from_player_info(player_info)
            if player_ball_x is not None:
                return player_ball_x
    return None


def extract_ball_x_from_player_info(player_info):
    if not isinstance(player_info, dict):
        return None
    ball_info = player_info.get("ball_info")
    if not isinstance(ball_info, dict):
        return None
    position = ball_info.get("position")
    if position is None or len(position) < 1:
        return None
    return float(position[0])


def make_reward_shaping_info(
    mode,
    progress_bonus,
    clear_bonus,
    base_reward,
    shaped_reward,
    ball_x,
):
    return {
        "mode": mode,
        "progress_bonus": progress_bonus,
        "clear_bonus": clear_bonus,
        "base_reward": base_reward,
        "shaped_reward": shaped_reward,
        "ball_x": ball_x,
    }


def compute_progress_clear_bonus(
    team_progress,
    progress_weight,
    clear_weight,
    is_clear,
):
    progress_bonus = progress_weight * team_progress
    clear_bonus = (
        clear_weight * team_progress
        if clear_weight > 0.0 and is_clear and team_progress > 0.0
        else 0.0
    )
    return progress_bonus, clear_bonus


def attach_team_reward_shaping_info(info, shaping_by_team):
    info = dict(info) if isinstance(info, dict) else {}
    for team_id, shaping in shaping_by_team.items():
        team_info = dict(info.get(team_id, {}))
        team_info["reward_shaping"] = shaping
        info[team_id] = team_info
    return info


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
        current_ball_x = extract_ball_x(info)
        previous_ball_x = self._previous_ball_x
        progress_bonus = 0.0
        clear_bonus = 0.0

        if current_ball_x is not None and previous_ball_x is not None:
            ball_delta_x = current_ball_x - previous_ball_x

            if self.shaping_mode == "custom":
                progress_bonus, clear_bonus = compute_progress_clear_bonus(
                    team_progress=ball_delta_x,
                    progress_weight=self.ball_progress_weight,
                    clear_weight=self.defensive_clear_weight,
                    is_clear=previous_ball_x <= self.defensive_half_threshold,
                )
                shaped_reward += progress_bonus + clear_bonus

                if self.debug and ball_delta_x != 0.0:
                    print(
                        "[reward_shaping_debug] "
                        f"wrapper={id(self)} "
                        f"prev_ball_x={previous_ball_x:.3f} "
                        f"ball_x={current_ball_x:.3f} "
                        f"delta_x={current_ball_x - previous_ball_x:.4f} "
                        f"base={reward:.4f} "
                        f"progress_bonus={progress_bonus:.4f} "
                        f"clear_bonus={clear_bonus:.4f} "
                        f"shaped={shaped_reward:.4f}"
                    )

        # Update the reference after shaping so the next step uses the current
        # ball location as the new baseline.
        self._previous_ball_x = current_ball_x

        info = dict(info) if isinstance(info, dict) else {}
        # Expose the decomposition for debugging and later analysis. This makes
        # it easier to verify that shaping is active and to inspect whether the
        # coefficient magnitudes are reasonable.
        info["reward_shaping"] = make_reward_shaping_info(
            mode=self.shaping_mode,
            progress_bonus=progress_bonus,
            clear_bonus=clear_bonus,
            base_reward=reward,
            shaped_reward=shaped_reward,
            ball_x=current_ball_x,
        )


        if is_episode_done(done):
            # Avoid carrying ball state across episode boundaries.
            self._previous_ball_x = None

        return obs, shaped_reward, done, info


class MultiagentTeamRewardShapingWrapper(gym.core.Wrapper):
    """
    Symmetric reward shaping for multiagent-team self-play.

    `MultiagentTeamWrapper` exposes two RLlib agents:
    - team 0 controls players 0 and 1, and attacks toward +x
    - team 1 controls players 2 and 3, and attacks toward -x

    The single-team shaping wrapper cannot be reused here because applying
    `+delta_x` to both teams would reward one side for moving the ball toward
    its own goal. This wrapper mirrors the shaping signal by team.

    Important assumption:
    The shaping direction is tied to Soccer-Twos' current team orientation:
    team 0/blue attacks toward +x and team 1/orange attacks toward -x. If a
    curriculum or custom wrapper ever swaps team orientation before this wrapper
    runs, the progress signs must be re-validated.
    """

    BLUE_TEAM_ID = 0
    ORANGE_TEAM_ID = 1

    def __init__(
        self,
        env,
        ball_progress_weight: float = 0.05,
        defensive_clear_weight: float = 0.0,
        defensive_half_threshold: float = -4.0,
        debug: bool = False,
    ):
        super().__init__(env)
        self.ball_progress_weight = ball_progress_weight
        self.defensive_clear_weight = defensive_clear_weight
        self.defensive_half_threshold = defensive_half_threshold
        self.debug = debug
        self._previous_ball_x = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._previous_ball_x = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        shaped_reward = dict(reward)
        current_ball_x = extract_ball_x(info)
        previous_ball_x = self._previous_ball_x
        shaping_by_team = {
            self.BLUE_TEAM_ID: make_reward_shaping_info(
                mode="multiagent_custom",
                progress_bonus=0.0,
                clear_bonus=0.0,
                base_reward=reward.get(self.BLUE_TEAM_ID, 0.0),
                shaped_reward=reward.get(self.BLUE_TEAM_ID, 0.0),
                ball_x=current_ball_x,
            ),
            self.ORANGE_TEAM_ID: make_reward_shaping_info(
                mode="multiagent_custom",
                progress_bonus=0.0,
                clear_bonus=0.0,
                base_reward=reward.get(self.ORANGE_TEAM_ID, 0.0),
                shaped_reward=reward.get(self.ORANGE_TEAM_ID, 0.0),
                ball_x=current_ball_x,
            ),
        }

        if current_ball_x is not None and previous_ball_x is not None:
            ball_delta_x = current_ball_x - previous_ball_x

            blue_progress = ball_delta_x
            orange_progress = -ball_delta_x

            self._apply_team_shaping(
                shaped_reward,
                shaping_by_team,
                self.BLUE_TEAM_ID,
                blue_progress,
                is_clear=(
                    previous_ball_x <= self.defensive_half_threshold
                    and ball_delta_x > 0.0
                ),
            )
            self._apply_team_shaping(
                shaped_reward,
                shaping_by_team,
                self.ORANGE_TEAM_ID,
                orange_progress,
                is_clear=(
                    previous_ball_x >= -self.defensive_half_threshold
                    and ball_delta_x < 0.0
                ),
            )

            if self.debug and ball_delta_x != 0.0:
                blue_shaping = shaping_by_team[self.BLUE_TEAM_ID]
                orange_shaping = shaping_by_team[self.ORANGE_TEAM_ID]
                print(
                    "[ma_reward_shaping_debug] "
                    f"wrapper={id(self)} "
                    f"prev_ball_x={previous_ball_x:.3f} "
                    f"ball_x={current_ball_x:.3f} "
                    f"delta_x={current_ball_x - previous_ball_x:.4f} "
                    f"blue_base={reward.get(self.BLUE_TEAM_ID, 0.0):.4f} "
                    f"blue_progress={blue_shaping['progress_bonus']:.4f} "
                    f"blue_clear={blue_shaping['clear_bonus']:.4f} "
                    f"blue_shaped={shaped_reward.get(self.BLUE_TEAM_ID, 0.0):.4f} "
                    f"orange_base={reward.get(self.ORANGE_TEAM_ID, 0.0):.4f} "
                    f"orange_progress={orange_shaping['progress_bonus']:.4f} "
                    f"orange_clear={orange_shaping['clear_bonus']:.4f} "
                    f"orange_shaped={shaped_reward.get(self.ORANGE_TEAM_ID, 0.0):.4f}"
            )

        self._previous_ball_x = current_ball_x
        info = attach_team_reward_shaping_info(info, shaping_by_team)

        if is_episode_done(done):
            self._previous_ball_x = None

        return obs, shaped_reward, done, info

    def _apply_team_shaping(
        self,
        shaped_reward,
        shaping_by_team,
        team_id,
        team_progress,
        is_clear,
    ):
        progress_bonus, clear_bonus = compute_progress_clear_bonus(
            team_progress=team_progress,
            progress_weight=self.ball_progress_weight,
            clear_weight=self.defensive_clear_weight,
            is_clear=is_clear,
        )
        shaped_reward[team_id] = (
            shaped_reward.get(team_id, 0.0) + progress_bonus + clear_bonus
        )
        shaping_by_team[team_id]["progress_bonus"] = progress_bonus
        shaping_by_team[team_id]["clear_bonus"] = clear_bonus
        shaping_by_team[team_id]["shaped_reward"] = shaped_reward[team_id]


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
        infos = []
        seen_info_ids = set()
        for agent_id in (None, 0, 1):
            info = (
                episode.last_info_for()
                if agent_id is None
                else episode.last_info_for(agent_id)
            )
            if info is None or id(info) in seen_info_ids:
                continue
            seen_info_ids.add(id(info))
            infos.append(info)

        for info in infos:
            if not isinstance(info, dict):
                continue

            shaping = info.get("reward_shaping")
            if not isinstance(shaping, dict):
                continue

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
        variation = env_config.get("variation")
        env_type = soccer_twos.EnvType(variation) if variation is not None else None
        if env_type is soccer_twos.EnvType.team_vs_policy:
            env = TeamRewardShapingWrapper(
                env,
                shaping_mode=shaping_mode,
                ball_progress_weight=env_config.get("ball_progress_weight", 0.05),
                defensive_clear_weight=env_config.get("defensive_clear_weight", 0.1),
                defensive_half_threshold=env_config.get(
                    "defensive_half_threshold", -6.0
                ),
                debug=env_config.get("reward_shaping_debug", False),
            )
        elif env_type is soccer_twos.EnvType.multiagent_team:
            env = MultiagentTeamRewardShapingWrapper(
                env,
                ball_progress_weight=env_config.get("ball_progress_weight", 0.05),
                defensive_clear_weight=env_config.get("defensive_clear_weight", 0.0),
                defensive_half_threshold=env_config.get(
                    "defensive_half_threshold", -4.0
                ),
                debug=env_config.get("reward_shaping_debug", False),
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
