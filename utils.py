from random import uniform as randfloat
import fcntl
import math
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


def extract_ball_position(info):
    """Extract ball (x, y) position from either single-team or multiagent-team info."""
    if not isinstance(info, dict):
        return None

    player_ball_position = extract_ball_position_from_player_info(info)
    if player_ball_position is not None:
        return player_ball_position

    for team_info in info.values():
        if not isinstance(team_info, dict):
            continue
        ball_position = extract_ball_position_from_player_info(team_info)
        if ball_position is not None:
            return ball_position
        for player_info in team_info.values():
            player_ball_position = extract_ball_position_from_player_info(player_info)
            if player_ball_position is not None:
                return player_ball_position
    return None


def extract_ball_position_from_player_info(player_info):
    if not isinstance(player_info, dict):
        return None
    ball_info = player_info.get("ball_info")
    if not isinstance(ball_info, dict):
        return None
    position = ball_info.get("position")
    if position is None or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def make_reward_shaping_info(
    mode,
    goal_progress_bonus,
    retreat_penalty,
    base_reward,
    shaped_reward,
    ball_x,
    ball_y=None,
    potential_delta=0.0,
):
    return {
        "mode": mode,
        "goal_progress_bonus": goal_progress_bonus,
        "retreat_penalty": retreat_penalty,
        "base_reward": base_reward,
        "shaped_reward": shaped_reward,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "potential_delta": potential_delta,
    }


def goal_potential(ball_position, attacking_goal_x, goal_potential_scale):
    if goal_potential_scale <= 0.0:
        raise ValueError("goal_potential_scale must be positive.")
    x, y = ball_position
    distance_to_goal = math.sqrt((attacking_goal_x - x) ** 2 + y ** 2)
    return math.exp(-distance_to_goal / goal_potential_scale)


def compute_goal_potential_bonus(
    previous_ball_position,
    current_ball_position,
    attacking_goal_x,
    goal_progress_weight,
    retreat_penalty_weight,
    goal_potential_scale,
):
    previous_potential = goal_potential(
        previous_ball_position,
        attacking_goal_x,
        goal_potential_scale,
    )
    current_potential = goal_potential(
        current_ball_position,
        attacking_goal_x,
        goal_potential_scale,
    )
    potential_delta = current_potential - previous_potential
    if potential_delta >= 0.0:
        return goal_progress_weight * potential_delta, 0.0, potential_delta
    return 0.0, retreat_penalty_weight * potential_delta, potential_delta


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
    changing weights without changing the wrapper logic. The current custom
    shaping is based on an exponential 2D ball-to-goal potential. Movement
    that increases goal proximity receives a bonus; movement that reduces it
    receives a stronger retreat penalty. The exponential potential makes
    shaping small in midfield and larger near scoring range.

    Important assumption:
    We treat increasing ball x as progress toward the attacking goal for the
    controlled team. That matches the current curriculum/setup in this repo,
    but should be re-checked if team orientation changes.
    """

    def __init__(
        self,
        env,
        shaping_mode: str,
        goal_progress_weight: float = 0.75,
        retreat_penalty_weight: float = 1.25,
        goal_potential_scale: float = 6.0,
        attacking_goal_x: float = 15.0,
        debug: bool = False,
    ):
        super().__init__(env)
        self.shaping_mode = shaping_mode
        self.goal_progress_weight = goal_progress_weight
        self.retreat_penalty_weight = retreat_penalty_weight
        self.goal_potential_scale = goal_potential_scale
        self.attacking_goal_x = attacking_goal_x
        self.debug = debug
        self._previous_ball_position = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # We shape from step-to-step ball movement, so each episode starts with
        # no previous position reference.
        self._previous_ball_position = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Start from the environment's original team reward so shaping remains
        # additive and easy to disable for baseline comparisons.
        shaped_reward = reward
        current_ball_position = extract_ball_position(info)
        previous_ball_position = self._previous_ball_position
        current_ball_x = (
            None if current_ball_position is None else current_ball_position[0]
        )
        current_ball_y = (
            None if current_ball_position is None else current_ball_position[1]
        )
        goal_progress_bonus = 0.0
        retreat_penalty = 0.0
        potential_delta = 0.0

        if current_ball_position is not None and previous_ball_position is not None:
            if self.shaping_mode == "custom":
                goal_progress_bonus, retreat_penalty, potential_delta = (
                    compute_goal_potential_bonus(
                        previous_ball_position=previous_ball_position,
                        current_ball_position=current_ball_position,
                        attacking_goal_x=self.attacking_goal_x,
                        goal_progress_weight=self.goal_progress_weight,
                        retreat_penalty_weight=self.retreat_penalty_weight,
                        goal_potential_scale=self.goal_potential_scale,
                    )
                )
                shaped_reward += goal_progress_bonus + retreat_penalty

                if self.debug and potential_delta != 0.0:
                    print(
                        "[reward_shaping_debug] "
                        f"wrapper={id(self)} "
                        f"prev_ball=({previous_ball_position[0]:.3f}, {previous_ball_position[1]:.3f}) "
                        f"ball=({current_ball_position[0]:.3f}, {current_ball_position[1]:.3f}) "
                        f"potential_delta={potential_delta:.4f} "
                        f"base={reward:.4f} "
                        f"goal_progress_bonus={goal_progress_bonus:.4f} "
                        f"retreat_penalty={retreat_penalty:.4f} "
                        f"shaped={shaped_reward:.4f}"
                    )

        # Update the reference after shaping so the next step uses the current
        # ball location as the new baseline.
        self._previous_ball_position = current_ball_position

        info = dict(info) if isinstance(info, dict) else {}
        # Expose the decomposition for debugging and later analysis. This makes
        # it easier to verify that shaping is active and to inspect whether the
        # coefficient magnitudes are reasonable.
        info["reward_shaping"] = make_reward_shaping_info(
            mode=self.shaping_mode,
            goal_progress_bonus=goal_progress_bonus,
            retreat_penalty=retreat_penalty,
            base_reward=reward,
            shaped_reward=shaped_reward,
            ball_x=current_ball_x,
            ball_y=current_ball_y,
            potential_delta=potential_delta,
        )

        if is_episode_done(done):
            # Avoid carrying ball state across episode boundaries.
            self._previous_ball_position = None

        return obs, shaped_reward, done, info


class MultiagentTeamRewardShapingWrapper(gym.core.Wrapper):
    """
    Symmetric reward shaping for multiagent-team self-play.

    `MultiagentTeamWrapper` exposes two RLlib agents:
    - team 0 controls players 0 and 1, and attacks toward +x
    - team 1 controls players 2 and 3, and attacks toward -x

    The single-team shaping wrapper cannot be reused here because each team has
    a different attacking goal. This wrapper applies the same 2D goal-potential
    shaping with mirrored goal centers for blue and orange. The exponential
    potential makes shaping small in midfield and larger near scoring range.

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
        goal_progress_weight: float = 0.75,
        retreat_penalty_weight: float = 1.25,
        goal_potential_scale: float = 6.0,
        goal_x: float = 15.0,
        debug: bool = False,
    ):
        super().__init__(env)
        self.goal_progress_weight = goal_progress_weight
        self.retreat_penalty_weight = retreat_penalty_weight
        self.goal_potential_scale = goal_potential_scale
        self.goal_x = goal_x
        self.debug = debug
        self._previous_ball_position = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._previous_ball_position = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        shaped_reward = dict(reward)
        current_ball_position = extract_ball_position(info)
        previous_ball_position = self._previous_ball_position
        current_ball_x = (
            None if current_ball_position is None else current_ball_position[0]
        )
        current_ball_y = (
            None if current_ball_position is None else current_ball_position[1]
        )
        shaping_by_team = {
            self.BLUE_TEAM_ID: make_reward_shaping_info(
                mode="multiagent_custom",
                goal_progress_bonus=0.0,
                retreat_penalty=0.0,
                base_reward=reward.get(self.BLUE_TEAM_ID, 0.0),
                shaped_reward=reward.get(self.BLUE_TEAM_ID, 0.0),
                ball_x=current_ball_x,
                ball_y=current_ball_y,
            ),
            self.ORANGE_TEAM_ID: make_reward_shaping_info(
                mode="multiagent_custom",
                goal_progress_bonus=0.0,
                retreat_penalty=0.0,
                base_reward=reward.get(self.ORANGE_TEAM_ID, 0.0),
                shaped_reward=reward.get(self.ORANGE_TEAM_ID, 0.0),
                ball_x=current_ball_x,
                ball_y=current_ball_y,
            ),
        }

        if current_ball_position is not None and previous_ball_position is not None:
            self._apply_team_shaping(
                shaped_reward,
                shaping_by_team,
                self.BLUE_TEAM_ID,
                previous_ball_position,
                current_ball_position,
                attacking_goal_x=self.goal_x,
            )
            self._apply_team_shaping(
                shaped_reward,
                shaping_by_team,
                self.ORANGE_TEAM_ID,
                previous_ball_position,
                current_ball_position,
                attacking_goal_x=-self.goal_x,
            )

            blue_shaping = shaping_by_team[self.BLUE_TEAM_ID]
            orange_shaping = shaping_by_team[self.ORANGE_TEAM_ID]
            if (
                self.debug
                and (
                    blue_shaping["potential_delta"] != 0.0
                    or orange_shaping["potential_delta"] != 0.0
                )
            ):
                print(
                    "[ma_reward_shaping_debug] "
                    f"wrapper={id(self)} "
                    f"prev_ball=({previous_ball_position[0]:.3f}, {previous_ball_position[1]:.3f}) "
                    f"ball=({current_ball_position[0]:.3f}, {current_ball_position[1]:.3f}) "
                    f"blue_base={reward.get(self.BLUE_TEAM_ID, 0.0):.4f} "
                    f"blue_goal_progress={blue_shaping['goal_progress_bonus']:.4f} "
                    f"blue_retreat={blue_shaping['retreat_penalty']:.4f} "
                    f"blue_shaped={shaped_reward.get(self.BLUE_TEAM_ID, 0.0):.4f} "
                    f"orange_base={reward.get(self.ORANGE_TEAM_ID, 0.0):.4f} "
                    f"orange_goal_progress={orange_shaping['goal_progress_bonus']:.4f} "
                    f"orange_retreat={orange_shaping['retreat_penalty']:.4f} "
                    f"orange_shaped={shaped_reward.get(self.ORANGE_TEAM_ID, 0.0):.4f}"
            )

        self._previous_ball_position = current_ball_position
        info = attach_team_reward_shaping_info(info, shaping_by_team)

        if is_episode_done(done):
            self._previous_ball_position = None

        return obs, shaped_reward, done, info

    def _apply_team_shaping(
        self,
        shaped_reward,
        shaping_by_team,
        team_id,
        previous_ball_position,
        current_ball_position,
        attacking_goal_x,
    ):
        goal_progress_bonus, retreat_penalty, potential_delta = compute_goal_potential_bonus(
            previous_ball_position=previous_ball_position,
            current_ball_position=current_ball_position,
            attacking_goal_x=attacking_goal_x,
            goal_progress_weight=self.goal_progress_weight,
            retreat_penalty_weight=self.retreat_penalty_weight,
            goal_potential_scale=self.goal_potential_scale,
        )
        shaped_reward[team_id] = (
            shaped_reward.get(team_id, 0.0) + goal_progress_bonus + retreat_penalty
        )
        shaping_by_team[team_id]["goal_progress_bonus"] = goal_progress_bonus
        shaping_by_team[team_id]["retreat_penalty"] = retreat_penalty
        shaping_by_team[team_id]["potential_delta"] = potential_delta
        shaping_by_team[team_id]["shaped_reward"] = shaped_reward[team_id]


class RewardShapingMetricsCallbacks(DefaultCallbacks):
    """
    Logs reward-shaping components into RLlib custom metrics so they appear in
    Tune results and TensorBoard.
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.user_data["reward_shaping_goal_progress_bonus_total"] = 0.0
        episode.user_data["reward_shaping_retreat_penalty_total"] = 0.0
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

            goal_progress_bonus = float(shaping.get("goal_progress_bonus", 0.0))
            retreat_penalty = float(shaping.get("retreat_penalty", 0.0))
            base_reward = float(shaping.get("base_reward", 0.0))
            shaped_reward = float(shaping.get("shaped_reward", base_reward))

            episode.user_data["reward_shaping_goal_progress_bonus_total"] += (
                goal_progress_bonus
            )
            episode.user_data["reward_shaping_retreat_penalty_total"] += retreat_penalty
            episode.user_data["reward_shaping_delta_total"] += shaped_reward - base_reward

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.custom_metrics["reward_shaping_goal_progress_bonus_total"] = episode.user_data[
            "reward_shaping_goal_progress_bonus_total"
        ]
        episode.custom_metrics["reward_shaping_retreat_penalty_total"] = episode.user_data[
            "reward_shaping_retreat_penalty_total"
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
                goal_progress_weight=env_config.get("goal_progress_weight", 0.75),
                retreat_penalty_weight=env_config.get(
                    "retreat_penalty_weight", 1.25
                ),
                goal_potential_scale=env_config.get(
                    "goal_potential_scale", 6.0
                ),
                debug=env_config.get("reward_shaping_debug", False),
            )
        elif env_type is soccer_twos.EnvType.multiagent_team:
            env = MultiagentTeamRewardShapingWrapper(
                env,
                goal_progress_weight=env_config.get("goal_progress_weight", 0.75),
                retreat_penalty_weight=env_config.get(
                    "retreat_penalty_weight", 1.25
                ),
                goal_potential_scale=env_config.get(
                    "goal_potential_scale", 6.0
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
