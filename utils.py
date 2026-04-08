from random import uniform as randfloat
import fcntl
import os
import socket

import gym
import ray
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass

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
