"""Compatibility wrapper for the unified reward-shaping trainer.

Prefer:
    python exp_reward_shaping.py --mode selfplay --ball-progress-weight 0.05 --defensive-clear-weight 0.1 --lr 2.5e-5 --num-sgd-iter 10 --sgd-batch-size 128 --port 50000 --exp-name PPO_selfplay_reward_prog005_clear01_lr25e6_sgd10
"""

from exp_reward_shaping import main


if __name__ == "__main__":
    import sys

    legacy_defaults = [
        "--mode",
        "selfplay",
        "--ball-progress-weight",
        "0.05",
        "--defensive-clear-weight",
        "0.1",
        "--lr",
        "2.5e-5",
        "--num-sgd-iter",
        "10",
        "--sgd-batch-size",
        "128",
        "--port",
        "50000",
        "--exp-name",
        "PPO_selfplay_reward_prog005_clear01_lr25e6_sgd10",
    ]
    sys.argv = [sys.argv[0]] + legacy_defaults + sys.argv[1:]
    main(default_mode="selfplay")
