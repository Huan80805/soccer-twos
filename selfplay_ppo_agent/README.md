# Self-Play PPO Agent

**Agent name:** SelfPlayPPO  
**Authors:** [Your names]  
**Emails:** [Your emails]  
**Description:** PPO team agent trained with historical-opponent self-play and progress reward shaping (ball_progress_weight=0.05). Both players on a team share one policy that observes the concatenated per-player observations.

## Source experiment

Run: `PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10`  
Best checkpoint: iteration 4400 (36 % win rate vs. CEIA baseline in 50-episode sweep)

## Checkpoint packaging

Before zipping this directory, copy two things from the Tune trial into `checkpoint/`:

```
ray_results/PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10/
  PPO_Soccer_83b04_00000_0_2026-04-21_14-04-18/
    params.json                          ← copy here
    checkpoint_004400/
      checkpoint-4400                    ← copy here
      checkpoint-4400.tune_metadata      ← copy here
```

The `checkpoint/` directory should end up containing:
- `checkpoint-4400`
- `checkpoint-4400.tune_metadata`
- `params.json`
