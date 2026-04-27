# Self-Play PPO Agent

**Agent name:** SelfPlayPPO  
**Authors:** Tsung-Huan Yang  
**Emails:** tyang416@gatech.edu   
**Description:** PPO team agent trained with seeded historical-opponent self-play and exponential 2D goal-proximity reward shaping. Both players on a team share one policy that observes the concatenated per-player observations. This agent is intended for exploration, the agent for submission is TEAM19_AGENT

## Source experiment

Run: `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runI`  
Best checkpoint: iteration 3400, 77.5\% win rate against CEIA baseline  
Training script is in `jason_workspace` branch, `exp_seeded_historical_selfplay.py`, which includes reward modification, seeding checkpoints, and self-play
