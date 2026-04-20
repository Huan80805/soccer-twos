# Soccer-Twos Final Project Plan

## Goal

Build and submit Soccer-Twos agents that satisfy the final project rubric:

- Modify the reward function or observation space to improve learning.
- Run controlled experiments against a baseline.
- Submit at least one trained agent that wins 9/10 against the Random Agent, and ideally one that also beats the Baseline Agent.
- Include training curves, win-rate comparisons, and a short report explaining what changed and why.

## Rubric Checklist

### Code Requirements

- Implement an agent class that inherits from `soccer_twos.AgentInterface`.
- Implement the `act()` method.
- Fill in agent metadata in each agent `README.md`.
- Package each submitted agent folder as a `.zip`.
- Verify each unzipped agent loads without errors.
- Include a visible reward, observation, architecture, curriculum, or self-play modification in source code.

### Report Requirements

- 1-2 pages excluding references.
- Explain PPO, RLlib, and the training setup.
- Include final hyperparameters for important runs.
- Include one training curve for each submitted or discussed agent.
- Include a direct comparison plot or table across agents.
- Label all figures clearly.
- State whether each modification improved learning speed, reward, or external win rate.
- Explain the observed results technically.
- Include references.

## Agent Strategy

- `Agent 1: Baseline PPO`
  - PPO team agent trained against Random.
  - Main comparison point.
- `Agent 2: Reward-Shaped PPO`
  - PPO team agent with ball-progress / defensive-clear reward shaping.
  - Satisfies the required environment modification.
- `Agent 3: Advanced Self-Play PPO`
  - Self-play or historical-opponent self-play.
  - Used for bonus credit and robustness comparison.

## Current Status - Apr 20

### Completed

- Implemented baseline PPO training against Random in `example_ray_team_vs_random.py`.
- Implemented reward shaping in `utils.py`:
  - `TeamRewardShapingWrapper` for `EnvType.team_vs_policy`
  - `MultiagentTeamRewardShapingWrapper` for symmetric `EnvType.multiagent_team`
  - `RewardShapingMetricsCallbacks` for TensorBoard custom metrics
- Trained and swept multiple team-vs-random variants:
  - baseline PPO
  - ball-progress reward shaping
  - ball-progress plus defensive-clear shaping
- Packaged checkpoint-backed agents:
  - `baseline_ppo_agent`
  - `reward_shaping_ppo_agent`
- Added checkpoint sweeping under `evaluation/sweep_checkpoint_winrates.py`.
- Added plotting under `evaluation/plot_selfplay_winrates.py`.
- Moved evaluation artifacts under `evaluation/`.
- plotting should use `~/.venv/hml/bin/python`, not the Soccer-Twos RL env, to avoid upgrading NumPy inside the old Ray/RLlib environment.

### Observations

- Team-vs-random reward shaping reached strong Random win rates but did not transfer well to CEIA.
- Shared-policy self-play improved over pure self-play baseline when reward shaping was enabled, but still remained weak against CEIA.
- Self-play TensorBoard losses are not reliable model-selection metrics.
- `vf_explained_var` looked healthy enough; rising `vf_loss` likely reflects higher return scale / variance rather than immediate critic failure.
- `cur_kl_coeff` rose early in self-play runs, suggesting PPO updates were too aggressive under default `lr=5e-5`, `num_sgd_iter=30`.
- The next promising direction is historical-opponent self-play:
  - train `current_team`
  - freeze and sample older checkpoints into `opponent_team`
  - randomize whether the current policy plays blue or orange
  - optionally keep reward shaping enabled

### Current Risks

- Current packaged agents may not point to the best external-win-rate checkpoints yet.
- CEIA win rate is still far below the 9/10 target.
- Checkpoint folders are ignored by git; final agent zips must include the required checkpoint files separately.
- Old Ray/RLlib requires an older NumPy; do not install plotting dependencies into the `soccertwos` conda env.

## TODOs And Deliverables

### Training And Experiments

- Run low-update-pressure self-play experiments:
  - `lr=2.5e-5`
  - `num_sgd_iter=10` or `5`
  - keep `train_batch_size=4000` initially to isolate update-pressure effects
- Implement historical-opponent self-play as the next major experiment:
  - two policies: `current_team`, `opponent_team`
  - train only `current_team`
  - periodically save `current_team` weights
  - sample frozen historical opponents into `opponent_team`
  - randomize current/opponent side assignment
- Decide whether to run historical self-play with:
  - no shaping
  - progress-only shaping
  - progress plus defensive-clear shaping
- Optional if time allows:
  - test `vf_share_layers=False`
  - test a medium model such as `[512, 256, 128]`
  - test lower shaping weights if CEIA transfer remains poor

### Evaluation

- Re-run checkpoint sweeps for any new experiments:
  - against Random
  - against CEIA
- Generate plots for:
  - team-vs-random PPO variants
  - self-play variants
  - tuned self-play variants if available
  - historical-opponent self-play if implemented
- Select checkpoints by external win rate:
  - primary: Random 9/10 requirement
  - secondary: CEIA/Baseline robustness
  - tertiary: side balance between blue and orange
- Update packaged agents to point to selected checkpoints.

### Packaging

- Verify packaged agents with `soccer_twos.evaluate`.
- Fill in each agent `README.md` with:
  - agent name
  - authors
  - emails
  - short description
  - checkpoint/source experiment
- Zip final agent folders.
- Confirm zipped agents load after extraction.

### Report

- Add training-curve figures from TensorBoard.
- Add external win-rate comparison tables/plots from JSONL sweeps.
- Explain reward shaping:
  - ball progress
  - defensive clear
  - symmetric self-play shaping
- Explain why shaped reward is not used to select final checkpoints.
- Explain why Random-trained policies transfer poorly to CEIA.
- If implemented, explain historical-opponent self-play as the advanced method.

### Deliverables

- Final selected checkpoint table.
- Final Random and CEIA evaluation numbers.
- Final agent folders and zips.
- Final 1-2 page report.
- Clean git commit with code, evaluation scripts, plots/tables, and plan/report materials.

## Useful Commands

```bash
# RL training: use the Soccer-Twos conda env
python example_ray_team_vs_random.py
python exp_reward_shaping.py --mode team_vs_random --port 55000
python exp_reward_shaping.py --mode selfplay --port 50000
python example_ray_ma_teams.py

# Evaluation
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 example_player_agent -e 50
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent -e 50

# Checkpoint sweeps
python evaluation/sweep_checkpoint_winrates.py --opponent random --episodes 50 --output random_eval.jsonl
python evaluation/sweep_checkpoint_winrates.py --opponent ceia --episodes 50 --output ceia_eval.jsonl

# Plotting: use the plotting venv, not the RL env
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite selfplay
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite team
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite selfplay_tuned

# Visual inspection
/nethome/tyang416/miniconda3/envs/soccertwos/bin/python -m soccer_twos.watch -m1 reward_shaping_ppo_agent -m2 example_player_agent
```

## Important Files

- `utils.py` - environment helpers, reward shaping wrappers, custom metrics
- `example_ray_team_vs_random.py` - baseline PPO training against Random
- `exp_reward_shaping.py` - unified reward-shaped PPO training entrypoint
- `exp_selfplay_reward_shaping.py` - compatibility wrapper for self-play reward shaping
- `example_ray_ma_teams.py` - shared-policy multiagent-team baseline self-play
- `evaluation/sweep_checkpoint_winrates.py` - checkpoint evaluation against Random / CEIA
- `evaluation/plot_selfplay_winrates.py` - plotting for checkpoint win-rate sweeps
- `evaluation/random_eval.jsonl` / `evaluation/ceia_eval.jsonl` - checkpoint sweep outputs
- `evaluation/plots/` - generated comparison plots and best-checkpoint CSVs
- `baseline_ppo_agent/` - packaged baseline PPO candidate
- `reward_shaping_ppo_agent/` - packaged reward-shaped PPO candidate
- `ray_results/` - local checkpoints, logs, and training outputs
