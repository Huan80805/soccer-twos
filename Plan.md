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

## Current Status - Apr 22

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
- Implemented and ran historical-opponent self-play in `exp_historical_selfplay.py`:
  - fixed side randomization for Ray 1.4 by setting per-episode policy mappings in the callback
  - explicitly syncs frozen opponent weights to rollout workers after sampling historical opponents
  - ran unshaped, progress-reward, low-update-pressure, larger-model, and slower-opponent-update variants
- Swept historical self-play checkpoints against Random and CEIA.
- Cleaned `evaluation/ceia_eval.jsonl` (rerun trial with `--trial-dir` to avoid mixing trials that share a run name.)

### Immediate TODOs

- Implement a seeded historical self-play experiment as a separate script, likely `exp_seeded_historical_selfplay.py`, while keeping `exp_historical_selfplay.py` for generating additional seed opponents.
- Add a seed-opponent builder/helper that:
  - reads `evaluation/ceia_eval.jsonl` and optionally `evaluation/random_eval.jsonl`
  - ranks candidate checkpoints by CEIA win rate, with a side-balance penalty
  - reads each checkpoint's `params.json`
  - keeps only checkpoints compatible with the new opponent policy architecture
  - extracts compatible policy weights into `.pkl` seed-opponent files
- Use the seeded run to sample opponents from:
  - high-performing historical self-play checkpoints
  - high-performing shared self-play checkpoints
  - current-run historical checkpoints
  - the current policy itself
- Run a small number of additional PR / PR+lowLR historical runs if more seed diversity is needed, then select 2-5 strong/balanced checkpoints from each run.
- Re-sweep the cleaned `PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30` trial with `--trial-dir`.
- Sweep seeded historical self-play checkpoints against Random and CEIA, then update plots/tables.

### Observations
- CEIA baseline may be sourced from [here](https://github.com/eduagarcia/teampequi-rl-ceia-2021/blob/main/ppo_deepmind_selfplay_v4.py). It looks like self-play with opponent sampled randomly from *unbounded* history (50% current, 50% random past checkpoint with triangular probability distribution.)
- Team-vs-random reward shaping reached strong Random win rates but did not transfer well to CEIA.
- Shared-policy self-play improved over pure self-play baseline when reward shaping was enabled, but still remained weak against CEIA.
- Self-play TensorBoard losses are not reliable model-selection metrics.
- `vf_explained_var` looked healthy enough; rising `vf_loss` likely reflects higher return scale / variance rather than immediate critic failure.
- `cur_kl_coeff` rose early in self-play runs, suggesting PPO updates were too aggressive under default `lr=5e-5`, `num_sgd_iter=30`.
- Historical self-play conclusions:
  - unshaped historical self-play performed poorly and should not receive more compute
  - progress reward shaping was necessary for useful learning
  - low-update-pressure historical self-play (`lr=2e-5`, `num_sgd_iter=10`) produced the best CEIA result so far, about 36%, but the final checkpoint was not necessarily the best checkpoint
  - high-update-pressure progress-reward historical self-play reached strong Random performance in some checkpoints, but CEIA transfer remained weak
  - larger unshaped model and slower opponent update interval did not solve transfer
- Historical self-play training reward is hard to interpret:
  - `episode_reward_mean` and `episode_reward_max` are not reliable model-selection signals in symmetric multiagent self-play
  - external checkpoint sweeps against Random and CEIA are the main selection metrics
  - side balance matters; prefer checkpoints with similar blue/orange win rates when overall win rates tie
- The next promising direction is seeded historical self-play:
  - bootstrap the opponent pool with high-performing prior checkpoints
  - rank seed candidates using CEIA win rate because the TA confirmed this is allowed
  - keep CEIA as an evaluation/model-selection signal, not as a direct training opponent
  - mix seed opponents with current-run historical checkpoints and the current policy to avoid overfitting to a small fixed pool
- Repeated PR / PR+lowLR runs should produce different policies because current configs have `seed=None` and use stochastic initialization, rollouts, Unity dynamics, random side assignment, and opponent sampling.
- Self-play checkpoints can be seed opponents if they are architecture-compatible. Shared self-play uses policy id `default`; historical self-play uses `current_team`, so the seed loader must support both.

### Current Risks

- Current packaged agents may not point to the best external-win-rate checkpoints yet.
- CEIA win rate is still far below the 9/10 target.
- Seeded opponent pools can silently fail if checkpoint architectures are incompatible; filter by `model.fcnet_hiddens`, `model.vf_share_layers`, observation space, and action space before loading weights.
- A small seed pool can overfit; keep some probability on current/current-run history and uniform seed sampling rather than always sampling only the top CEIA checkpoint.
- Checkpoint folders are ignored by git; final agent zips must include the required checkpoint files separately.
- Old Ray/RLlib requires an older NumPy; do not install plotting dependencies into the `soccertwos` conda env.

## TODOs And Deliverables

### Training And Experiments

- Keep `exp_historical_selfplay.py` as the clean baseline historical-opponent implementation and use it to generate more seed checkpoints if needed.
- Implement `exp_seeded_historical_selfplay.py`:
  - two policies: `current_team`, `opponent_team`
  - train only `current_team`
  - randomize current/opponent side assignment
  - periodically save current policy weights into the current-run history pool
  - load seed opponents from extracted `.pkl` weight files
  - sample from seed opponents with score-weighted probability
  - mix in current-run historical opponents and current-policy mirror matches
- Initial seeded sampling mix:
  - 40% score-weighted seed pool
  - 30% current-run historical pool
  - 20% current policy
  - 10% uniform compatible seed pool
- Initial seeded hparams:
  - reward shaping: progress-only, `ball_progress_weight=0.05`, `defensive_clear_weight=0.0`
  - `lr=2e-5` or `2.5e-5`
  - `num_sgd_iter=10`
  - `train_batch_size=4000` initially; consider 8000 if updates remain noisy
  - keep model architecture compatible with most high-value seeds, likely `[256, 256]` with `vf_share_layers=True`
- Seed pool candidates:
  - historical PR+lowLR checkpoints with high CEIA win rate and balanced sides
  - shaped shared self-play checkpoints with high CEIA or Random win rate
  - historical PR checkpoints with high Random win rate
  - optionally team-vs-random shaped checkpoints if architecture-compatible and useful for diversity

### Evaluation

- Re-run checkpoint sweeps for any new experiments:
  - against Random
  - against CEIA
- For run names with multiple trials, use `--trial-dir` in `evaluation/sweep_checkpoint_winrates.py` so results do not mix stale and new trials.
- Generate plots for:
  - team-vs-random PPO variants
  - self-play variants
  - tuned self-play variants if available
  - historical-opponent self-play
  - seeded historical-opponent self-play
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
python exp_historical_selfplay.py --port 58000

# Evaluation
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 example_player_agent -e 50
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent -e 50

# Checkpoint sweeps
python evaluation/sweep_checkpoint_winrates.py --opponent random --episodes 50 --output random_eval.jsonl
python evaluation/sweep_checkpoint_winrates.py --opponent ceia --episodes 50 --output ceia_eval.jsonl
python evaluation/sweep_checkpoint_winrates.py --opponent ceia --episodes 50 --output evaluation/ceia_eval.jsonl --run-name PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30 --trial-dir /path/to/specific/PPO_Soccer_trial

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
- `exp_historical_selfplay.py` - historical-opponent self-play training entrypoint
- `exp_seeded_historical_selfplay.py` - planned seeded historical-opponent self-play entrypoint
- `evaluation/sweep_checkpoint_winrates.py` - checkpoint evaluation against Random / CEIA
- `evaluation/plot_selfplay_winrates.py` - plotting for checkpoint win-rate sweeps
- `evaluation/random_eval.jsonl` / `evaluation/ceia_eval.jsonl` - checkpoint sweep outputs
- `evaluation/plots/` - generated comparison plots and best-checkpoint CSVs
- `baseline_ppo_agent/` - packaged baseline PPO candidate
- `reward_shaping_ppo_agent/` - packaged reward-shaped PPO candidate
- `ray_results/` - local checkpoints, logs, and training outputs
