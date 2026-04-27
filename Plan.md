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
  - PPO team agent with exponential 2D ball-to-goal potential reward shaping.
  - Satisfies the required environment modification.
- `Agent 3: Advanced Self-Play PPO`
  - Self-play or historical-opponent self-play.
  - Used for bonus credit and robustness comparison.

## Logs

### Evaluation Artifacts

- All CEIA/Baseline checkpoint sweeps for the discussed runs are complete.
- Random-agent sweeps are complete for team-vs-random and shared self-play runs. Random-agent sweeps are intentionally incomplete for later historical / seeded historical runs because those runs were selected by CEIA robustness and several weak intermediate runs were not worth extra evaluation.
- Main result files:
  - team-vs-random: `evaluation/random_random_eval.jsonl`, `evaluation/random_ceia_eval.jsonl`
  - shared self-play: `evaluation/selfplay_random_eval.jsonl`, `evaluation/selfplay_ceia_eval.jsonl`
  - historical self-play: `evaluation/historical_selfplay_ceia_eval.jsonl`
  - seeded historical self-play: `evaluation/seedrun_ceia_eval.jsonl`
- Main figures / tables:
  - `evaluation/plots/team_winrate_over_time.png`, `evaluation/plots/team_side_winrate_over_time.png`, `evaluation/plots/team_best_checkpoints.csv`
  - `evaluation/plots/selfplay_winrate_over_time.png`, `evaluation/plots/selfplay_side_winrate_over_time.png`, `evaluation/plots/selfplay_best_checkpoints.csv`
  - `evaluation/plots/historical_winrate_over_time.png`, `evaluation/plots/historical_side_winrate_over_time.png`, `evaluation/plots/historical_best_checkpoints.csv`
  - `evaluation/plots/seeded_historical_winrate_over_time.png`, `evaluation/plots/seeded_historical_side_winrate_over_time.png`, `evaluation/plots/seeded_historical_best_checkpoints.csv`
- Plotting should use `~/.venv/hml/bin/python`, not the Soccer-Twos RL env, to avoid upgrading NumPy inside the old Ray/RLlib environment.

### Shared Implementation Notes

- Reward shaping is implemented in `utils.py`:
  - `TeamRewardShapingWrapper` for `EnvType.team_vs_policy`
  - `MultiagentTeamRewardShapingWrapper` for symmetric `EnvType.multiagent_team`
  - `RewardShapingMetricsCallbacks` for TensorBoard custom metrics
- Earlier reward shaping variants used:
  - 1D x-axis progress reward: positive reward when the ball moved toward the opponent goal along the x axis
  - optional defensive clear reward: reward for moving / keeping the ball away from the controlled team's defensive half
- The current reward-shaping implementation uses exponential 2D goal-proximity potential:
  - potential: `exp(-distance_to_goal / goal_potential_scale)`
  - default final setting: `goal_progress_weight=0.75`, `retreat_penalty_weight=1.25`, `goal_potential_scale=6.0`
  - stronger final-third shaping: midfield progress is small, movement near the goal mouth is larger
  - asymmetric retreat penalty makes back-and-forth ball movement net negative
- External checkpoint sweeps are the main selection metric. Training reward, `episode_reward_mean`, `episode_reward_max`, PPO losses, and value loss were useful diagnostics but were not reliable checkpoint selectors in self-play.
- CEIA/Baseline is used only for evaluation and checkpoint selection. It is not used as a direct training opponent.

### Attempt 1: PPO Trained Against Random

Rationale: establish a working PPO baseline and test whether simple reward shaping improves sample efficiency and Random-agent win rate.

Training setup:

- Script: `example_ray_team_vs_random.py` for baseline, `exp_reward_shaping.py --mode team_vs_random` for shaped variants.
- Checkpoint families: `PPO_vs_random_*`.
- Main plot: `evaluation/plots/team_winrate_over_time.png`.
- Result files: `evaluation/random_random_eval.jsonl`, `evaluation/random_ceia_eval.jsonl`.

Best checkpoint results:

| run | shaping | Random best | CEIA best | key checkpoint |
|---|---|---:|---:|---|
| `PPO_vs_random_baseline` | none | 0.875 | 0.175 | Random ckpt 1200; CEIA ckpt 1100 |
| `PPO_vs_random_reward_exp_prog005_clear0` | x-progress 0.05 | 0.925 | 0.175 | ckpt 1000 |
| `PPO_vs_random_reward_exp_prog005_clear01` | x-progress 0.05 + clear 0.1 | 0.675 | 0.250 | ckpt 1100 |
| `PPO_vs_random_reward_exp_prog01_clear00` | x-progress 0.1 | 0.600 | 0.100 | Random ckpt 800; CEIA ckpt 500 |
| `PPO_vs_random_reward_exp_prog01_clear01` | x-progress 0.1 + clear 0.1 | 1.000 | 0.200 | Random ckpt 1100; CEIA ckpt 1000 |

Conclusion:

- Reward shaping helped satisfy the Random-agent requirement. The best Random-agent result was `PPO_vs_random_reward_exp_prog01_clear01` with 40/40 wins.
- Random-trained policies transferred poorly to CEIA/Baseline. The best CEIA result in this family was only 0.25, from `PPO_vs_random_reward_exp_prog005_clear01`.
- The likely reason is opponent overfitting: a policy trained only against Random can learn direct ball-chasing and simple scoring behaviors without learning robust defense, side balance, or adversarial recovery.

### Attempt 2: Shared-Policy Self-Play

Rationale: improve robustness by training against a learning opponent rather than only Random.

Training setup:

- Script family: shared-policy multiagent-team self-play.
- Checkpoint families: `PPO_selfplay_*`.
- Main plot: `evaluation/plots/selfplay_winrate_over_time.png`.
- Result files: `evaluation/selfplay_random_eval.jsonl`, `evaluation/selfplay_ceia_eval.jsonl`.

Best checkpoint results:

| run | shaping / hparams | Random best | CEIA best | key checkpoint |
|---|---|---:|---:|---|
| `PPO_selfplay_baseline_teams` | none | 0.650 | 0.100 | ckpt 400 |
| `PPO_selfplay_reward_prog005_clear0` | x-progress 0.05 | 0.950 | 0.175 | Random ckpt 300; CEIA ckpt 500 |
| `PPO_selfplay_reward_prog005_clear01` | x-progress 0.05 + clear 0.1 | 0.900 | 0.300 | ckpt 600 |
| `PPO_selfplay_reward_prog005_clear01_lr2p5em05_sgd10_mb128` | lower LR / lower update pressure | 0.600 | 0.225 | Random ckpt 500; CEIA ckpt 300 |

Conclusion:

- Shared self-play plus reward shaping improved over pure self-play and could still beat Random well.
- CEIA transfer improved compared with team-vs-random in the best case, but remained weak. The best CEIA result was 0.30.
- The unstable moving target in shared self-play likely made optimization noisy. TensorBoard PPO metrics did not cleanly identify the best external checkpoint.
- Lower update pressure alone did not fix the robustness gap.

### Attempt 3: Historical-Opponent Self-Play

Rationale: reduce self-play nonstationarity by training `current_team` against a frozen `opponent_team` sampled from previous checkpoints in the same run.

Implementation details:

- Script: `exp_historical_selfplay.py`.
- Checkpoint families: `PPO_historical_selfplay_*`.
- Main plot: `evaluation/plots/historical_winrate_over_time.png`.
- Result file: `evaluation/historical_selfplay_ceia_eval.jsonl`.
- Two policies are registered: `current_team` and `opponent_team`; only `current_team` is trained.
- Side assignment is randomized each episode so `current_team` trains as both blue and orange.
- Ray 1.4 can call `policy_mapping_fn(agent_id)` without episode context, so callbacks pre-fill `episode._agent_to_policy` on episode start to prevent vectorized workers from interleaving side assignment incorrectly.
- Opponent weights are periodically loaded from current-run history; `trainer.workers.sync_weights()` is required because `opponent_team` is frozen and otherwise remote rollout workers can keep stale weights.

Best CEIA checkpoint results:

| run | main change | CEIA best | side split | key checkpoint |
|---|---|---:|---|---|
| `PPO_historical_selfplay_baseline_lr5em05_sgd30` | unshaped | 0.025 | 0.05 / 0.00 | ckpt 1200 |
| `PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30` | x-progress, high update pressure | 0.150 | 0.25 / 0.05 | ckpt 1200 |
| `PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30_updateInterval50` | slower opponent update | 0.100 | 0.15 / 0.05 | ckpt 600 |
| `PPO_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_seedrunA` | progress + clear, lower LR | 0.150 | 0.15 / 0.15 | ckpt 1200 |
| `PPO_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_seedrunC_ocp03` | progress + clear + current-opponent probability | 0.225 | 0.35 / 0.10 | ckpt 2200 |
| `PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrun0` | progress, lower update pressure | 0.375 | 0.35 / 0.40 | ckpt 4600 |
| `PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrunD_ocp03` | progress + current-opponent probability | 0.225 | 0.20 / 0.25 | ckpt 4600 |

Conclusion:

- Unshaped historical self-play failed and should not receive more compute.
- Progress reward shaping was necessary for useful learning.
- Lower update pressure helped more than slower opponent updates by itself. The best historical-only CEIA result was 0.375 from `seedrun0`.
- Historical self-play still underperformed because it starts from scratch and, early in training, the history pool mostly contains weak versions of itself. Later, past checkpoints may become too easy and provide limited new learning signal.

### Attempt 4: Seeded Historical Self-Play

Rationale: preserve the stability benefits of historical-opponent self-play while avoiding the cold-start problem by bootstrapping the opponent pool with stronger prior checkpoints.

Implementation details:

- Script: `exp_seeded_historical_selfplay.py`.
- Main plot: `evaluation/plots/seeded_historical_winrate_over_time.png`.
- Result file: `evaluation/seedrun_ceia_eval.jsonl`.
- Seed candidates are selected from CEIA sweep records and filtered for architecture compatibility.
- Seed extraction supports checkpoints with policy id `current_team`, `default`, or a single restored policy, so both historical and shared self-play checkpoints can be used as seed opponents.
- Opponent source mixture:
  - score-weighted seed opponent
  - current-run historical checkpoint
  - current policy mirror match
  - uniform seed opponent
- Final mixture for runs D-H: 30% score-weighted seed, 40% current-run history, 20% current policy, 10% uniform seed.
- Seeded side assignment uses the same callback-side Ray 1.4 workaround as historical self-play.
- Final shaping used exponential 2D goal-proximity potential instead of the older flat x-progress shaping.

Round summary and best CEIA checkpoint results:

| run | main change | CEIA best | side split | key checkpoint |
|---|---|---:|---|---|
| `PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA` | first seeded run, old x-progress + clear shaping | 0.500 | 0.60 / 0.40 | ckpt 4600 |
| `PPO_seeded_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_runC` | second round, old shaping | 0.300 | 0.35 / 0.25 | ckpt 2400 |
| `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr2p5em05_sgd10_runD` | new exponential shaping, conservative PPO update | 0.375 | 0.35 / 0.40 | ckpt 3600 |
| `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr1em04_sgd6_runE` | high LR / larger-batch PPO update | 0.350 | 0.40 / 0.30 | ckpt 2200 |
| `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runF` | mid LR / larger batch | 0.400 | 0.50 / 0.30 | ckpt 2400 |
| `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG` | round-3 seeds, warm-start from runA@4600, opponent update interval 50 | 0.775 | 0.75 / 0.80 | ckpt 4600 |
| `PPO_seeded_historical_selfplay_reward_goal200_retreat250_scale6_lr5em05_sgd6_runH` | runG setup with stronger shaping weights | 0.750 | 0.80 / 0.70 | ckpt 4600 |

RunG / runH attribution:

- The major performance jump was likely caused by the combination of:
  - warm-starting from `runA` checkpoint 4600 instead of a weaker historical-only checkpoint
  - using a cleaner round-3 seed pool built from newer seeded/exponential-shaping policies
  - slowing history save and opponent update intervals from 10 to 50, which reduced opponent-distribution churn
  - using lower PPO update pressure: `lr=5e-5`, `train_batch_size=8000`, `sgd_minibatch_size=1024`, `num_sgd_iter=6`
  - using exponential goal-proximity shaping
- PPO update changes alone were not sufficient: runF used the mid-LR larger-batch PPO settings but only reached 0.400.
- Stronger reward shaping in runH accelerated learning and produced a strong best checkpoint, but it did not clearly beat runG and ended less stably. RunG is the cleaner model-selection candidate because it had the best CEIA score and balanced side performance.

### Consolidated Observations For Report

- Random-agent win rate is not a reliable proxy for CEIA/Baseline robustness. Several models exceeded 0.90 Random win rate but stayed below 0.30 CEIA.
- Reward shaping improves exploration and scoring, but overly local shaping can overfit to simple ball-pushing behavior.
- The older 1D x-progress reward was too flat:
  - it gave the same reward for midfield progress and near-goal progress
  - it ignored lateral alignment with the goal mouth
  - it did not strongly penalize losing attacking progress
- Exponential 2D goal-proximity shaping is more defensible for the report:
  - it rewards movement toward the actual opponent goal center, not just positive x movement
  - it naturally increases shaping magnitude near the goal
  - the asymmetric retreat penalty makes exploitative back-and-forth movement net negative
- Self-play training curves are harder to interpret than external win-rate sweeps. In symmetric self-play, `episode_reward_mean` can plateau or oscillate even when external win rate changes.
- Side balance matters. A high overall score with one side much weaker is risky for final evaluation because submitted agents can be assigned either color.
- The strongest method was seeded historical self-play, not direct CEIA training. CEIA was used only to rank and select checkpoints after training, which the TA confirmed is allowed.
- Repeated runs with the same nominal hyperparameters can produce different policies because configs use stochastic initialization, rollout sampling, Unity dynamics, random side assignment, and opponent sampling.

### Checkpoints To Keep For Report / Packaging

Keep these at minimum until the report and final packaged agents are done:

| purpose | run | checkpoint | why keep |
|---|---|---:|---|
| Random baseline comparison | `PPO_vs_random_baseline` | 1200 | baseline best Random result: 0.875 |
| Reward-shaped Random agent | `PPO_vs_random_reward_exp_prog01_clear01` | 1100 | best Random result: 1.000 |
| Best Random-trained CEIA comparison | `PPO_vs_random_reward_exp_prog005_clear01` | 1100 | best CEIA among team-vs-random: 0.250 |
| Best shared self-play comparison | `PPO_selfplay_reward_prog005_clear01` | 600 | best CEIA shared self-play: 0.300; Random 0.900 |
| Best historical-only comparison | `PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrun0` | 4600 | best historical-only CEIA: 0.375 |
| First seeded milestone | `PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA` | 4600 | best first seeded CEIA: 0.500; init for round 3 |
| Final candidate | `PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG` | 4600 | best CEIA result: 0.775, balanced 0.75 / 0.80 |
| Final ablation candidate | `PPO_seeded_historical_selfplay_reward_goal200_retreat250_scale6_lr5em05_sgd6_runH` | 4600 | stronger-shaping ablation: 0.750 |

Do not delete the corresponding `params.json` / `params.pkl` files inside each trial directory, because restore and packaging need the model config as well as the checkpoint files.

### Current Risks

- RunG is strong against CEIA but still below the 0.90 goal. Continue evaluating whether the newer round-4 runs improve over runG before locking the final package.
- Later seeded runs have not all been evaluated against Random. If a final candidate is selected from seeded historical self-play, run a final Random sweep or direct `soccer_twos.evaluate` before submission.
- A small seed pool can overfit; keep some probability on current/current-run history and uniform seed sampling rather than always sampling only the top CEIA checkpoint.

## TODOs And Deliverables

### Training And Experiments

- Keep `exp_historical_selfplay.py` as the clean baseline historical-opponent implementation and use it to generate more seed checkpoints if needed.
- Run `exp_seeded_historical_selfplay.py`:
  - two policies: `current_team`, `opponent_team`
  - train only `current_team`
  - randomize current/opponent side assignment
  - periodically save current policy weights into the current-run history pool
  - load seed opponents from extracted `.pkl` weight files
  - sample from seed opponents with score-weighted probability
  - mix in current-run historical opponents and current-policy mirror matches
- Current best seeded sampling mix:
  - 30% score-weighted seed pool
  - 40% current-run historical pool
  - 20% current policy
  - 10% uniform compatible seed pool
- Current best seeded hparams:
  - reward shaping: exponential 2D goal-proximity shaping, `goal_progress_weight=0.75`, `retreat_penalty_weight=1.25`, `goal_potential_scale=6.0`
  - `lr=5e-5`
  - `num_sgd_iter=6`
  - `train_batch_size=8000`
  - `sgd_minibatch_size=1024`
  - `history_save_interval=50`, `opponent_update_interval=50`
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
  - 2D ball-to-goal potential
  - asymmetric retreat penalty
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
# Evaluation
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 example_player_agent -e 50
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent -e 50

# Checkpoint sweeps
python evaluation/sweep_checkpoint_winrates.py --opponent random --episodes 40 --output evaluation/random_random_eval.jsonl
python evaluation/sweep_checkpoint_winrates.py --opponent ceia --episodes 40 --output evaluation/random_ceia_eval.jsonl
python evaluation/sweep_checkpoint_winrates.py --opponent ceia --episodes 40 --output evaluation/seedrun_ceia_eval.jsonl --run-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG

# Plotting: use the plotting venv, not the RL env
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite selfplay
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite team
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite historical
~/.venv/hml/bin/python evaluation/plot_selfplay_winrates.py --suite seeded_historical

# Visual inspection
python -m soccer_twos.watch -m1 reward_shaping_ppo_agent -m2 example_player_agent
```

## Important Files

- `utils.py` - environment helpers, reward shaping wrappers, custom metrics
- `example_ray_team_vs_random.py` - baseline PPO training against Random
- `exp_reward_shaping.py` - unified reward-shaped PPO training entrypoint
- `exp_selfplay_reward_shaping.py` - compatibility wrapper for self-play reward shaping
- `example_ray_ma_teams.py` - shared-policy multiagent-team baseline self-play
- `exp_historical_selfplay.py` - historical-opponent self-play training entrypoint
- `exp_seeded_historical_selfplay.py` - seeded historical-opponent self-play entrypoint
- `evaluation/sweep_checkpoint_winrates.py` - checkpoint evaluation against Random / CEIA
- `evaluation/plot_selfplay_winrates.py` - plotting for checkpoint win-rate sweeps
- `evaluation/random_random_eval.jsonl` / `evaluation/random_ceia_eval.jsonl` - team-vs-random checkpoint sweep outputs
- `evaluation/selfplay_random_eval.jsonl` / `evaluation/selfplay_ceia_eval.jsonl` - shared self-play checkpoint sweep outputs
- `evaluation/historical_selfplay_ceia_eval.jsonl` - historical self-play CEIA sweep output
- `evaluation/seedrun_ceia_eval.jsonl` - seeded historical self-play CEIA sweep output
- `evaluation/plots/` - generated comparison plots and best-checkpoint CSVs
- `ray_results/` - local checkpoints, logs, and training outputs
