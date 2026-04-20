# Soccer-Twos Final Project Plan

## Goal

Build and submit multiple SoccerTwos agents that satisfy the final project rubric:

- Modify the reward function or observation space to improve learning.
- Run controlled experiments against a baseline.
- Submit trained agents that can beat the Random Agent and, ideally, the Baseline Agent.
- Include training plots, comparison plots, and a short report explaining what was changed and why it worked.

## Rubric Checklist

### Code Requirements (100 pts)

- Submission integrity
  - Implement an agent class that inherits from `soccer_twos.AgentInterface`.
  - Implement the `act()` method.
  - Fill in agent metadata in the agent `README.md`.
  - Package each submitted agent folder as a `.zip`.
  - Verify each unzipped agent loads without errors.
- Reward / observation / architecture modification
  - Add or change the reward function or observation space in source code.
  - Keep the code syntactically correct and logically consistent.
  - Optional bonus: add a novel learning idea such as curriculum learning or self-play.
- Policy performance
  - Train at least one agent that wins 9/10 against the Random Agent.
  - Train at least one agent that wins 9/10 against the Baseline Agent.
  - Optional bonus: beat the competitive agent when it is released.

### Report Requirements (100 pts)

- 1-2 pages excluding references.
- Explain the algorithm used, the library used, and the theory behind the method.
- Include the final hyperparameters for each important run.
- Include at least one training curve for every submitted or discussed agent.
- Include a direct comparison plot or table across agents.
- Label all figures clearly with axis names.
- State whether the modification improved reward, convergence speed, or performance.
- Give a technical explanation for why the observed results happened.
- Include references.

## Agent Strategy

We should plan to produce at least 3 named agents so the final submission is easy to map to rubric items:

- `Agent 1: Baseline PPO`
  - Minimal or no environment modification.
  - Used as the comparison point.
- `Agent 2: Reward / Observation Modified PPO`
  - Main required agent for the modification rubric.
  - Should be the strongest candidate for beating Random and Baseline.
- `Agent 3: Bonus or Advanced Agent`
  - Self-play, curriculum learning, or another novel idea.
  - Used for bonus credit or additional comparison in the report.

## Current Status - Apr 19

### Completed

- Implemented a clean PPO baseline training path in `example_ray_team_vs_random.py`.
- Implemented reward shaping in `utils.py` through `TeamRewardShapingWrapper`.
- Current shaping experiment uses:
  - `reward_shaping = "custom"`
  - ball progress shaping along x-axis
  - optional defensive-clear shaping controlled by a separate weight
- Added reward-shaping custom metrics through `RewardShapingMetricsCallbacks`.
- Added unshaped evaluation config to training scripts so shaped reward is not the only comparison signal.
- Trained multiple PPO variants:
  - baseline team-vs-random PPO
  - reward-shaped PPO with different progress / clear weights
- Packaged two RLlib checkpoint-backed agents:
  - `baseline_ppo_agent`
  - `reward_shaping_ppo_agent`
- Added `sweep_checkpoint_winrates.py` to evaluate all saved checkpoints against:
  - Random Agent
  - CEIA Baseline Agent
- Confirmed that final checkpoint selection should use external win rate, not shaped training reward.

### Key Lessons

- Shaped training reward is not comparable to baseline training reward.
- For final model selection, use direct win rate from `soccer_twos.evaluate`.
- The current reward shaping wrapper is valid for `team_vs_policy`, but not for `multiagent_team`.
- In `multiagent_team`, blue and orange attack opposite directions, so any progress-based reward must be symmetric:
  - blue progress: `+delta_x`
  - orange progress: `-delta_x`
- Current PPO variants were trained against the Random Agent, so poor performance against CEIA is not surprising.

### Current Risks

- Existing reward-shaped checkpoint in `reward_shaping_ppo_agent` was hand-selected from a late checkpoint, not necessarily the best external win-rate checkpoint.
- Existing reward shaping does not apply correctly to `multiagent_team`; do not use it there unchanged.

### Week of Apr 19 - Checkpoint Selection and Optional Self-Play

**Main objective:** choose the strongest existing checkpoint and decide whether one more training run is worth the time.

**TODOs**

- Summarize sweep results:
  - best baseline checkpoint vs Random
  - best reward-shaped checkpoint vs Random
  - best baseline checkpoint vs CEIA
  - best reward-shaped checkpoint vs CEIA
- Compare side-specific performance:
  - blue win rate
  - orange win rate
  - overall win rate
- Update `baseline_ppo_agent` and `reward_shaping_ppo_agent` to point to the selected checkpoints.
- Verify packaged agents with:
  - `python -m soccer_twos.watch -m1 baseline_ppo_agent -m2 example_player_agent`
  - `python -m soccer_twos.watch -m1 reward_shaping_ppo_agent -m2 example_player_agent`
  - `python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 example_player_agent -e 10`
  - `python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent -e 10`
- If current agents do not meet target performance, run one advanced experiment:
  - start with unshaped `multiagent_team` shared-policy PPO
  - only add symmetric reward shaping if the unshaped run is not competitive
- Write a short technical note explaining:
  - why random-opponent PPO may not transfer to CEIA
  - why shaped reward is not used as final checkpoint-selection metric
  - why multiagent-team self-play is the next logical experiment

**Deliverables by Apr 19**

- Best-checkpoint table from JSONL sweeps.
- Final selected checkpoint for each packaged agent.
- Decision on whether to run multiagent-team self-play.

### Week of Apr 23 - Finalization, Packaging, and Report

**Main objective:** lock down the final submission package and make sure every rubric item is covered.

**TODOs**

- Select the final submitted agents and map each one to a rubric purpose.
- Run final evaluations:
  - final agent vs Random Agent
  - final agent vs Baseline Agent
  - bonus agent vs strong opponent if available
- Verify the final packaged agents load correctly with `soccer_twos.watch` or evaluation tools.
- Fill in each agent `README.md` with:
  - agent name
  - authors
  - emails
  - short description
- Zip each agent folder exactly as required for submission.
- Finalize figures:
  - one training curve per submitted/discussed agent
  - one direct comparison figure or table
  - clearly labeled axes
- Write the final 1-2 page report with:
  - algorithm and library used
  - theory background
  - exact modification made
  - motivation / hypothesis
  - hyperparameter table
  - results and comparison
  - technical discussion of why the results happened
  - references
- Do one final rubric audit before submission.

**Deliverables by Apr 23**

- Final agent folders with `AgentInterface` implementations.
- Final zipped submission packages.
- Final evaluation numbers.
- Final report PDF or document draft ready for submission.

**Success criteria**

- Every submitted agent loads without errors.
- The report directly answers every report rubric category.
- The final package is organized enough that the TA can tell which agent satisfies which requirement.

## Submission Checklist

- `AgentInterface` subclass implemented: done for packaged PPO agents.
- `act()` implemented: done.
- Reward or observation modification visible in source code: done in `utils.py`.
- Training curves saved for every submitted or discussed agent: needs final plot export.
- Comparison plot or table created: needs JSONL summarization.
- Final evaluation vs Random completed: in progress.
- Final evaluation vs Baseline completed: in progress.
- Agent `README.md` files completed: needs author/email update.
- Agent folders zipped: not done.
- Report limited to 1-2 pages excluding references: not done.
- Report includes algorithm, theory, hyperparameters, modification, results, analysis, and references: not done.

## Useful Commands

```bash
# Training
python example_ray_team_vs_random.py
python exp_reward_shaping.py
python example_ray_ma_teams.py

# Evaluation
python -m soccer_twos.evaluate -m1 baseline_ppo_agent -m2 example_player_agent -e 10
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 example_player_agent -e 10
python -m soccer_twos.evaluate -m1 reward_shaping_ppo_agent -m2 ceia_baseline_agent -e 10

# Checkpoint sweep
python sweep_checkpoint_winrates.py --opponent random --episodes 10 --output random_eval.jsonl
python sweep_checkpoint_winrates.py --opponent ceia --episodes 10 --output ceia_eval.jsonl

# Visual inspection
python -m soccer_twos.watch -m1 reward_shaping_ppo_agent -m2 example_player_agent
```

## Important Files

- `utils.py` - environment helpers, reward shaping wrapper, custom metrics
- `example_ray_team_vs_random.py` - baseline PPO training against Random
- `exp_reward_shaping.py` - reward-shaped PPO training against Random
- `example_ray_ma_teams.py` - shared-policy multiagent-team training entry point
- `sweep_checkpoint_winrates.py` - checkpoint evaluation against Random / CEIA
- `baseline_ppo_agent/` - packaged baseline PPO candidate
- `reward_shaping_ppo_agent/` - packaged reward-shaped PPO candidate
- `curriculum.yaml` - curriculum configuration
- `ray_results/` - checkpoints, logs, and training outputs
- `random_eval.jsonl` / `ceia_eval.jsonl` - checkpoint sweep outputs
