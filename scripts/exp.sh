
# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/selfplay_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrun0/PPO_Soccer_83b04_00000_0_2026-04-21_14-04-18/checkpoint_003200/checkpoint-3200 \
#   --lr 2.5e-5 \
#   --num-sgd-iter 10 \
#   --timesteps-total 20000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr2p5em05_sgd10_runD


# python exp_seeded_historical_selfplay.py \
#   --port 58000 \
#   --ceia-eval-path evaluation/selfplay_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrun0/PPO_Soccer_83b04_00000_0_2026-04-21_14-04-18/checkpoint_003200/checkpoint-3200 \
#   --lr 1e-4 \
#   --num-sgd-iter 6 \
#   --train-batch-size 8000 \
#   --sgd-batch-size 1024 \
#   --timesteps-total 20000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr1em04_sgd6_runE

# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/selfplay_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10_seedrun0/PPO_Soccer_83b04_00000_0_2026-04-21_14-04-18/checkpoint_003200/checkpoint-3200 \
#   --lr 5e-5 \
#   --num-sgd-iter 6 \
#   --sgd-batch-size 1024 \
#   --train-batch-size 8000 \
#   --timesteps-total 20000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runF

# round 3
# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/seedrun_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA/PPO_Soccer_37473_00000_0_2026-04-23_00-52-22/checkpoint_004600/checkpoint-4600 \
#   --lr 5e-5 \
#   --num-sgd-iter 6 \
#   --sgd-batch-size 1024 \
#   --train-batch-size 8000 \
#   --timesteps-total 40000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --max-seeds-total 6 \
#   --max-seeds-per-run 2 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG


# # round 3
# # used update interval 50 instead of 10
# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/seedrun_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA/PPO_Soccer_37473_00000_0_2026-04-23_00-52-22/checkpoint_004600/checkpoint-4600 \
#   --lr 5e-5 \
#   --num-sgd-iter 6 \
#   --sgd-batch-size 1024 \
#   --train-batch-size 8000 \
#   --timesteps-total 40000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --history-save-interval 50 \
#   --opponent-update-interval 50 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --max-seeds-total 6 \
#   --max-seeds-per-run 2 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG


# upweight goal progress and retreat penalty
# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/seedrun_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_seeded_historical_selfplay_reward_prog005_clear005_lr2em05_sgd10_runA/PPO_Soccer_37473_00000_0_2026-04-23_00-52-22/checkpoint_004600/checkpoint-4600 \
#   --lr 5e-5 \
#   --num-sgd-iter 6 \
#   --sgd-batch-size 1024 \
#   --train-batch-size 8000 \
#   --timesteps-total 40000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --history-save-interval 50 \
#   --opponent-update-interval 50 \
#   --reward-shaping custom \
#   --goal-progress-weight 2.0 \
#   --retreat-penalty-weight 2.5 \
#   --goal-potential-scale 6.0 \
#   --max-seeds-total 6 \
#   --max-seeds-per-run 2 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal200_retreat250_scale6_lr5em05_sgd6_runH


# round 4
# python exp_seeded_historical_selfplay.py \
#   --port 59000 \
#   --ceia-eval-path evaluation/seedrun_ceia_eval.jsonl \
#   --init-checkpoint ray_results/PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG/PPO_Soccer_5d4ab_00000_0_2026-04-25_13-37-00/checkpoint_004600/checkpoint-4600 \
#   --lr 5e-5 \
#   --num-sgd-iter 6 \
#   --sgd-batch-size 1024 \
#   --train-batch-size 8000 \
#   --timesteps-total 40000000 \
#   --seed-weighted-prob 0.30 \
#   --history-prob 0.40 \
#   --current-prob 0.20 \
#   --seed-uniform-prob 0.10 \
#   --history-save-interval 50 \
#   --opponent-update-interval 50 \
#   --reward-shaping custom \
#   --goal-progress-weight 0.75 \
#   --retreat-penalty-weight 1.25 \
#   --goal-potential-scale 6.0 \
#   --max-seeds-total 6 \
#   --max-seeds-per-run 3 \
#   --exp-name PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runI

python exp_seeded_historical_selfplay.py \
  --port 59000 \
  --ceia-eval-path evaluation/seedrun_ceia_eval.jsonl \
  --init-checkpoint ray_results/PPO_seeded_historical_selfplay_reward_goal075_retreat125_scale6_lr5em05_sgd6_runG/PPO_Soccer_5d4ab_00000_0_2026-04-25_13-37-00/checkpoint_004600/checkpoint-4600 \
  --lr 5e-5 \
  --num-sgd-iter 6 \
  --sgd-batch-size 1024 \
  --train-batch-size 8000 \
  --timesteps-total 40000000 \
  --seed-weighted-prob 0.30 \
  --history-prob 0.40 \
  --current-prob 0.20 \
  --seed-uniform-prob 0.10 \
  --history-save-interval 50 \
  --opponent-update-interval 50 \
  --reward-shaping custom \
  --goal-progress-weight 1.5 \
  --retreat-penalty-weight 2.0 \
  --goal-potential-scale 6.0 \
  --max-seeds-total 6 \
  --max-seeds-per-run 3 \
  --exp-name PPO_seeded_historical_selfplay_reward_goal150_retreat200_scale6_lr5em05_sgd6_runJ