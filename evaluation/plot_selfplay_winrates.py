import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_RANDOM_RESULTS = EVALUATION_DIR / "random_eval.jsonl"
DEFAULT_CEIA_RESULTS = EVALUATION_DIR / "ceia_eval.jsonl"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "plots"

PLOT_SUITES = {
    "selfplay": {
        "title": "Self-play PPO checkpoint win rate",
        "side_title": "Self-play PPO side-specific win rate",
        "labels": {
            "PPO_selfplay_baseline_teams": "Self-play baseline",
            "PPO_selfplay_reward_prog005_clear0": "Self-play progress",
            "PPO_selfplay_reward_prog005_clear01": "Self-play progress + clear",
        },
        "colors": {
            "PPO_selfplay_baseline_teams": "#4059ad",
            "PPO_selfplay_reward_prog005_clear0": "#d95f02",
            "PPO_selfplay_reward_prog005_clear01": "#1b9e77",
        },
    },
    "selfplay_tuned": {
        "title": "Tuned self-play PPO checkpoint win rate",
        "side_title": "Tuned self-play PPO side-specific win rate",
        "labels": {
            "PPO_selfplay_baseline_teams_lr25e6_sgd10": "Baseline lr 2.5e-5 / SGD 10",
            "PPO_selfplay_reward_prog005_clear01_lr25e6_sgd10": "Progress + clear lr 2.5e-5 / SGD 10",
        },
        "colors": {
            "PPO_selfplay_baseline_teams_lr25e6_sgd10": "#4059ad",
            "PPO_selfplay_reward_prog005_clear01_lr25e6_sgd10": "#1b9e77",
        },
    },
    "historical": {
        "title": "Historical-opponent self-play PPO checkpoint win rate",
        "side_title": "Historical-opponent self-play PPO side-specific win rate",
        "labels": {
            "PPO_historical_selfplay_baseline_lr5em05_sgd30": "baseline",
            "PPO_historical_selfplay_baseline_lr5em05_sgd30_larger_model": "baseline - large",
            "PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30": "PR",
            "PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10": "PR + low LR",
            "PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30_updateInterval50": "PR + longer update"
        },
        "colors": {
            "PPO_historical_selfplay_baseline_lr5em05_sgd30": "#4059ad",
            "PPO_historical_selfplay_baseline_lr5em05_sgd30_larger_model": "#cfbc53",
            "PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30": "#1b9e77",
            "PPO_historical_selfplay_reward_prog005_clear0_lr2em05_sgd10": "#d95f02",
            "PPO_historical_selfplay_reward_prog005_clear0_lr5em05_sgd30_updateInterval50": "#a270b3"
        },
    },
    "team": {
        "title": "Team-vs-random PPO checkpoint win rate",
        "side_title": "Team-vs-random PPO side-specific win rate",
        "labels": {
            "PPO_baseline_team_vs_random": "Team baseline",
            "PPO_reward_exp_prog005_clear0": "Progress 0.05",
            "PPO_reward_exp_prog005_clear01": "Progress 0.05 + clear 0.1",
            "PPO_reward_exp_prog01_clear00": "Progress 0.1",
            "PPO_reward_exp_prog01_clear01": "Progress 0.1 + clear 0.1",
        },
        "colors": {
            "PPO_baseline_team_vs_random": "#4059ad",
            "PPO_reward_exp_prog005_clear0": "#d95f02",
            "PPO_reward_exp_prog005_clear01": "#1b9e77",
            "PPO_reward_exp_prog01_clear00": "#7570b3",
            "PPO_reward_exp_prog01_clear01": "#e7298a",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot checkpoint win rates against Random and CEIA."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(PLOT_SUITES),
        default="selfplay",
        help="Which experiment family to plot.",
    )
    parser.add_argument("--random-results", default=DEFAULT_RANDOM_RESULTS)
    parser.add_argument("--ceia-results", default=DEFAULT_CEIA_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(path, run_labels):
    rows = defaultdict(list)
    with open(path) as results_file:
        for line in results_file:
            if not line.strip():
                continue
            record = json.loads(line)
            run_name = record.get("run_name", "")
            if run_name not in run_labels:
                continue

            metrics = get_agent_metrics(record)
            rows[run_name].append(
                {
                    "run_name": run_name,
                    "checkpoint_iteration": record["checkpoint_iteration"],
                    "checkpoint_path": record["checkpoint_path"],
                    "overall": metrics["overall"],
                    "blue": metrics["blue"],
                    "orange": metrics["orange"],
                    "wins": metrics["wins"],
                    "losses": metrics["losses"],
                    "draws": metrics["draws"],
                    "games": metrics["games"],
                }
            )

    for run_rows in rows.values():
        run_rows.sort(key=lambda row: row["checkpoint_iteration"])
    return rows


def get_agent_metrics(record):
    opponent = record["opponent"]
    for policy_name, metrics in record["winrates"].items():
        if policy_name != opponent:
            return metrics
    raise ValueError(f"No checkpoint policy metrics found for {record['checkpoint_path']}")


def best_rows(rows_by_opponent):
    best = []
    for opponent_name, rows_by_run in rows_by_opponent.items():
        for run_name, rows in rows_by_run.items():
            row = max(
                rows,
                key=lambda item: (
                    item["overall"],
                    item["wins"],
                    -item["losses"],
                    item["checkpoint_iteration"],
                ),
            )
            best.append({"opponent": opponent_name, **row})
    best.sort(key=lambda row: (row["opponent"], row["run_name"]))
    return best


def write_best_summary(best, output_path):
    with open(output_path, "w", newline="") as csv_file:
        fieldnames = [
            "opponent",
            "run_name",
            "checkpoint_iteration",
            "overall",
            "blue",
            "orange",
            "wins",
            "losses",
            "draws",
            "games",
            "checkpoint_path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in best:
            writer.writerow({field: row[field] for field in fieldnames})


def plot_overall(rows_by_opponent, output_path, run_labels, run_colors, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, (opponent_name, rows_by_run) in zip(axes, rows_by_opponent.items()):
        for run_name, rows in sorted(rows_by_run.items()):
            axis.plot(
                [row["checkpoint_iteration"] for row in rows],
                [row["overall"] for row in rows],
                marker="o",
                linewidth=2,
                color=run_colors[run_name],
                label=run_labels[run_name],
            )
        axis.set_title(f"vs {opponent_name}")
        axis.set_xlabel("Checkpoint iteration")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0.0, 1.05)

    axes[0].set_ylabel("Overall win rate")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_side_breakdown(rows_by_opponent, output_path, run_labels, title):
    run_names = list(run_labels)
    fig, axes = plt.subplots(
        2,
        len(run_names),
        figsize=(4.3 * len(run_names), 7),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    for row_idx, (opponent_name, rows_by_run) in enumerate(rows_by_opponent.items()):
        for col_idx, run_name in enumerate(run_names):
            axis = axes[row_idx][col_idx]
            rows = rows_by_run.get(run_name, [])
            axis.plot(
                [row["checkpoint_iteration"] for row in rows],
                [row["blue"] for row in rows],
                marker="o",
                linewidth=2,
                color="#4059ad",
                label="Blue",
            )
            axis.plot(
                [row["checkpoint_iteration"] for row in rows],
                [row["orange"] for row in rows],
                marker="s",
                linewidth=2,
                color="#d95f02",
                label="Orange",
            )
            axis.plot(
                [row["checkpoint_iteration"] for row in rows],
                [row["overall"] for row in rows],
                linestyle="--",
                linewidth=1.8,
                color="#111111",
                alpha=0.75,
                label="Overall",
            )
            axis.set_title(f"{run_labels[run_name]} vs {opponent_name}")
            axis.set_xlabel("Checkpoint iteration")
            axis.grid(True, alpha=0.3)
            axis.set_ylim(0.0, 1.05)
            if col_idx == 0:
                axis.set_ylabel("Win rate")
            if row_idx == 0 and col_idx == len(run_names) - 1:
                axis.legend(loc="lower right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = PLOT_SUITES[args.suite]
    run_labels = suite["labels"]
    run_colors = suite["colors"]

    rows_by_opponent = {
        "Random": load_rows(Path(args.random_results), run_labels),
        "CEIA": load_rows(Path(args.ceia_results), run_labels),
    }

    plot_overall(
        rows_by_opponent,
        output_dir / f"{args.suite}_winrate_over_time.png",
        run_labels,
        run_colors,
        suite["title"],
    )
    plot_side_breakdown(
        rows_by_opponent,
        output_dir / f"{args.suite}_side_winrate_over_time.png",
        run_labels,
        suite["side_title"],
    )
    write_best_summary(
        best_rows(rows_by_opponent),
        output_dir / f"{args.suite}_best_checkpoints.csv",
    )

    for row in best_rows(rows_by_opponent):
        print(
            f"{row['opponent']} | {run_labels[row['run_name']]} | "
            f"iter={row['checkpoint_iteration']} | overall={row['overall']:.2f} | "
            f"blue={row['blue']:.2f} | orange={row['orange']:.2f} | "
            f"W/L/D={row['wins']}/{row['losses']}/{row['draws']}"
        )


if __name__ == "__main__":
    main()
