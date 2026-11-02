"""Plot logged training MA100 success rates and the matched SFT evaluation.

Run with the RLinf environment; no training, GPU access, or TensorBoard server
is needed. The default inputs are the completed chunk20 stability experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE_TAG = "env/success_rate_ma100"
COUNT_TAG = "env/success_rate_ma100_count"
EPISODES_TAG = "env/num_trajectories"


def training_curve(logdir: Path) -> list[dict]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    reader = EventAccumulator(str(logdir), size_guidance={"scalars": 0}).Reload()
    series = {}
    for tag in (RATE_TAG, COUNT_TAG, EPISODES_TAG):
        events = reader.Scalars(tag)
        values = {event.step: event.value for event in events}
        if not values or len(values) != len(events):
            raise ValueError(f"Empty or duplicate scalar steps for {tag}: {logdir}")
        series[tag] = values
    steps = sorted(series[RATE_TAG])
    if steps != list(range(len(steps))) or any(
        set(values) != set(steps) for values in series.values()
    ):
        raise ValueError(
            f"Expected matching, contiguous steps starting at zero: {logdir}"
        )

    rows = []
    completed = 0
    for step in steps:
        rate = series[RATE_TAG][step]
        count = series[COUNT_TAG][step]
        episodes = series[EPISODES_TAG][step]
        if not all(math.isfinite(value) for value in (rate, count, episodes)):
            raise ValueError(f"Non-finite metric at step {step}: {logdir}")
        if not 0 <= rate <= 1 or episodes <= 0 or not episodes.is_integer():
            raise ValueError(f"Invalid success rate or episode count at step {step}")
        completed += int(episodes)
        if count != min(100, completed):
            raise ValueError(f"Unexpected MA100 denominator {count} at step {step}")
        rows.append(
            {
                "tensorboard_step": step,
                "collection_iteration": step + 1,
                "completed_training_episodes": completed,
                "new_episodes": int(episodes),
                "success_rate_ma100": rate,
                "window_count": int(count),
                "full_window": count == 100,
            }
        )
    if not any(row["full_window"] for row in rows):
        raise ValueError(f"No complete 100-episode window: {logdir}")
    return rows


def evaluation(path: Path) -> dict:
    value = json.loads(path.read_text())
    episodes = value["episodes"]
    rate = float(value["success_rate"])
    if (
        not isinstance(episodes, int)
        or episodes <= 0
        or not math.isfinite(rate)
        or not 0 <= rate <= 1
    ):
        raise ValueError(f"Invalid evaluation record: {path}")
    successes = round(rate * episodes)
    if abs(successes / episodes - rate) > 1e-6:
        raise ValueError(f"Success rate is inconsistent with episode count: {path}")
    return {
        "record": str(path.resolve()),
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "checkpoint": value.get("checkpoint"),
    }


def fingerprint(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("icon", "robometer"):
        parser.add_argument(
            f"--{name}-logdir",
            type=Path,
            default=ROOT / f"outputs/{name}_dsrl_chunk20_stable_s1234/tensorboard",
        )
        parser.add_argument(
            f"--{name}-evaluation",
            type=Path,
            default=ROOT / f"outputs/chunk20_stable/{name}_evaluation.json",
        )
    parser.add_argument(
        "--sft-evaluation",
        type=Path,
        default=ROOT / "outputs/icon_chunk20/baseline_evaluation.json",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "outputs/chunk20_stable/figures/success_rate_comparison",
    )
    parser.add_argument(
        "--include-partial-window",
        action="store_true",
        help="Also show the first 12 collections, before 100 episodes are available.",
    )
    parser.add_argument(
        "--y-min", type=float, default=None, help="Lower axis limit in percent."
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # Keep the plotting backend and font cache independent of display/GPU state.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/runtime/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    curves = {
        "ICON": training_curve(args.icon_logdir),
        "Robometer": training_curve(args.robometer_logdir),
    }
    evaluations = {
        "SFT": evaluation(args.sft_evaluation),
        "ICON": evaluation(args.icon_evaluation),
        "Robometer": evaluation(args.robometer_evaluation),
    }
    plotted = {
        name: [row for row in rows if row["full_window"] or args.include_partial_window]
        for name, rows in curves.items()
    }
    sft = evaluations["SFT"]
    lowest = min(
        sft["success_rate"] * 100,
        *(row["success_rate_ma100"] * 100 for rows in plotted.values() for row in rows),
    )
    y_min = (
        max(0, 10 * math.floor((lowest - 5) / 10)) if args.y_min is None else args.y_min
    )
    if not 0 <= y_min < 100 or y_min > lowest or args.dpi <= 0:
        parser.error(
            "--y-min must include every plotted rate within [0, 100); --dpi must be positive"
        )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    fields = ["method", *next(iter(curves.values()))[0].keys()]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rows in curves.items():
            writer.writerows({"method": name, **row} for row in rows)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(10, 6.2))
        fig.subplots_adjust(left=0.095, right=0.975, top=0.86, bottom=0.24)
        colors = {"ICON": "#D55E00", "Robometer": "#0072B2"}
        for name, rows in plotted.items():
            ax.plot(
                [row["completed_training_episodes"] for row in rows],
                [100 * row["success_rate_ma100"] for row in rows],
                color=colors[name],
                linewidth=1.9,
                label=name,
            )
        ax.axhline(
            sft["success_rate"] * 100,
            color="#555555",
            linewidth=1.7,
            linestyle=(0, (6, 4)),
            label=f"SFT evaluation ({100 * sft['success_rate']:.2f}%)",
            zorder=1,
        )
        max_episodes = max(
            rows[-1]["completed_training_episodes"] for rows in curves.values()
        )
        ax.set(
            xlim=(0, max_episodes),
            ylim=(y_min, 100),
            xlabel="Completed training episodes",
            ylabel="Success rate",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(400))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.grid(axis="x", alpha=0.10, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#dddddd")
        fig.text(0.095, 0.945, "ICON vs. Robometer", fontsize=18, weight="bold")
        fig.text(
            0.095,
            0.897,
            "RoboTwin adjust_bottle · π0.5 + DSRL · 20-action execution",
            color="#555555",
            fontsize=11,
        )
        window_note = (
            "partial startup windows included"
            if args.include_partial_window
            else "partial startup windows omitted"
        )
        fig.text(
            0.095,
            0.115,
            f"Curves: training MA100 ({window_note}); no additional smoothing.",
            fontsize=9,
            color="#555555",
        )
        fig.text(
            0.095,
            0.080,
            f"Dashed: matched SFT held-out evaluation ({sft['successes']}/{sft['episodes']}). One training seed per method.",
            fontsize=9,
            color="#555555",
        )
        final_scores = " · ".join(
            f"{name} {100 * evaluations[name]['success_rate']:.2f}% ({evaluations[name]['successes']}/{evaluations[name]['episodes']})"
            for name in curves
        )
        fig.text(
            0.095,
            0.045,
            "Final held-out evaluation: " + final_scores,
            fontsize=9,
            color="#333333",
        )
        figures = []
        for extension in ("png", "pdf", "svg"):
            output = prefix.with_suffix("." + extension)
            fig.savefig(output, dpi=args.dpi, facecolor="white")
            figures.append(str(output))
        plt.close(fig)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": fingerprint(Path(__file__)),
        "training_metric": RATE_TAG,
        "x_axis": "Cumulative env/num_trajectories (completed training episodes)",
        "window": 100,
        "include_partial_window": args.include_partial_window,
        "additional_smoothing": False,
        "y_axis_percent": [y_min, 100],
        "comparison_note": "Training curves and held-out SFT baseline use different splits; final held-out DSRL scores are separate caption annotations.",
        "training_seeds_per_method": 1,
        "evaluations": evaluations,
        "training": {},
        "outputs": [*figures, str(csv_path)],
    }
    for name, logdir in (
        ("ICON", args.icon_logdir),
        ("Robometer", args.robometer_logdir),
    ):
        report["training"][name] = {
            "logdir": str(logdir.resolve()),
            "event_files": [
                fingerprint(path)
                for path in sorted(logdir.glob("events.out.tfevents.*"))
            ],
            "logged_points": len(curves[name]),
            "plotted_points": len(plotted[name]),
            "first_plotted_episode": plotted[name][0]["completed_training_episodes"],
            "completed_episodes": curves[name][-1]["completed_training_episodes"],
            "final_training_ma100": curves[name][-1]["success_rate_ma100"],
        }
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "figures": figures,
                "data": str(csv_path),
                "metadata": str(prefix.with_suffix(".json")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
