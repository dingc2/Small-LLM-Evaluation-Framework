#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Path shim — makes `python analyze.py` work from inside sLLM_eval_framework/
# ---------------------------------------------------------------------------
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ---------------------------------------------------------------------------

"""
Result analysis and visualisation for the Small-LLM Skill-Uplift study.

Reads the JSON output from the evaluation runner and produces:
  1. Summary comparison table (terminal + CSV)
  2. Bar chart: skill uplift per model  (PNG)
  3. Heatmap: model × benchmark scores  (PNG)
  4. Latency comparison chart            (PNG)
  5. Model-size scatter plot             (PNG)
  6. Per-skill breakdown bar chart       (PNG)

Usage (run from inside sLLM_eval_framework/)
-----
    python analyze.py results/<run_id>_results.json
    python analyze.py results/<run_id>_results.json --output results/charts
    python analyze.py results/<run_id>_results.json --no-charts   # text only

Requires: matplotlib, numpy  (pip install matplotlib numpy)
"""



import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Approximate parameter counts (billions) for common Ollama models
# Used for the model-size scatter plot
# ---------------------------------------------------------------------------
MODEL_PARAMS_B: dict[str, float] = {
    # --- Study models ---
    "qwen3.5:2b": 2.0,
    "qwen3.5:4b": 4.0,
    "qwen3.5:9b": 9.0,
    "gemma4:e2b": 2.0,
    "gemma4:e4b": 4.0,
    "nemotron-3-nano:4b": 4.0,
    "gpt-oss:20b": 20.0,
    "ministral-3:3b": 3.0,
    "ministral-3:8b": 8.0,
    "ministral-3:14b": 14.0,
    # --- Common extras ---
    "gemma3:4b": 4.0,
    "gemma3:12b": 12.0,
    "llama3.2:3b": 3.0,
    "llama3.1:8b": 8.0,
    "qwen3:4b": 4.0,
    "qwen3:8b": 8.0,
    "phi3:3.8b": 3.8,
    "mistral:7b": 7.0,
}


def load_results(path: str | Path) -> dict[str, Any]:
    """Load a runner JSON results file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(data: dict[str, Any]) -> None:
    """Print a human-readable summary of the evaluation run."""
    print("=" * 72)
    print(f"  Small-LLM Skill-Uplift Evaluation — Run {data['run_id']}")
    print(f"  Duration: {data['duration_s']:.1f}s")
    print("=" * 72)

    table = data.get("comparison_table", [])
    if not table:
        print("  No comparison data found.")
        return

    # Column widths
    cols = list(table[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r.get(c, ""))) for r in table)) for c in cols}
    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    hdr = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"

    print(sep)
    print(hdr)
    print(sep)
    for row in table:
        line = "| " + " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols) + " |"
        print(line)
    print(sep)

    # Highlight key findings
    print("\n--- Key Findings ---")
    _print_uplift_summary(table)


def _print_uplift_summary(table: list[dict[str, Any]]) -> None:
    """Extract and print the skill uplift highlights."""
    for row in table:
        delta = row.get("skill_delta", "—")
        if delta not in ("—", "n/a", ""):
            try:
                d = float(delta)
                model = row["model"]
                bench = row["benchmark"]
                direction = "improvement" if d > 0 else "degradation"
                print(f"  {model} on {bench}: {delta} ({direction})")
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Charts (matplotlib)
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping chart generation.")
        return None


def generate_charts(data: dict[str, Any], output_dir: str | Path) -> list[str]:
    """Generate all visualisation charts. Returns list of saved file paths."""
    plt = _ensure_matplotlib()
    if plt is None:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    results = data.get("results", [])
    table = data.get("comparison_table", [])

    if not results:
        print("No results to chart.")
        return saved

    # 1. Skill uplift bar chart
    path = _chart_skill_uplift(plt, table, output_dir)
    if path:
        saved.append(path)

    # 2. Score heatmap
    path = _chart_score_heatmap(plt, table, output_dir)
    if path:
        saved.append(path)

    # 3. Latency comparison
    path = _chart_latency(plt, table, output_dir)
    if path:
        saved.append(path)

    # 4. Model-size vs score scatter
    path = _chart_size_vs_score(plt, table, output_dir)
    if path:
        saved.append(path)

    # 5. Per-skill breakdown
    path = _chart_per_skill_breakdown(plt, results, output_dir)
    if path:
        saved.append(path)

    return saved


def _chart_skill_uplift(plt, table: list[dict], output_dir: Path) -> str | None:
    """Bar chart showing skill uplift (delta) per model for end-to-end benchmark."""
    # Filter to end-to-end + all_skills rows (which have deltas)
    rows = [r for r in table if r.get("skill_delta") not in ("—", "n/a", "")]

    if not rows:
        return None

    models = [r["model"] for r in rows]
    deltas = []
    errs = []
    for r in rows:
        try:
            deltas.append(float(r["skill_delta"]))
        except ValueError:
            deltas.append(0.0)
        try:
            errs.append(float(r.get("score_std", 0)))
        except (ValueError, TypeError):
            errs.append(0.0)

    # Only show error bars when there's meaningful variance (n_runs > 1)
    show_errs = any(e > 0 for e in errs)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    bars = ax.barh(
        models, deltas,
        xerr=errs if show_errs else None,
        color=colors, edgecolor="white", linewidth=0.5,
        error_kw={"ecolor": "black", "capsize": 4, "linewidth": 1.2},
    )

    ax.set_xlabel("Score Uplift (with skills − without skills)")
    ax.set_title("Skill-Augmentation Uplift by Model", fontweight="bold", fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.8)

    if show_errs:
        ax.set_xlabel("Score Uplift (with skills − without skills) ± 1 SD")

    # Add value labels
    for bar, val in zip(bars, deltas):
        x_pos = bar.get_width()
        ax.text(
            x_pos + 0.01 if x_pos >= 0 else x_pos - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=10,
        )

    plt.tight_layout()
    path = str(output_dir / "skill_uplift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _chart_score_heatmap(plt, table: list[dict], output_dir: Path) -> str | None:
    """Heatmap of scores: rows=models, columns=benchmark×skill_config."""
    if not table:
        return None

    import numpy as np

    # Build matrix
    models = sorted(set(r["model"] for r in table))
    combos = sorted(set((r["benchmark"], r["skill_config"]) for r in table))
    combo_labels = [f"{b}\n({s})" for b, s in combos]

    matrix = np.zeros((len(models), len(combos)))
    for r in table:
        i = models.index(r["model"])
        j = combos.index((r["benchmark"], r["skill_config"]))
        try:
            matrix[i, j] = float(r["score"])
        except ValueError:
            matrix[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(combo_labels)))
    ax.set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(combos)):
            val = matrix[i, j]
            color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Evaluation Scores by Model and Configuration", fontweight="bold", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
    plt.tight_layout()

    path = str(output_dir / "score_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _chart_latency(plt, table: list[dict], output_dir: Path) -> str | None:
    """Grouped bar chart of average latency per model."""
    if not table:
        return None

    # Group by model, collecting latency mean and std
    model_latencies: dict[str, dict[str, float]] = defaultdict(dict)
    model_lat_stds: dict[str, dict[str, float]] = defaultdict(dict)
    for r in table:
        try:
            model_latencies[r["model"]][r["skill_config"]] = float(r["avg_latency_ms"])
            model_lat_stds[r["model"]][r["skill_config"]] = float(r.get("latency_std", 0) or 0)
        except (ValueError, KeyError):
            pass

    if not model_latencies:
        return None

    models = sorted(model_latencies.keys())
    configs = sorted(set(c for m in model_latencies.values() for c in m.keys()))

    import numpy as np
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    show_errs = any(
        model_lat_stds[m].get(cfg, 0) > 0
        for m in models for cfg in configs
    )

    for i, cfg in enumerate(configs):
        vals = [model_latencies[m].get(cfg, 0) for m in models]
        stds = [model_lat_stds[m].get(cfg, 0) for m in models]
        offset = (i - len(configs) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width, label=cfg, alpha=0.85,
            yerr=stds if show_errs else None,
            error_kw={"ecolor": "black", "capsize": 3, "linewidth": 1.0},
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Latency (ms)" + (" ± 1 SD" if show_errs else ""))
    ax.set_title("Inference Latency by Model and Skill Configuration", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    path = str(output_dir / "latency_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _chart_size_vs_score(plt, table: list[dict], output_dir: Path) -> str | None:
    """Scatter plot: model parameter count vs score, colored by skill config."""
    if not table:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))

    configs = sorted(set(r["skill_config"] for r in table))
    colors = {"all_skills": "#2ecc71", "no_skills": "#e74c3c", "calc_only": "#3498db"}
    markers = {"all_skills": "o", "no_skills": "s", "calc_only": "^"}

    for cfg in configs:
        rows = [r for r in table if r["skill_config"] == cfg]
        xs, ys, labels = [], [], []
        for r in rows:
            model = r["model"]
            params = MODEL_PARAMS_B.get(model)
            if params is None:
                continue
            try:
                score = float(r["score"])
            except ValueError:
                continue
            xs.append(params)
            ys.append(score)
            labels.append(model)

        if xs:
            ax.scatter(
                xs, ys,
                label=cfg,
                color=colors.get(cfg, "#95a5a6"),
                marker=markers.get(cfg, "o"),
                s=120,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(
                    lbl, (x, y),
                    textcoords="offset points",
                    xytext=(8, 4),
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.set_title(
        "Model Size vs. Performance: Does Skill Access Close the Gap?",
        fontweight="bold", fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = str(output_dir / "size_vs_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _chart_per_skill_breakdown(plt, results: list[dict], output_dir: Path) -> str | None:
    """Breakdown of pass rate per skill category per model (from test_results)."""
    # Parse test results to categorise by skill
    skill_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for br in results:
        model = br.get("model_name", "unknown")
        sc_name = br.get("metadata", {}).get("skill_config_name", "")
        if sc_name != "all_skills":
            continue

        for tr in br.get("test_results", []):
            test_id = tr.get("test_id", "")
            score = tr.get("score", 0.0)

            # Categorise by prefix
            if "calc" in test_id:
                skill_scores[model]["Calculator"].append(score)
            elif "conv" in test_id:
                skill_scores[model]["Unit Converter"].append(score)
            elif "dict" in test_id:
                skill_scores[model]["Dictionary"].append(score)
            elif "date" in test_id:
                skill_scores[model]["Date/Time"].append(score)
            elif "no_tool" in test_id or "no_skill" in test_id:
                skill_scores[model]["No Tool"].append(score)

    if not skill_scores:
        return None

    import numpy as np

    models = sorted(skill_scores.keys())
    skills = ["Calculator", "Unit Converter", "Dictionary", "Date/Time", "No Tool"]
    skills = [s for s in skills if any(s in skill_scores[m] for m in models)]

    x = np.arange(len(models))
    width = 0.8 / max(len(skills), 1)
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.Set2
    for i, skill in enumerate(skills):
        vals = []
        for m in models:
            scores = skill_scores[m].get(skill, [])
            vals.append(sum(scores) / len(scores) if scores else 0.0)
        offset = (i - len(skills) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=skill, color=cmap(i / len(skills)), alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Per-Skill Performance Breakdown (with skills enabled)", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=min(len(skills), 3), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = str(output_dir / "per_skill_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse and visualise Small-LLM evaluation results"
    )
    parser.add_argument(
        "results_json",
        help="Path to the *_results.json output from the runner",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for charts (default: same dir as results)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (text summary only)",
    )
    args = parser.parse_args()

    data = load_results(args.results_json)
    print_summary(data)

    if not args.no_charts:
        output_dir = args.output or str(Path(args.results_json).parent / "charts")
        print(f"\nGenerating charts in {output_dir}...")
        saved = generate_charts(data, output_dir)
        if saved:
            print(f"\n  {len(saved)} chart(s) saved to {output_dir}/")
        else:
            print("  No charts generated (install matplotlib for visualisations).")


if __name__ == "__main__":
    main()
