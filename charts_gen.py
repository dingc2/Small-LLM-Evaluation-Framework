"""Generate all charts from the latest full local sweep.

Snapshot source: run ``20260421T024743Z`` (10 Ollama models, 33+33 cases,
``runs=3``). Most local metrics are loaded from that run's summary CSV so the
slide-ready charts stay aligned with the README. Frontier bars in chart 8 and
the local intersection slice remain hardcoded because they come from a separate
manual subset analysis.
"""
import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RUN_ID = "20260421T024743Z"
SUMMARY_PATH = BASE_DIR / "results" / f"{RUN_ID}_summary.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "charts")
os.makedirs(OUT, exist_ok=True)

MODELS = [
    "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b",
    "gemma4:e2b", "gemma4:e4b",
    "nemotron-3-nano:4b", "gpt-oss:20b",
    "ministral-3:3b", "ministral-3:8b", "ministral-3:14b",
]

PARAMS_B = {
    "qwen3.5:2b": 2, "qwen3.5:4b": 4, "qwen3.5:9b": 9,
    "gemma4:e2b": 2, "gemma4:e4b": 4,
    "nemotron-3-nano:4b": 4, "gpt-oss:20b": 20,
    "ministral-3:3b": 3, "ministral-3:8b": 8, "ministral-3:14b": 14,
}

FAMILIES = {
    "qwen3.5:2b": "Qwen", "qwen3.5:4b": "Qwen", "qwen3.5:9b": "Qwen",
    "gemma4:e2b": "Gemma", "gemma4:e4b": "Gemma",
    "nemotron-3-nano:4b": "Nemotron", "gpt-oss:20b": "GPT-OSS",
    "ministral-3:3b": "Ministral", "ministral-3:8b": "Ministral", "ministral-3:14b": "Ministral",
}

FAMILY_COLORS = {
    "Qwen": "#4C72B0", "Gemma": "#55A868", "Nemotron": "#C44E52",
    "GPT-OSS": "#8172B2", "Ministral": "#DD8452",
}

def _load_summary_metrics(path: Path):
    e2e_with = {}
    e2e_without = {}
    sel_with = {}
    latency_with = {}
    latency_without = {}

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            bench = row["benchmark"]
            skill_config = row["skill_config"]
            score = float(row["score"])
            avg_latency_ms = float(row["avg_latency_ms"])

            if bench == "end_to_end_task_completion":
                if skill_config == "all_skills":
                    e2e_with[model] = score
                    latency_with[model] = avg_latency_ms
                elif skill_config == "no_skills":
                    e2e_without[model] = score
                    latency_without[model] = avg_latency_ms
            elif bench == "skill_selection_accuracy" and skill_config == "all_skills":
                sel_with[model] = score

    return e2e_with, e2e_without, sel_with, latency_with, latency_without


E2E_WITH, E2E_WITHOUT, SEL_WITH, LATENCY_WITH, LATENCY_WITHOUT = _load_summary_metrics(SUMMARY_PATH)
E2E_DELTA = {m: E2E_WITH[m] - E2E_WITHOUT[m] for m in MODELS}

SHORT = {
    "qwen3.5:2b": "Qwen\n2B", "qwen3.5:4b": "Qwen\n4B", "qwen3.5:9b": "Qwen\n9B",
    "gemma4:e2b": "Gemma\ne2B", "gemma4:e4b": "Gemma\ne4B",
    "nemotron-3-nano:4b": "Nemotron\n4B", "gpt-oss:20b": "GPT-OSS\n20B",
    "ministral-3:3b": "Ministral\n3B", "ministral-3:8b": "Ministral\n8B", "ministral-3:14b": "Ministral\n14B",
}


def family_color(m):
    return FAMILY_COLORS[FAMILIES[m]]


plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── 1. Skill Uplift Bar Chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
sorted_models = sorted(MODELS, key=lambda m: E2E_DELTA[m], reverse=True)
deltas = [E2E_DELTA[m] for m in sorted_models]
colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
labels = [SHORT[m] for m in sorted_models]

bars = ax.bar(range(len(sorted_models)), deltas, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Score Uplift  (with skills − without skills)", fontsize=11)
ax.set_title(f"Skill-Augmentation Uplift by Model (End-to-End Task Completion)\nrun {RUN_ID} — 33 cases × 3 runs",
             fontweight="bold", fontsize=13)

for bar, val in zip(bars, deltas):
    ypos = val + 0.015 if val >= 0 else val - 0.035
    ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:+.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "+25%", "+50%", "+75%", "+100%"])

patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=patches, loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}/1_skill_uplift.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/1_skill_uplift.png")

# ── 2. With vs Without Side-by-Side ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(MODELS))
w = 0.38
ax.bar(x - w / 2, [E2E_WITH[m] for m in MODELS], w, label="With Skills", color="#2ecc71", alpha=0.9)
ax.bar(x + w / 2, [E2E_WITHOUT[m] for m in MODELS], w, label="Without Skills", color="#e74c3c", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=9)
ax.set_ylabel("End-to-End Score")
ax.set_title("With Skills vs Without Skills — End-to-End Task Completion", fontweight="bold", fontsize=13)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.legend(fontsize=10)
ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
fig.savefig(f"{OUT}/2_with_vs_without.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/2_with_vs_without.png")

# ── 3. Model Size vs Score Scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

for m in MODELS:
    fc = family_color(m)
    params = PARAMS_B[m]
    ax.scatter(params, E2E_WITH[m], color=fc, marker="o", s=160, zorder=3,
               edgecolors="white", linewidth=1.2)
    ax.scatter(params, E2E_WITHOUT[m], color=fc, marker="s", s=120, zorder=3,
               edgecolors="white", linewidth=1.2, alpha=0.55)
    ax.annotate(SHORT[m].replace("\n", " "), (params, E2E_WITH[m]),
                textcoords="offset points", xytext=(8, 4), fontsize=8, color=fc)
    ax.plot([params, params], [E2E_WITHOUT[m], E2E_WITH[m]],
            color=fc, linewidth=1.2, alpha=0.4, linestyle="--")

# Reference line: gpt-oss:20b without tools (the "large baseline")
ax.axhline(E2E_WITHOUT["gpt-oss:20b"], color="#8172B2", linewidth=1.2,
           linestyle=":", alpha=0.7,
           label=f"gpt-oss:20b without tools ({E2E_WITHOUT['gpt-oss:20b']:.2f})")

family_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
dot = plt.Line2D([0], [0], marker="o", color="gray", label="With Skills", markersize=9, linestyle="None")
square = plt.Line2D([0], [0], marker="s", color="gray", label="Without Skills", markersize=8, linestyle="None", alpha=0.55)
ax.legend(handles=family_patches + [dot, square], fontsize=9, loc="lower right")

ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
ax.set_ylabel("End-to-End Task Completion Score", fontsize=12)
ax.set_title("Model Size vs. Performance\n● with skills  ▪ without skills  (dashed = uplift gap)",
             fontweight="bold", fontsize=13)
ax.set_xlim(0, 22)
ax.set_ylim(0, 1.05)
ax.set_xticks([2, 3, 4, 8, 9, 14, 20])
ax.set_xticklabels(["2B", "3B", "4B", "8B", "9B", "14B", "20B"])
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(f"{OUT}/3_size_vs_score.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/3_size_vs_score.png")

# ── 4. Latency Comparison ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(MODELS))
w = 0.38
lat_with = [LATENCY_WITH[m] / 1000 for m in MODELS]
lat_without = [LATENCY_WITHOUT[m] / 1000 for m in MODELS]
ax.bar(x - w / 2, lat_with, w, label="With Skills", color="#2ecc71", alpha=0.9)
ax.bar(x + w / 2, lat_without, w, label="Without Skills", color="#e74c3c", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=9)
ax.set_ylabel("Avg Latency per Test Case (seconds)", fontsize=11)
ax.set_title("Inference Latency by Model\n(With Skills includes multi-turn tool calls)",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/4_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/4_latency.png")

# ── 5. Score Heatmap ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
row_labels = [SHORT[m].replace("\n", " ") for m in MODELS]
col_labels = ["E2E\nw/ skills", "E2E\nw/o skills", "SkillSel\nw/ skills"]
matrix = np.array([
    [E2E_WITH[m], E2E_WITHOUT[m], SEL_WITH[m]]
    for m in MODELS
])
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(row_labels, fontsize=10)
for i in range(len(MODELS)):
    for j in range(3):
        v = matrix[i, j]
        color = "white" if v < 0.35 or v > 0.75 else "black"
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
ax.set_title("Score Heatmap — All Models × All Conditions", fontweight="bold", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/5_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/5_heatmap.png")

# ── 6. Qwen Family: Not-so-clean diminishing returns ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
qwen = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
qsize = [2, 4, 9]
qwith = [E2E_WITH[m] for m in qwen]
qwithout = [E2E_WITHOUT[m] for m in qwen]

ax.plot(qsize, qwith, "o-", color="#2ecc71", linewidth=2.5, markersize=11, label="With Skills")
ax.plot(qsize, qwithout, "s--", color="#e74c3c", linewidth=2.5, markersize=9, label="Without Skills", alpha=0.8)

for x_, yw, yo in zip(qsize, qwith, qwithout):
    ax.annotate(f"{yw:.0%}", (x_, yw), textcoords="offset points", xytext=(6, 8), color="#2ecc71", fontweight="bold")
    ax.annotate(f"{yo:.0%}", (x_, yo), textcoords="offset points", xytext=(6, -14), color="#e74c3c", fontweight="bold")
    ax.annotate(f"Δ{yw-yo:+.0%}", (x_, (yw + yo) / 2), textcoords="offset points",
                xytext=(14, 0), color="#555", fontsize=10, fontweight="bold")

ax.fill_between(qsize, qwithout, qwith, alpha=0.12, color="#2ecc71", label="Uplift gap")
ax.set_xticks(qsize)
ax.set_xticklabels(["2B", "4B", "9B"])
ax.set_xlabel("Model Size (Parameters)", fontsize=12)
ax.set_ylabel("End-to-End Score", fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_title("Qwen3.5 Family — 9B Edges 4B, Both Far Above 2B",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=10, loc="lower right")
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(f"{OUT}/6_qwen_family.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/6_qwen_family.png")

# ── 7. Research-Question Chart: small-with-tools vs large-without ─────────────
fig, ax = plt.subplots(figsize=(10, 6))
small_with = sorted(
    [(m, E2E_WITH[m]) for m in MODELS if PARAMS_B[m] <= 9],
    key=lambda t: t[1], reverse=True,
)
ax.barh([m for m, _ in small_with], [s for _, s in small_with],
        color="#2ecc71", alpha=0.9, label="Small model WITH tools (≤9B)")
ax.axvline(E2E_WITHOUT["gpt-oss:20b"], color="#8172B2", linewidth=2.5,
           linestyle="--", label=f"gpt-oss:20b WITHOUT tools ({E2E_WITHOUT['gpt-oss:20b']:.2f})")

for i, (m, s) in enumerate(small_with):
    ax.text(s + 0.01, i, f"{s:.2f}", va="center", fontsize=10, fontweight="bold")

ax.set_xlim(0, 1.1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("End-to-End Score", fontsize=11)
ax.set_title("Research Question: Does small-with-tools beat large-without?\n"
             "(Models ≤9B with tools vs. gpt-oss:20b baseline without tools)",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/7_research_question.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ {OUT}/7_research_question.png")

print(f"\nAll charts saved to {OUT}/")


# ── 8. Frontier (no tools) vs Small-local (with tools) — intersection-24 ──────
def chart_frontier_vs_small_tool(out_dir):
    """Grouped bar chart comparing frontier-no-tool vs small-local-with-tool
    on the 24 intersection cases (17 strong + 7 moderate) where precision math
    and niche formulas make frontier-no-tool plausibly struggle."""

    INTERSECTION24 = {
        # (model_label, config_label): accuracy
        ("gpt-4.1-mini",      "no tools\n(frontier)"):   0.764,
        ("gpt-4.1-nano",      "no tools\n(frontier)"):   0.667,
        ("gpt-5.4-mini",      "no tools\n(frontier)"):   0.778,
        ("qwen3.5:4b",        "with tools\n(local)"):    0.958,
        ("ministral-3:8b",    "with tools\n(local)"):    1.000,
        ("nemotron-3-nano:4b","with tools\n(local)"):    0.722,
        ("qwen3.5:4b",        "no tools\n(local ref)"):  0.042,
        ("ministral-3:8b",    "no tools\n(local ref)"):  0.458,
        ("nemotron-3-nano:4b","no tools\n(local ref)"):  0.639,
    }

    # Colors keyed to model family; frontier models get GPT-OSS purple
    FRONTIER_COLOR = "#8172B2"   # GPT-OSS / frontier (matches FAMILY_COLORS["GPT-OSS"])
    MODEL_COLORS = {
        "gpt-4.1-mini":       FRONTIER_COLOR,
        "gpt-4.1-nano":       FRONTIER_COLOR,
        "gpt-5.4-mini":       FRONTIER_COLOR,
        "qwen3.5:4b":         "#4C72B0",  # Qwen
        "ministral-3:8b":     "#DD8452",  # Ministral
        "nemotron-3-nano:4b": "#C44E52",  # Nemotron
    }

    # Three logical groups in display order
    groups = [
        # (model, config_label, x_label)
        ("gpt-4.1-mini",       "no tools\n(frontier)",  "gpt-4.1-mini\nno tools"),
        ("gpt-4.1-nano",       "no tools\n(frontier)",  "gpt-4.1-nano\nno tools"),
        ("gpt-5.4-mini",       "no tools\n(frontier)",  "gpt-5.4-mini\nno tools"),
        ("qwen3.5:4b",         "with tools\n(local)",   "qwen3.5:4b\n+ tools"),
        ("ministral-3:8b",     "with tools\n(local)",   "ministral-3:8b\n+ tools"),
        ("nemotron-3-nano:4b", "with tools\n(local)",   "nemotron\n4B + tools"),
        ("qwen3.5:4b",         "no tools\n(local ref)", "qwen3.5:4b\nno tools"),
        ("ministral-3:8b",     "no tools\n(local ref)", "ministral-3:8b\nno tools"),
        ("nemotron-3-nano:4b", "no tools\n(local ref)", "nemotron\n4B no tools"),
    ]

    bar_values = [INTERSECTION24[(m, cfg)] for m, cfg, _ in groups]
    bar_colors = [MODEL_COLORS[m] for m, _, _ in groups]
    x_labels   = [lbl for _, _, lbl in groups]
    x_pos      = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(14.5, 7))

    bars = ax.bar(x_pos, bar_values, color=bar_colors,
                  edgecolor="white", linewidth=0.8, width=0.6, zorder=3)

    # Numeric labels on top of each bar
    for bar, val in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.018, f"{val:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference line: best frontier-no-tool score (gpt-5.4-mini = 0.778)
    ax.axhline(0.778, color="#333333", linewidth=1.5, linestyle="--", zorder=2,
               label="best frontier-no-tool (gpt-5.4-mini = 0.778)")
    ax.text(len(groups) - 0.35, 0.778 + 0.018, "best frontier-no-tool",
            ha="right", va="bottom", fontsize=9, color="#333333", style="italic")

    # Group divider lines between logical groups
    # Divider after bar index 2 (between Frontier and Small+tools)
    # Divider after bar index 5 (between Small+tools and Small no-tools)
    for div_x, label_left, label_right in [
        (2.5, "Frontier\n(no tools)", "Small-local\n(with tools)"),
        (5.5, "Small-local\n(with tools)", "Small-local\n(no tools, ref)"),
    ]:
        ax.axvline(div_x, color="gray", linewidth=1.2, linestyle="-", alpha=0.5, zorder=1)

    # Group labels above plot area
    group_label_y = 1.075
    group_spans = [
        (0,   2,   "Group A — Frontier (no tools)"),
        (3,   5,   "Group B — Small-local (with tools)"),
        (6,   8,   "Group C — Small-local (no tools, ref)"),
    ]
    for x_start, x_end, glabel in group_spans:
        mid = (x_start + x_end) / 2
        ax.text(mid, group_label_y, glabel,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#444444",
                transform=ax.get_xaxis_transform())
        # Underline bracket
        ax.annotate("", xy=(x_end + 0.4, group_label_y - 0.04),
                    xytext=(x_start - 0.4, group_label_y - 0.04),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=1.0))

    # Axes formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Accuracy on 24 intersection cases", fontsize=11)
    ax.set_ylim(0.0, 1.20)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(
        "Frontier (no tools) vs. Small-local (with tools) — 24 intersection cases",
        fontweight="bold", fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend for model families
    legend_patches = [
        mpatches.Patch(color="#8172B2", label="Frontier / GPT-4.1 (no tools)"),
        mpatches.Patch(color="#4C72B0", label="Qwen"),
        mpatches.Patch(color="#DD8452", label="Ministral"),
        mpatches.Patch(color="#C44E52", label="Nemotron"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    # Footer caption
    caption = (
        "Intersection-24: cases where frontier-no-tools plausibly struggles (precision math, niche formulas).\n"
        "Strong-17 subset: frontier-no-tool caps at 0.745 (gpt-5.4-mini) / 0.725 (gpt-4.1-mini) / 0.588 (gpt-4.1-nano) "
        "while ministral-3:8b with tools hits 1.000."
    )
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8,
             color="#555555", wrap=True)

    plt.tight_layout()
    fig.savefig(f"{out_dir}/8_frontier_vs_small_tool.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {out_dir}/8_frontier_vs_small_tool.png")


chart_frontier_vs_small_tool(OUT)
