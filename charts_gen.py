"""Generate all charts from the full sweep results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/sessions/festive-eager-gates/mnt/small-llm-test/eval_framework/results/charts"

# ── Raw results ──────────────────────────────────────────────────────────────
MODELS = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b",
          "gemma4:e2b", "gemma4:e4b",
          "nemotron-3-nano:4b", "gpt-oss:20b"]

PARAMS_B = {"qwen3.5:2b": 2, "qwen3.5:4b": 4, "qwen3.5:9b": 9,
             "gemma4:e2b": 2, "gemma4:e4b": 4,
             "nemotron-3-nano:4b": 4, "gpt-oss:20b": 20}

FAMILIES = {"qwen3.5:2b": "Qwen", "qwen3.5:4b": "Qwen", "qwen3.5:9b": "Qwen",
             "gemma4:e2b": "Gemma", "gemma4:e4b": "Gemma",
             "nemotron-3-nano:4b": "Nemotron", "gpt-oss:20b": "GPT-OSS"}

FAMILY_COLORS = {"Qwen": "#4C72B0", "Gemma": "#55A868", "Nemotron": "#C44E52", "GPT-OSS": "#8172B2"}

E2E_WITH    = {"qwen3.5:2b": 0.900, "qwen3.5:4b": 0.950, "qwen3.5:9b": 0.900,
               "gemma4:e2b": 0.400, "gemma4:e4b": 1.000,
               "nemotron-3-nano:4b": 0.800, "gpt-oss:20b": 1.000}
E2E_WITHOUT = {"qwen3.5:2b": 0.350, "qwen3.5:4b": 0.600, "qwen3.5:9b": 0.650,
               "gemma4:e2b": 0.750, "gemma4:e4b": 0.500,
               "nemotron-3-nano:4b": 0.750, "gpt-oss:20b": 0.750}
E2E_DELTA   = {m: E2E_WITH[m] - E2E_WITHOUT[m] for m in MODELS}

SEL_WITH    = {"qwen3.5:2b": 1.000, "qwen3.5:4b": 1.000, "qwen3.5:9b": 1.000,
               "gemma4:e2b": 1.000, "gemma4:e4b": 1.000,
               "nemotron-3-nano:4b": 0.960, "gpt-oss:20b": 1.000}
SEL_WITHOUT = {"qwen3.5:2b": 0.000, "qwen3.5:4b": 1.000, "qwen3.5:9b": 0.400,
               "gemma4:e2b": 1.000, "gemma4:e4b": 1.000,
               "nemotron-3-nano:4b": 1.000, "gpt-oss:20b": 1.000}

LATENCY_WITH    = {"qwen3.5:2b": 95677, "qwen3.5:4b": 90120, "qwen3.5:9b": 161398,
                   "gemma4:e2b": 15951,  "gemma4:e4b": 35202,
                   "nemotron-3-nano:4b": 100508, "gpt-oss:20b": 27408}
LATENCY_WITHOUT = {"qwen3.5:2b": 72966,  "qwen3.5:4b": 68751, "qwen3.5:9b": 122616,
                   "gemma4:e2b": 17347,  "gemma4:e4b": 12439,
                   "nemotron-3-nano:4b": 28245, "gpt-oss:20b": 20188}

SHORT = {  # short display labels
    "qwen3.5:2b": "Qwen3.5\n2B", "qwen3.5:4b": "Qwen3.5\n4B", "qwen3.5:9b": "Qwen3.5\n9B",
    "gemma4:e2b": "Gemma4\ne2B", "gemma4:e4b": "Gemma4\ne4B",
    "nemotron-3-nano:4b": "Nemotron\n4B", "gpt-oss:20b": "GPT-OSS\n20B"
}

def family_color(m): return FAMILY_COLORS[FAMILIES[m]]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ── 1. Skill Uplift Bar Chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
sorted_models = sorted(MODELS, key=lambda m: E2E_DELTA[m], reverse=True)
deltas = [E2E_DELTA[m] for m in sorted_models]
colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
labels = [SHORT[m] for m in sorted_models]
fcolors = [family_color(m) for m in sorted_models]

bars = ax.bar(range(len(sorted_models)), deltas, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Score Uplift  (with skills − without skills)", fontsize=11)
ax.set_title("Skill-Augmentation Uplift by Model\n(End-to-End Task Completion)", fontweight="bold", fontsize=14)
fig.text(0.5, -0.02,
         "Note: 'with skills' runs 20 tool-dependent + 2 baseline cases; 'without skills' runs 2 baseline cases only.",
         ha="center", fontsize=8, style="italic", color="#666")

for bar, val in zip(bars, deltas):
    ypos = val + 0.01 if val >= 0 else val - 0.03
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:+.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(-0.55, 0.80)
ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75])
ax.set_yticklabels(["-50%", "-25%", "0%", "+25%", "+50%", "+75%"])

# Family legend
patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=patches, loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(f"{OUT}/1_skill_uplift.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 1_skill_uplift.png")

# ── 2. With vs Without Side-by-Side ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for ax, title, with_d, without_d, ylabel in [
    (axes[0], "End-to-End Task Completion", E2E_WITH, E2E_WITHOUT, "Score"),
    (axes[1], "Skill Selection Accuracy",   SEL_WITH, SEL_WITHOUT, "Score"),
]:
    x = np.arange(len(MODELS))
    w = 0.35
    b1 = ax.bar(x - w/2, [with_d[m] for m in MODELS],    w, label="With Skills",    color="#2ecc71", alpha=0.9)
    b2 = ax.bar(x + w/2, [without_d[m] for m in MODELS], w, label="Without Skills", color="#e74c3c", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")

plt.suptitle("With Skills vs Without Skills — All Models", fontweight="bold", fontsize=14)
plt.tight_layout()
fig.savefig(f"{OUT}/2_with_vs_without.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 2_with_vs_without.png")

# ── 3. Model Size vs Score Scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for m in MODELS:
    fc = family_color(m)
    params = PARAMS_B[m]
    ax.scatter(params, E2E_WITH[m],    color=fc, marker="o", s=160, zorder=3,
               edgecolors="white", linewidth=1.2)
    ax.scatter(params, E2E_WITHOUT[m], color=fc, marker="s", s=120, zorder=3,
               edgecolors="white", linewidth=1.2, alpha=0.55)
    ax.annotate(SHORT[m].replace("\n", " "), (params, E2E_WITH[m]),
                textcoords="offset points", xytext=(8, 4), fontsize=8, color=fc)

# Connecting lines (same model, different conditions)
for m in MODELS:
    ax.plot([PARAMS_B[m], PARAMS_B[m]],
            [E2E_WITHOUT[m], E2E_WITH[m]],
            color=family_color(m), linewidth=1.2, alpha=0.4, linestyle="--")

# Legend
family_patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
dot   = plt.Line2D([0],[0], marker="o", color="gray", label="With Skills",    markersize=9, linestyle="None")
square= plt.Line2D([0],[0], marker="s", color="gray", label="Without Skills", markersize=8, linestyle="None", alpha=0.55)
ax.legend(handles=family_patches + [dot, square], fontsize=9, loc="lower right")

ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
ax.set_ylabel("End-to-End Task Completion Score", fontsize=12)
ax.set_title("Model Size vs. Performance\n● with skills  ▪ without skills  (dashed = uplift gap)",
             fontweight="bold", fontsize=13)
ax.set_xlim(0, 22)
ax.set_ylim(-0.05, 1.1)
ax.set_xticks([2, 4, 9, 20])
ax.set_xticklabels(["2B", "4B", "9B", "20B"])
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(f"{OUT}/3_size_vs_score.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 3_size_vs_score.png")

# ── 4. Latency Comparison ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(MODELS))
w = 0.35
lat_with    = [LATENCY_WITH[m]    / 1000 for m in MODELS]
lat_without = [LATENCY_WITHOUT[m] / 1000 for m in MODELS]
ax.bar(x - w/2, lat_with,    w, label="With Skills",    color="#2ecc71", alpha=0.9)
ax.bar(x + w/2, lat_without, w, label="Without Skills", color="#e74c3c", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[m] for m in MODELS], fontsize=9)
ax.set_ylabel("Avg Latency per Test Case (seconds)", fontsize=11)
ax.set_title("Inference Latency by Model\n(With Skills includes multi-turn tool calls)", fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/4_latency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 4_latency.png")

# ── 5. Score Heatmap ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
row_labels = [SHORT[m].replace("\n"," ") for m in MODELS]
col_labels = ["E2E\nw/ skills", "E2E\nw/o skills", "SkillSel\nw/ skills", "SkillSel\nw/o skills"]
matrix = np.array([
    [E2E_WITH[m], E2E_WITHOUT[m], SEL_WITH[m], SEL_WITHOUT[m]]
    for m in MODELS
])
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(row_labels, fontsize=10)
for i in range(len(MODELS)):
    for j in range(4):
        v = matrix[i, j]
        color = "white" if v < 0.35 or v > 0.75 else "black"
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
ax.set_title("Score Heatmap — All Models × All Conditions", fontweight="bold", fontsize=13)
plt.tight_layout()
fig.savefig(f"{OUT}/5_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 5_heatmap.png")

# ── 6. Qwen Family: Diminishing Returns ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
qwen = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
qsize = [2, 4, 9]
qwith    = [E2E_WITH[m]    for m in qwen]
qwithout = [E2E_WITHOUT[m] for m in qwen]
qdelta   = [E2E_DELTA[m]   for m in qwen]

ax.plot(qsize, qwith,    "o-", color="#2ecc71", linewidth=2.5, markersize=10, label="With Skills")
ax.plot(qsize, qwithout, "s--",color="#e74c3c", linewidth=2.5, markersize=9,  label="Without Skills", alpha=0.8)

for x_, yw, yo in zip(qsize, qwith, qwithout):
    ax.annotate(f"{yw:.0%}", (x_, yw), textcoords="offset points", xytext=(6, 5), color="#2ecc71", fontweight="bold")
    ax.annotate(f"{yo:.0%}", (x_, yo), textcoords="offset points", xytext=(6,-14), color="#e74c3c", fontweight="bold")
    ax.annotate(f"Δ{yw-yo:+.0%}", (x_, (yw+yo)/2), textcoords="offset points",
                xytext=(14, 0), color="#555", fontsize=9)

ax.fill_between(qsize, qwithout, qwith, alpha=0.1, color="#2ecc71", label="Uplift gap")
ax.set_xticks(qsize)
ax.set_xticklabels(["2B", "4B", "9B"])
ax.set_xlabel("Model Size (Parameters)", fontsize=12)
ax.set_ylabel("End-to-End Score", fontsize=12)
ax.set_ylim(0.1, 1.1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"])
ax.set_title("Qwen3.5 Family: Diminishing Returns\nas Model Size Increases", fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
fig.savefig(f"{OUT}/6_qwen_diminishing_returns.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 6_qwen_diminishing_returns.png")

print(f"\nAll charts saved to {OUT}/")
