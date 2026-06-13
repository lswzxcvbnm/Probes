"""
Generate publication-quality figures for the ACL experiment report.

Generates:
  fig_report_1_ppl_vs_saplma.png   - PPL vs SAPLMA comparison (grouped bar)
  fig_report_2_layer_analysis.png  - SAPLMA layer depth analysis (line + bar)
  fig_report_3_token_position.png  - Token position analysis
  fig_report_4_head_heatmap.png    - 336-head AUC heatmap (top-5 highlighted)
  fig_report_5_method_comparison.png - All methods comparison on TriviaQA
  fig_report_6_head_distribution.png - Top-5 head layer distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

# ── Global style ──────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

TOPIC_DISPLAY = {
    "cities": "Cities", "inventions": "Inventions",
    "elements": "Elements", "animals": "Animals",
    "companies": "Companies", "facts": "Facts",
}
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#CCB974']

# ── Load data ─────────────────────────────────────────────────
ppl_df = pd.read_csv("processed_datasets/ppl_metrics.csv")
saplma_df = pd.read_csv("processed_datasets/qwen2_saplma_metrics.csv")
token_df = pd.read_csv("processed_datasets/qwen2_token_position_metrics.csv")
head_auc_df = pd.read_csv("processed_datasets/head_aucs.csv")
probe_df = pd.read_csv("processed_datasets/head_probe_metrics.csv")
with open("processed_datasets/top5_heads.json") as f:
    top5 = json.load(f)

output_dir = "processed_datasets"

# ============================================================
# Figure 1: PPL vs SAPLMA (Layer 18) — Accuracy & AUC by topic
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ppl_topics = ppl_df["dataset"].str.replace("_true_false", "", regex=False)
ppl_acc = ppl_df["accuracy"].values
ppl_auc = ppl_df["auc"].values

saplma_l18 = saplma_df[saplma_df["layer"] == 18].sort_values("test_topic")
saplma_acc = saplma_l18["avg_accuracy"].values
saplma_auc = saplma_l18["avg_auc"].values
topics = [TOPIC_DISPLAY[t] for t in saplma_l18["test_topic"].values]

x = np.arange(len(topics))
width = 0.35

for ax, metric_name, ppl_vals, saplma_vals in [
    (axes[0], "Accuracy", ppl_acc, saplma_acc),
    (axes[1], "AUC", ppl_auc, saplma_auc),
]:
    b1 = ax.bar(x - width/2, ppl_vals, width, label="PPL",
                color=COLORS[0], edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, saplma_vals, width, label="SAPLMA (L18)",
                color=COLORS[1], edgecolor="white", linewidth=0.5)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_1_ppl_vs_saplma.png")
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: SAPLMA Layer Analysis — Average Acc/AUC + per-topic AUC
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

layers = sorted(saplma_df["layer"].unique())
layer_avg = saplma_df.groupby("layer").agg(
    avg_acc=("avg_accuracy", "mean"),
    avg_auc=("avg_auc", "mean"),
).reset_index()

# Left: average performance
ax = axes[0]
ax.plot(layer_avg["layer"], layer_avg["avg_acc"], marker="o", linewidth=2,
        markersize=7, color=COLORS[0], label="Avg Accuracy")
ax.plot(layer_avg["layer"], layer_avg["avg_auc"], marker="s", linewidth=2,
        markersize=7, color=COLORS[1], label="Avg AUC")
ax.fill_between(layer_avg["layer"], layer_avg["avg_acc"], layer_avg["avg_auc"],
                alpha=0.12, color='#8172B2')
ax.set_xlabel("Layer")
ax.set_ylabel("Score")
ax.set_title("Average Performance vs Layer Depth")
ax.set_xticks(layers)
ax.legend()
ax.set_ylim(0.55, 0.92)
for _, row in layer_avg.iterrows():
    ax.annotate(f'{row["avg_acc"]:.3f}', (row["layer"], row["avg_acc"]),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color=COLORS[0])
    ax.annotate(f'{row["avg_auc"]:.3f}', (row["layer"], row["avg_auc"]),
                textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=COLORS[1])

# Right: per-topic AUC across layers
ax = axes[1]
for i, topic in enumerate(saplma_df["test_topic"].unique()):
    td = saplma_df[saplma_df["test_topic"] == topic].sort_values("layer")
    ax.plot(td["layer"], td["avg_auc"], marker="o", linewidth=1.5,
            markersize=5, color=COLORS[i], label=TOPIC_DISPLAY[topic])
ax.set_xlabel("Layer")
ax.set_ylabel("AUC")
ax.set_title("Per-Topic AUC vs Layer Depth")
ax.set_xticks(layers)
ax.legend(fontsize=8, loc="lower left", ncol=2)
ax.set_ylim(0.6, 0.95)

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_2_layer_analysis.png")
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Token Position Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

positions = ["first", "second", "second_to_last", "last"]
pos_display = ["First", "Second", "2nd-to-Last", "Last"]
pos_colors = [COLORS[3], COLORS[4], COLORS[5], COLORS[2]]

# Left: per-topic AUC grouped by position
ax = axes[0]
x = np.arange(len(topics))
width = 0.18
for j, (pos, pos_d, col) in enumerate(zip(positions, pos_display, pos_colors)):
    pos_data = token_df[token_df["position"] == pos].sort_values("test_topic")
    auc_vals = pos_data["avg_auc"].values
    offset = (j - 1.5) * width
    bars = ax.bar(x + offset, auc_vals, width, label=pos_d, color=col, edgecolor="white", linewidth=0.5)
ax.set_ylabel("AUC")
ax.set_title("AUC by Token Position (Layer 18)")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=30, ha="right")
ax.legend(fontsize=8)
ax.set_ylim(0.35, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

# Right: average performance bar chart
ax = axes[1]
pos_avg = token_df.groupby("position").agg(
    avg_acc=("avg_accuracy", "mean"),
    avg_auc=("avg_auc", "mean"),
).reindex(positions).reset_index()

x = np.arange(len(positions))
width = 0.35
b1 = ax.bar(x - width/2, pos_avg["avg_acc"], width, label="Avg Accuracy",
            color=COLORS[0], edgecolor="white")
b2 = ax.bar(x + width/2, pos_avg["avg_auc"], width, label="Avg AUC",
            color=COLORS[1], edgecolor="white")
ax.set_ylabel("Score")
ax.set_title("Average Performance by Position")
ax.set_xticks(x)
ax.set_xticklabels(pos_display)
ax.legend()
ax.set_ylim(0.3, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_3_token_position.png")
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: Head AUC Heatmap (with top-5 highlighted)
# ============================================================
num_layers = 28
num_heads = 12
auc_matrix = np.full((num_layers, num_heads), np.nan)
for _, row in head_auc_df.iterrows():
    auc_matrix[int(row["layer"]), int(row["head"])] = row["auc"]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(auc_matrix, cmap='YlOrRd', aspect='auto', vmin=0.48, vmax=0.80)

# Highlight top-5 heads
top5_heads = [(d[0], d[1]) for d in top5["selected_heads"]]
for l, h in top5_heads:
    rect = plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False,
                          edgecolor='blue', linewidth=2.5)
    ax.add_patch(rect)

ax.set_xlabel("Attention Head")
ax.set_ylabel("Layer")
ax.set_title("Per-Head Classifier AUC on Validation Set")
ax.set_xticks(range(num_heads))
ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
ax.set_yticks(range(num_layers))
ax.set_yticklabels([f"L{l}" for l in range(num_layers)], fontsize=7)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("AUC")

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_4_head_heatmap.png")
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: All Methods Comparison on TriviaQA Test Set
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

# Select key methods for comparison
key_methods = ['PPL', 'SAPLMA-L18', 'Single-L15_H6', 'Summed-Heads', 'Voting-Ensemble']
method_labels = ['PPL', 'SAPLMA\n(L18)', 'Single\nL15\_H6', 'Summed\nHeads', 'Voting\nEnsemble']
method_colors = [COLORS[0], COLORS[1], COLORS[4], COLORS[2], COLORS[3]]

x = np.arange(len(metric_labels))
width = 0.15

for i, (method, label, color) in enumerate(zip(key_methods, method_labels, method_colors)):
    row = probe_df[probe_df["method"] == method].iloc[0]
    vals = [row[m] for m in metrics_names]
    offset = (i - 2) * width
    bars = ax.bar(x + offset, vals, width, label=label, color=color,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

ax.set_ylabel("Score")
ax.set_title("Hallucination Detection Performance on TriviaQA Test Set (N=203)")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_5_method_comparison.png")
plt.close()
print("Figure 5 saved.")

# ============================================================
# Figure 6: Top-5 Head Layer Distribution + AUC bar
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))

head_keys = list(top5["head_details"].keys())
head_aucs = [top5["head_details"][k]["auc"] for k in head_keys]
head_layers = [top5["selected_heads"][i][0] for i in range(5)]

bars = ax.bar(range(5), head_aucs, color=[COLORS[i] for i in range(5)],
              edgecolor="white", linewidth=0.5)
ax.set_xticks(range(5))
ax.set_xticklabels([f"{k}\n(L{top5['selected_heads'][i][0]})"
                     for i, k in enumerate(head_keys)], fontsize=9)
ax.set_ylabel("Validation AUC")
ax.set_title("Top-5 Attention Heads by Validation AUC")
ax.set_ylim(0.70, 0.82)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

for bar, val in zip(bars, head_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dir}/fig_report_6_head_distribution.png")
plt.close()
print("Figure 6 saved.")

print("\nAll report figures generated successfully!")
