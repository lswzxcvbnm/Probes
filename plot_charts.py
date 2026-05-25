import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150

ppl_df = pd.read_csv("processed_datasets/ppl_metrics.csv")
saplma_df = pd.read_csv("processed_datasets/qwen2_saplma_metrics.csv")
token_df = pd.read_csv("processed_datasets/qwen2_token_position_metrics.csv")

topic_display = {
    "cities": "Cities",
    "inventions": "Inventions",
    "elements": "Elements",
    "animals": "Animals",
    "companies": "Companies",
    "facts": "Facts",
}

# ============================================================
# Figure 1: PPL vs SAPLMA (best layer=18) comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ppl_topics = ppl_df["dataset"].str.replace("_true_false", "", regex=False)
ppl_acc = ppl_df["accuracy"].values
ppl_auc = ppl_df["auc"].values

saplma_l18 = saplma_df[saplma_df["layer"] == 18].sort_values("test_topic")
saplma_acc = saplma_l18["avg_accuracy"].values
saplma_auc = saplma_l18["avg_auc"].values

topics = [topic_display[t] for t in saplma_l18["test_topic"].values]
x = np.arange(len(topics))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width / 2, ppl_acc, width, label="PPL", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width / 2, saplma_acc, width, label="SAPLMA (L18)", color="#DD8452", edgecolor="white")
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Accuracy: PPL vs SAPLMA", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=30, ha="right")
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax = axes[1]
bars1 = ax.bar(x - width / 2, ppl_auc, width, label="PPL", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width / 2, saplma_auc, width, label="SAPLMA (L18)", color="#DD8452", edgecolor="white")
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("AUC: PPL vs SAPLMA", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=30, ha="right")
ax.legend(fontsize=10)
ax.set_ylim(0.4, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("processed_datasets/fig1_ppl_vs_saplma.png", bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# ============================================================
# Figure 2: SAPLMA layer analysis (accuracy & AUC across layers)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

layers = sorted(saplma_df["layer"].unique())
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

ax = axes[0]
for i, topic in enumerate(saplma_df["test_topic"].unique()):
    topic_data = saplma_df[saplma_df["test_topic"] == topic].sort_values("layer")
    ax.plot(topic_data["layer"], topic_data["avg_accuracy"], marker="o",
            label=topic_display[topic], color=colors[i], linewidth=2, markersize=6)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("SAPLMA: Accuracy vs Layer Depth", fontsize=13, fontweight="bold")
ax.set_xticks(layers)
ax.legend(fontsize=9, loc="lower left")
ax.set_ylim(0.5, 0.9)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
for i, topic in enumerate(saplma_df["test_topic"].unique()):
    topic_data = saplma_df[saplma_df["test_topic"] == topic].sort_values("layer")
    ax.plot(topic_data["layer"], topic_data["avg_auc"], marker="s",
            label=topic_display[topic], color=colors[i], linewidth=2, markersize=6)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("SAPLMA: AUC vs Layer Depth", fontsize=13, fontweight="bold")
ax.set_xticks(layers)
ax.legend(fontsize=9, loc="lower left")
ax.set_ylim(0.5, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("processed_datasets/fig2_saplma_layer_analysis.png", bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# ============================================================
# Figure 3: Token position analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

positions = ["first", "second", "second_to_last", "last"]
pos_display = ["First", "Second", "Second-to-Last", "Last"]
pos_colors = ["#C44E52", "#8172B2", "#CCB974", "#55A868"]

ax = axes[0]
x = np.arange(len(topics))
width = 0.18
for j, (pos, pos_d, col) in enumerate(zip(positions, pos_display, pos_colors)):
    pos_data = token_df[token_df["position"] == pos].sort_values("test_topic")
    acc_vals = pos_data["avg_accuracy"].values
    offset = (j - 1.5) * width
    bars = ax.bar(x + offset, acc_vals, width, label=pos_d, color=col, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Accuracy by Token Position (Layer 18)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=30, ha="right")
ax.legend(fontsize=9)
ax.set_ylim(0.3, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.2, axis="y")

ax = axes[1]
for j, (pos, pos_d, col) in enumerate(zip(positions, pos_display, pos_colors)):
    pos_data = token_df[token_df["position"] == pos].sort_values("test_topic")
    auc_vals = pos_data["avg_auc"].values
    offset = (j - 1.5) * width
    bars = ax.bar(x + offset, auc_vals, width, label=pos_d, color=col, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
ax.set_ylabel("AUC", fontsize=12)
ax.set_title("AUC by Token Position (Layer 18)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(topics, rotation=30, ha="right")
ax.legend(fontsize=9)
ax.set_ylim(0.3, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig("processed_datasets/fig3_token_position.png", bbox_inches="tight")
plt.close()
print("Figure 3 saved.")

# ============================================================
# Figure 4: Average performance across layers (SAPLMA)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

layer_avg = saplma_df.groupby("layer").agg(
    avg_acc=("avg_accuracy", "mean"),
    avg_auc=("avg_auc", "mean")
).reset_index()

ax.plot(layer_avg["layer"], layer_avg["avg_acc"], marker="o", linewidth=2.5,
        markersize=8, color="#4C72B0", label="Avg Accuracy")
ax.plot(layer_avg["layer"], layer_avg["avg_auc"], marker="s", linewidth=2.5,
        markersize=8, color="#DD8452", label="Avg AUC")
ax.fill_between(layer_avg["layer"], layer_avg["avg_acc"], layer_avg["avg_auc"],
                alpha=0.15, color="#8172B2")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("SAPLMA: Average Performance Across Layers", fontsize=13, fontweight="bold")
ax.set_xticks(layers)
ax.legend(fontsize=11)
ax.set_ylim(0.55, 0.95)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)

for _, row in layer_avg.iterrows():
    ax.annotate(f'{row["avg_acc"]:.3f}', (row["layer"], row["avg_acc"]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color="#4C72B0")
    ax.annotate(f'{row["avg_auc"]:.3f}', (row["layer"], row["avg_auc"]),
                textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9, color="#DD8452")

plt.tight_layout()
plt.savefig("processed_datasets/fig4_saplma_avg_by_layer.png", bbox_inches="tight")
plt.close()
print("Figure 4 saved.")

# ============================================================
# Figure 5: Token position average (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

pos_avg = token_df.groupby("position").agg(
    avg_acc=("avg_accuracy", "mean"),
    avg_auc=("avg_auc", "mean")
).reindex(positions).reset_index()

x = np.arange(len(positions))
width = 0.35
bars1 = ax.bar(x - width / 2, pos_avg["avg_acc"], width, label="Avg Accuracy", color="#4C72B0", edgecolor="white")
bars2 = ax.bar(x + width / 2, pos_avg["avg_auc"], width, label="Avg AUC", color="#DD8452", edgecolor="white")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Token Position: Average Performance (Layer 18)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(pos_display, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.2, axis="y")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("processed_datasets/fig5_position_avg.png", bbox_inches="tight")
plt.close()
print("Figure 5 saved.")

print("\nAll figures generated successfully!")
