from matplotlib import pyplot as plt
import numpy as np
def plot_learning_curves(df, metric, title, file):
    plt.figure(figsize=(7,4))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("n_sampled")
        plt.plot(sub["n_sampled"], sub[metric], marker="o", label=method.capitalize())
    plt.xlabel("# labelled samples (|L|)")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(title); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(f'{file}.png'); plt.show()

random_test_f1_baseline = 0.475
active_test_f1_baseline = 0.476
random_test_f1_pretrained = 0.455
active_test_f1_pretrained = 0.491

random_test_recall_baseline = 0.607
active_test_recall_baseline = 0.614
random_test_recall_pretrained = 0.559
active_test_recall_pretrained = 0.582

random_test_prec_baseline = 0.437
active_test_prec_baseline = 0.421
random_test_prec_pretrained = 0.419
active_test_prec_pretrained = 0.451

# Data arranged as [Random, Active] per subplot
scores = {
    "F1": [random_test_f1_baseline, active_test_f1_baseline, random_test_f1_pretrained, active_test_f1_pretrained],
    "Recall": [random_test_recall_baseline, active_test_recall_baseline, random_test_recall_pretrained, active_test_recall_pretrained],
    "Precision": [random_test_prec_baseline, active_test_prec_baseline, random_test_prec_pretrained,active_test_prec_pretrained],
}

fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True, constrained_layout=True)

labels = ["R-NP", "AL-NP", "R-P", "AL-P"]
colours = ["tab:orange", "tab:blue", "tab:green", "tab:purple"]
x = np.arange(len(labels))
width = 0.8

bars_all = []
for ax, (title, vals) in zip(axes, scores.items()):
    bars = ax.bar(x, vals, width, color = colours, label=labels)
    bars_all.extend(bars)

    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{h:.3f}", ha='center', va='bottom', fontsize=9)

handles = axes[0].containers[0]
fig.legend(handles, labels, loc="center right", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))

axes[0].set_ylabel("Test Macro Evaluation")
plt.savefig("test_macrometrics_results.png", dpi=300, bbox_inches="tight")
plt.show()