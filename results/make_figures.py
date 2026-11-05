"""results/*.csv의 수치로 README 그림을 다시 그린다.

실행: python results/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GRAY = "#2a6fdb", "#e8702a", "#9aa0a6"
PHASES = ["Phase1_CIFAKE", "Phase3_AI_vs_Real", "Phase4_AI_vs_Human", "Phase5_Faces"]
LABELS = ["Phase1\nCIFAKE", "Phase3\nAI vs Real", "Phase4\nAI vs Human", "Phase5\nFaces"]


def bar_labels(ax, bars):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)


def synthbuster_chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, backbone in zip(axes, ["CNN", "ResNet18"]):
        d = df[df.backbone == backbone]
        ind = d[d.training == "individual"].set_index("phase").loc[PHASES, "test_accuracy"]
        cum = d[d.training == "cumulative"]["test_accuracy"].iloc[0]
        bars = ax.bar(LABELS + ["Cumulative\n1→3→4→5"], list(ind) + [cum],
                      color=[BLUE] * 4 + [ORANGE], width=0.6)
        bar_labels(ax, bars)
        ax.axhline(0.5, color=GRAY, ls="--", lw=1)
        ax.set_title(f"{backbone}: SynthBuster test accuracy")
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "synthbuster_accuracy.png", dpi=150)


def forgetting_chart(df):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(LABELS, df["val_accuracy_of_final_cumulative_cnn"], color=BLUE, width=0.6)
    bar_labels(ax, bars)
    ax.axhline(0.5, color=GRAY, ls="--", lw=1)
    ax.set_title("Final cumulative CNN on each phase's validation set")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "catastrophic_forgetting.png", dpi=150)


if __name__ == "__main__":
    synthbuster_chart(pd.read_csv(HERE / "synthbuster_results.csv"))
    forgetting_chart(pd.read_csv(HERE / "forgetting_cnn.csv"))
    print("saved:", *sorted(p.name for p in OUT.glob("*.png")))
