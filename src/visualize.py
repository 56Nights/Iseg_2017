"""
Figures for the report:
  - stage 1 vs two-stage vs ground truth, across the three anatomical planes,
    plus a panel highlighting exactly which voxels stage 2 changed
  - the stage-1 residual (error) map that stage 2 is pointed at

Usage:
    python visualize.py                # saves figures to outputs/
"""
import os

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from config import Config
from dataset import load_all_subjects, split_train_val
from predict import sliding_window_predict
from stage2 import compute_error_map, two_stage_predict

SEG_COLORS = ["#000000", "#1f77b4", "#2ca02c", "#f5c518"]  # bg, CSF, GM, WM
CLASS_LABELS = ["background", "CSF", "gray matter", "white matter"]


def _seg_cmap():
    cmap = ListedColormap(SEG_COLORS)
    return cmap, BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


def _take(vol, axis, idx):
    if axis == 0:
        return vol[idx, :, :].T
    if axis == 1:
        return vol[:, idx, :].T
    return vol[:, :, idx].T


def plot_stage_comparison(subject, pred1, pred2, save_path):
    cmap, norm = _seg_cmap()
    mids = [subject.t1.shape[i] // 2 for i in range(3)]
    plane_names = ["Sagittal", "Coronal", "Axial"]

    fig, axes = plt.subplots(3, 5, figsize=(21, 12))
    for row, (axis, mid, plane) in enumerate(zip(range(3), mids, plane_names)):
        t1s = _take(subject.t1, axis, mid)
        axes[row, 0].imshow(t1s, cmap="gray", origin="lower")
        axes[row, 0].set_ylabel(plane, fontsize=12, fontweight="bold")
        axes[row, 1].imshow(_take(pred1, axis, mid), cmap=cmap, norm=norm, origin="lower")
        axes[row, 2].imshow(_take(pred2, axis, mid), cmap=cmap, norm=norm, origin="lower")
        axes[row, 3].imshow(_take(subject.label, axis, mid), cmap=cmap, norm=norm, origin="lower")

        # Panel 5: what the boosting step actually changed
        axes[row, 4].imshow(t1s, cmap="gray", origin="lower")
        changed = np.ma.masked_where(
            _take(pred1, axis, mid) == _take(pred2, axis, mid), np.ones_like(t1s))
        axes[row, 4].imshow(changed, cmap=ListedColormap(["#ff2d55"]), origin="lower", alpha=0.85)

        if row == 0:
            for col, title in enumerate(
                    ["T1", "Stage 1", "Two-stage", "Ground truth", "Changed by stage 2"]):
                axes[0, col].set_title(title, fontsize=12)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEG_COLORS[i]) for i in range(1, 4)]
    fig.legend(handles, CLASS_LABELS[1:], loc="lower center", ncol=3, fontsize=11)
    fig.suptitle(f"Subject {subject.subject_id}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print("saved", save_path)


def plot_error_map(subject, probs1, cfg, save_path):
    err = compute_error_map(probs1, subject.label, cfg.num_classes)
    mid = subject.t1.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(subject.t1[:, :, mid].T, cmap="gray", origin="lower")
    axes[0].set_title("T1")
    axes[1].imshow(subject.label[:, :, mid].T, cmap="viridis", origin="lower")
    axes[1].set_title("Ground truth")
    im = axes[2].imshow(err[:, :, mid].T, cmap="inferno", origin="lower", vmin=0, vmax=1)
    axes[2].set_title("Stage-1 residual")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(f"Subject {subject.subject_id} — where stage 1 is wrong", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    print("saved", save_path)


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    stage1 = keras.models.load_model(
        os.path.join(cfg.checkpoint_dir, "stage1.keras"), compile=False)
    stage2_path = os.path.join(cfg.checkpoint_dir, "stage2.keras")
    stage2 = keras.models.load_model(stage2_path, compile=False) if os.path.isfile(stage2_path) else None

    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    _, val_subjects = split_train_val(subjects, cfg)
    subject = val_subjects[0]

    probs1 = sliding_window_predict(stage1, subject, cfg)
    plot_error_map(subject, probs1, cfg,
                   os.path.join(cfg.output_dir, f"error_map_subject{subject.subject_id}.png"))

    if stage2 is not None:
        probs2 = two_stage_predict(stage1, stage2, subject, cfg)
        plot_stage_comparison(subject, np.argmax(probs1, -1), np.argmax(probs2, -1),
                              os.path.join(cfg.output_dir, f"stages_subject{subject.subject_id}.png"))
    else:
        print("No stage-2 checkpoint; skipped the comparison figure.")


if __name__ == "__main__":
    main()
