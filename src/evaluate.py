"""
Evaluates stage 1 alone against the full two-stage (boosted) pipeline on the
held-out validation subjects, and reports Dice per class plus the Dice/size
ratio that the hackathon grades on.

This script is deliberately blunt: if stage 2 does not help, the delta column
will say so. With only 8 training subjects that is a real possibility, and it
is better to find out here than in front of a jury.

Usage:
    python evaluate.py
"""
import gc
import os

import keras
import numpy as np

from config import Config
from dataset import load_all_subjects, split_train_val
from model import count_params
from predict import sliding_window_predict, dice_score
from stage2 import two_stage_predict


def main():
    cfg = Config()

    stage1 = keras.models.load_model(
        os.path.join(cfg.checkpoint_dir, "stage1.keras"), compile=False)
    stage2_path = os.path.join(cfg.checkpoint_dir, "stage2.keras")
    stage2 = keras.models.load_model(stage2_path, compile=False) if os.path.isfile(stage2_path) else None

    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    _, val_subjects = split_train_val(subjects, cfg)

    stage1_scores = {n: [] for n in cfg.class_names}
    two_stage_scores = {n: [] for n in cfg.class_names}

    for subject in val_subjects:
        probs1 = sliding_window_predict(stage1, subject, cfg)
        pred1 = np.argmax(probs1, axis=-1)
        for cid, name in enumerate(cfg.class_names):
            stage1_scores[name].append(dice_score(pred1, subject.label, cid))

        if stage2 is not None:
            probs2 = two_stage_predict(stage1, stage2, subject, cfg)
            pred2 = np.argmax(probs2, axis=-1)
            for cid, name in enumerate(cfg.class_names):
                two_stage_scores[name].append(dice_score(pred2, subject.label, cid))
            del probs2, pred2

        print(f"subject {subject.subject_id} evaluated")
        del probs1, pred1
        gc.collect()

    m1 = {k: float(np.mean(v)) for k, v in stage1_scores.items()}
    p1 = count_params(stage1)

    if stage2 is None:
        print("\nNo stage-2 checkpoint found; reporting stage 1 only.")
        for name in cfg.class_names:
            print(f"  {name}: {m1[name]:.4f}")
        return

    m2 = {k: float(np.mean(v)) for k, v in two_stage_scores.items()}
    p2 = count_params(stage2)

    print(f"\n{'class':<15}{'stage 1':>10}{'two-stage':>12}{'delta':>10}")
    print("-" * 47)
    for name in cfg.class_names:
        print(f"{name:<15}{m1[name]:>10.4f}{m2[name]:>12.4f}{m2[name] - m1[name]:>+10.4f}")

    tissue = [n for n in cfg.class_names if n != "background"]
    t1 = float(np.mean([m1[n] for n in tissue]))
    t2 = float(np.mean([m2[n] for n in tissue]))
    print("-" * 47)
    print(f"{'tissue mean':<15}{t1:>10.4f}{t2:>12.4f}{t2 - t1:>+10.4f}")

    print(f"\nParameters: stage 1 {p1:,} | stage 2 {p2:,} | combined {p1 + p2:,}")
    print(f"Tissue Dice per million params: "
          f"stage 1 {t1 / (p1 / 1e6):.3f} -> two-stage {t2 / ((p1 + p2) / 1e6):.3f}")

    if t2 <= t1:
        print("\nNOTE: stage 2 did not improve tissue Dice. Worth checking: more "
              "stage-1 epochs, a smaller stage 2, or a lower error_oversample_ratio.")


if __name__ == "__main__":
    main()
