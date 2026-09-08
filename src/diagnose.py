"""
Per-subject diagnostic for the boosting stage: of the voxels stage 2 changed
relative to stage 1, how many did it fix vs. break?

Aggregate Dice (see evaluate.py) can hide this — a small positive average can
come from a consistent small improvement, or from one subject improving while
another gets worse. This script reports both outcomes per validation subject
so the two numbers can be read together.

Usage:
    python diagnose.py
"""
import os

import keras
import numpy as np

from config import Config
from dataset import load_all_subjects, split_train_val
from predict import sliding_window_predict
from stage2 import two_stage_predict


def main():
    cfg = Config()

    stage1 = keras.models.load_model(
        os.path.join(cfg.checkpoint_dir, "stage1.keras"), compile=False)
    stage2_path = os.path.join(cfg.checkpoint_dir, "stage2.keras")
    if not os.path.isfile(stage2_path):
        raise FileNotFoundError(
            f"No stage-2 checkpoint at {stage2_path}. Run train_stage2.py first."
        )
    stage2 = keras.models.load_model(stage2_path, compile=False)

    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    _, val_subjects = split_train_val(subjects, cfg)

    total_fixed, total_broke = 0, 0
    for vs in val_subjects:
        p1 = np.argmax(sliding_window_predict(stage1, vs, cfg), -1)
        p2 = np.argmax(two_stage_predict(stage1, stage2, vs, cfg), -1)
        gt = vs.label

        changed = p1 != p2
        fixed = changed & (p1 != gt) & (p2 == gt)   # stage 1 wrong -> stage 2 right
        broke = changed & (p1 == gt) & (p2 != gt)   # stage 1 right -> stage 2 wrong
        still = changed & (p1 != gt) & (p2 != gt)   # wrong both ways, differently

        net = int(fixed.sum()) - int(broke.sum())
        total_fixed += int(fixed.sum())
        total_broke += int(broke.sum())

        print(f"Subject {vs.subject_id}")
        print(f"  changed: {int(changed.sum()):,} | fixed: {int(fixed.sum()):,} | "
              f"broke: {int(broke.sum()):,} | still wrong: {int(still.sum()):,}")
        print(f"  net: {net:+,}")

    print(f"\nTotal across validation subjects: fixed {total_fixed:,}, broke {total_broke:,}, "
          f"net {total_fixed - total_broke:+,}")
    if total_fixed <= total_broke:
        print("Stage 2 did not net-improve the validation subjects at the voxel level.")


if __name__ == "__main__":
    main()
