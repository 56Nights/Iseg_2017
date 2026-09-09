"""
Runs the trained pipeline on the 13 unlabeled test subjects and writes
predictions in the challenge's label scheme (0 / 10 / 150 / 250).

Falls back to stage 1 alone if no stage-2 checkpoint exists.

Usage:
    python export_predictions.py
"""
import gc
import os

import keras
import numpy as np

from config import Config
from dataset import load_subject, discover_subject_ids
from predict import sliding_window_predict, save_prediction
from stage2 import two_stage_predict


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    stage1 = keras.models.load_model(
        os.path.join(cfg.checkpoint_dir, "stage1.keras"), compile=False)
    stage2_path = os.path.join(cfg.checkpoint_dir, "stage2.keras")
    stage2 = keras.models.load_model(stage2_path, compile=False) if os.path.isfile(stage2_path) else None
    print("Using", "two-stage pipeline" if stage2 is not None else "stage 1 only")

    for sid in discover_subject_ids(cfg.test_dir, has_labels=False):
        subject = load_subject(cfg.test_dir, sid, cfg, has_labels=False)
        if stage2 is not None:
            probs = two_stage_predict(stage1, stage2, subject, cfg)
        else:
            probs = sliding_window_predict(stage1, subject, cfg)
        pred = np.argmax(probs, axis=-1)

        out_path = os.path.join(cfg.output_dir, f"subject-{sid}-prediction.nii.gz")
        save_prediction(pred, os.path.join(cfg.test_dir, f"subject-{sid}-T1.hdr"), out_path, cfg)
        print(f"  subject {sid} -> {out_path}")

        del subject, probs, pred
        gc.collect()

    print("\nAll predictions written to", cfg.output_dir)


if __name__ == "__main__":
    main()
