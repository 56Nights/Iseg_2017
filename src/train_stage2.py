"""
Trains stage 2: the boosting-style refinement network.

Requires stage 1 to have been trained first (checkpoints/stage1.keras).

Usage:
    python train_stage2.py
"""
import os

import keras

from config import Config
from dataset import load_all_subjects, split_train_val
from model import build_unet3d, count_params
from stage2 import build_stage2_subjects, make_stage2_dataset, make_weighted_loss


def main():
    cfg = Config()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    stage1_path = os.path.join(cfg.checkpoint_dir, "stage1.keras")
    if not os.path.isfile(stage1_path):
        raise FileNotFoundError(
            f"No stage-1 checkpoint at {stage1_path}. Run train_stage1.py first."
        )
    stage1 = keras.models.load_model(stage1_path, compile=False)
    print(f"Loaded stage 1 ({count_params(stage1):,} params)")

    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    train_subjects, val_subjects = split_train_val(subjects, cfg)

    # The boosting step: measure where stage 1 is wrong on the labeled data.
    print("Computing stage-1 residual maps (train)...")
    s2_train = build_stage2_subjects(stage1, train_subjects, cfg)
    print("Computing stage-1 residual maps (val)...")
    s2_val = build_stage2_subjects(stage1, val_subjects, cfg)

    ps = cfg.patch_size
    cfg_s2 = Config(**cfg.__dict__)
    cfg_s2.base_filters = cfg.stage2_base_filters
    cfg_s2.n_levels = cfg.stage2_n_levels

    # Stage 2 sees T1, T2 AND stage 1's class probabilities.
    stage2 = build_unet3d(cfg_s2, input_shape=(ps, ps, ps, 2 + cfg.num_classes))
    print(f"Stage 2 parameters: {count_params(stage2):,}")
    print(f"Combined: {count_params(stage1) + count_params(stage2):,}")

    stage2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=make_weighted_loss(cfg),
    )

    train_ds = make_stage2_dataset(s2_train, cfg)
    val_ds = make_stage2_dataset(s2_val, cfg)

    steps = max(1, (len(train_subjects) * cfg.patches_per_volume) // cfg.batch_size)
    val_steps = max(1, (len(val_subjects) * cfg.patches_per_volume) // cfg.batch_size // 2)

    stage2.fit(
        train_ds,
        steps_per_epoch=steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=cfg.stage2_epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                os.path.join(cfg.checkpoint_dir, "stage2.keras"),
                monitor="val_loss", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=25, restore_best_weights=True),
            keras.callbacks.CSVLogger(os.path.join(cfg.checkpoint_dir, "stage2_log.csv")),
        ],
    )

    stage2.save(os.path.join(cfg.checkpoint_dir, "stage2_final.keras"))
    print("Stage 2 done. Best model at checkpoints/stage2.keras")


if __name__ == "__main__":
    main()
