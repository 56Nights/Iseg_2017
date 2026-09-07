"""
Trains the frugal 3D U-Net on iSeg-2017.

Usage:
    python train.py

Adjust config.py to change patch size, model width/depth, epochs, etc.
On a Colab GPU, this should train at a reasonable pace; on CPU it will be
slow (3D convs are expensive) but will still run correctly on a small
number of epochs for testing.
"""
import os

import keras
import tensorflow as tf

from config import Config
from dataset import load_all_subjects, split_train_val, make_tf_dataset
from model import build_unet3d, count_params
from losses import make_combined_loss, make_metrics


def steps_per_epoch(subjects, cfg: Config) -> int:
    total_patches = len(subjects) * cfg.patches_per_volume
    return max(1, total_patches // cfg.batch_size)


def main():
    cfg = Config()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print("Loading subjects...")
    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    train_subjects, val_subjects = split_train_val(subjects, cfg)
    print(f"Train subjects: {[s.subject_id for s in train_subjects]}")
    print(f"Val subjects:   {[s.subject_id for s in val_subjects]}")

    train_ds = make_tf_dataset(train_subjects, cfg)
    val_ds = make_tf_dataset(val_subjects, cfg)

    model = build_unet3d(cfg)
    print(f"Model parameter count: {count_params(model):,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=make_combined_loss(cfg),
        metrics=make_metrics(cfg),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(cfg.checkpoint_dir, "stage1.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=25, restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(os.path.join(cfg.checkpoint_dir, "stage1_log.csv")),
    ]

    spe = steps_per_epoch(train_subjects, cfg)
    val_steps = max(1, steps_per_epoch(val_subjects, cfg) // 2)

    history = model.fit(
        train_ds,
        steps_per_epoch=spe,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=cfg.epochs,
        callbacks=callbacks,
    )

    model.save(os.path.join(cfg.checkpoint_dir, "stage1_final.keras"))
    print("Training complete. Best stage-1 model saved to checkpoints/stage1.keras")
    return history


if __name__ == "__main__":
    main()
