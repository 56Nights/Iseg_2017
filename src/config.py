"""
Central configuration for the iSeg-2017 frugal 3D U-Net project.

Everything that affects the parameter count / Dice tradeoff lives here,
so it's easy to sweep for the hackathon's "Dice / network size" criterion.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # --- Paths ---
    train_dir: str = "data/train_raw"
    test_dir: str = "data/test_raw"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    # --- Data ---
    # iSeg labels on disk are {0, 10, 150, 250} -> remapped to {0, 1, 2, 3}
    # 0 = background, 1 = CSF, 2 = gray matter, 3 = white matter
    raw_label_values: Tuple[int, ...] = (0, 10, 150, 250)
    num_classes: int = 4
    class_names: Tuple[str, ...] = ("background", "csf", "gray_matter", "white_matter")

    # Held-out subjects for validation (rest of the 10 labeled subjects train)
    val_subject_ids: Tuple[int, ...] = (9, 10)

    # --- Patches ---
    patch_size: int = 64          # cubic patch, must be divisible by 2**n_levels
    patches_per_volume: int = 40  # sampled per epoch per training subject
    foreground_oversample_ratio: float = 0.8  # fraction of patches centered on non-background voxels

    # --- Model (kept small on purpose: hackathon criterion is Dice / params) ---
    base_filters: int = 16        # filters at the first encoder level
    n_levels: int = 3             # number of downsampling steps (4 resolution stages total)
    groupnorm_groups: int = 8

    # --- Training ---
    batch_size: int = 2           # 3D volumes at 64^3 are memory-hungry; keep this small
    epochs: int = 150
    learning_rate: float = 1e-3
    dice_loss_weight: float = 0.7
    ce_loss_weight: float = 0.3

    # --- Inference ---
    sliding_window_overlap: float = 0.5  # fraction of patch_size to step by

    # --- Stage 2 (boosting-style error refinement) ---
    stage2_base_filters: int = 8   # stage 2 only fixes residual errors, so it can be smaller
    stage2_n_levels: int = 3
    stage2_epochs: int = 100
    error_threshold: float = 0.3        # voxel counts as a "stage 1 mistake" above this
    max_error_fraction: float = 0.25    # if more than this share of brain qualifies, keep only the worst
    error_oversample_ratio: float = 0.65  # fraction of patches centered on mistakes
    error_weight_scale: float = 4.0     # loss weight = 1 + scale * error
