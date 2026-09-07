"""
Loading, preprocessing, and patch sampling for the iSeg-2017 dataset.

iSeg-2017 gives you, per subject:
    subject-N-T1.hdr/.img      - T1-weighted MRI volume
    subject-N-T2.hdr/.img      - T2-weighted MRI volume
    subject-N-label.hdr/.img   - ground-truth labels (training subjects only)

Volumes are Analyze-format (.hdr/.img pairs), shape (144, 192, 256), 1mm
isotropic. We stack T1 and T2 as two input channels since they carry
complementary contrast for the gray/white matter distinction (T1 and T2
brighten/darken tissues differently, which is exactly why the challenge
provides both).
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import tensorflow as tf

from config import Config


@dataclass
class Subject:
    subject_id: int
    t1: np.ndarray            # (D, H, W) float32, z-score normalized
    t2: np.ndarray            # (D, H, W) float32, z-score normalized
    label: np.ndarray = None  # (D, H, W) int32, values in {0,1,2,3}, or None for test subjects


def _load_volume(path_no_ext: str) -> np.ndarray:
    """Loads a .hdr/.img Analyze volume and squeezes the trailing singleton dim."""
    img = nib.load(path_no_ext + ".hdr")
    data = img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    return data


def _zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    Normalizes intensities within the brain mask (nonzero voxels) only.
    MRI intensities aren't on a fixed scale across scans/scanners the way
    natural images are, so per-volume normalization is standard practice.
    """
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std() + 1e-8
    normed = np.zeros_like(volume)
    normed[mask] = (volume[mask] - mean) / std
    return normed


def _remap_labels(raw_label: np.ndarray, raw_values: Tuple[int, ...]) -> np.ndarray:
    """Maps {0, 10, 150, 250} -> {0, 1, 2, 3}."""
    remapped = np.zeros_like(raw_label, dtype=np.int32)
    for new_idx, raw_val in enumerate(raw_values):
        remapped[raw_label == raw_val] = new_idx
    return remapped


def discover_subject_ids(data_dir: str, has_labels: bool) -> List[int]:
    pattern = os.path.join(data_dir, "subject-*-T1.hdr")
    ids = []
    for path in glob.glob(pattern):
        match = re.search(r"subject-(\d+)-T1\.hdr", os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def load_subject(data_dir: str, subject_id: int, cfg: Config, has_labels: bool = True) -> Subject:
    base = os.path.join(data_dir, f"subject-{subject_id}")
    t1 = _zscore_normalize(_load_volume(base + "-T1"))
    t2 = _zscore_normalize(_load_volume(base + "-T2"))
    label = None
    if has_labels:
        raw_label = _load_volume(base + "-label")
        label = _remap_labels(raw_label, cfg.raw_label_values)
    return Subject(subject_id=subject_id, t1=t1, t2=t2, label=label)


def load_all_subjects(data_dir: str, cfg: Config, has_labels: bool = True) -> List[Subject]:
    ids = discover_subject_ids(data_dir, has_labels)
    return [load_subject(data_dir, sid, cfg, has_labels) for sid in ids]


def split_train_val(subjects: List[Subject], cfg: Config) -> Tuple[List[Subject], List[Subject]]:
    val = [s for s in subjects if s.subject_id in cfg.val_subject_ids]
    train = [s for s in subjects if s.subject_id not in cfg.val_subject_ids]
    return train, val


def _sample_patch_origin(volume_shape, patch_size, center_voxel=None, rng=None):
    """Picks a valid top-left-front corner for a patch, optionally biased near a center voxel."""
    rng = rng or np.random
    origins = []
    for dim, size in zip(volume_shape, (patch_size,) * 3):
        max_start = dim - size
        if max_start <= 0:
            origins.append(0)
            continue
        if center_voxel is not None:
            c = center_voxel[len(origins)]
            lo = max(0, min(max_start, c - size // 2))
            origins.append(int(lo))
        else:
            origins.append(int(rng.randint(0, max_start + 1)))
    return tuple(origins)


def _extract_patch(volume: np.ndarray, origin: Tuple[int, int, int], patch_size: int) -> np.ndarray:
    d0, h0, w0 = origin
    patch = volume[d0:d0 + patch_size, h0:h0 + patch_size, w0:w0 + patch_size]
    # Pad if the volume is smaller than the patch along some axis
    pad = [(0, max(0, patch_size - patch.shape[i])) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant", constant_values=0)
    return patch


def patch_generator(subjects: List[Subject], cfg: Config, rng_seed: int = 0):
    """
    Yields (input_patch, label_patch) pairs forever, sampling randomly across
    subjects. A configurable fraction of patches are centered on a random
    foreground voxel (CSF/GM/WM) rather than a uniformly random location,
    because background makes up ~87% of voxels (see EDA) -- without this,
    most random patches would contain little or no tissue to learn from.
    """
    rng = np.random.RandomState(rng_seed)
    while True:
        subject = subjects[rng.randint(0, len(subjects))]
        use_foreground_center = rng.rand() < cfg.foreground_oversample_ratio

        center_voxel = None
        if use_foreground_center and subject.label is not None:
            fg_coords = np.argwhere(subject.label > 0)
            if len(fg_coords) > 0:
                center_voxel = fg_coords[rng.randint(0, len(fg_coords))]

        origin = _sample_patch_origin(subject.t1.shape, cfg.patch_size, center_voxel, rng)
        t1_patch = _extract_patch(subject.t1, origin, cfg.patch_size)
        t2_patch = _extract_patch(subject.t2, origin, cfg.patch_size)
        input_patch = np.stack([t1_patch, t2_patch], axis=-1).astype(np.float32)

        label_patch = _extract_patch(subject.label, origin, cfg.patch_size).astype(np.int32)
        label_patch = np.expand_dims(label_patch, axis=-1)

        yield input_patch, label_patch


def make_tf_dataset(subjects: List[Subject], cfg: Config, shuffle_buffer: int = 32) -> tf.data.Dataset:
    ps = cfg.patch_size
    output_signature = (
        tf.TensorSpec(shape=(ps, ps, ps, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(ps, ps, ps, 1), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: patch_generator(subjects, cfg),
        output_signature=output_signature,
    )
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
