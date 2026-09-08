"""
Stage 2: a boosting-style refinement U-Net.

Idea (analogous to gradient boosting): stage 1 makes a first attempt; we then
measure where it went wrong on the *training* subjects (we have labels there),
and train a second, smaller network whose job is specifically to fix those
mistakes. Like XGBoost, no ground truth is needed at inference -- stage 2 has
learned "inputs that look like this are where stage 1 typically errs", so it
just runs forward on new subjects.

Two mechanisms carry the "focus on the errors" idea:
  1. Error-biased patch sampling: training patches are preferentially centered
     on voxels stage 1 got wrong (but the patch itself is full-size, so normal
     surrounding anatomy is still seen, and a fraction of patches are drawn
     uniformly -- with only 8 training subjects, sampling *exclusively* from
     error regions would overfit).
  2. Error-weighted loss: within a patch, voxels stage 1 got wrong contribute
     more to stage 2's loss than voxels it already got right.

Stage 2's input is [T1, T2, stage1_class_probabilities], so it sees both the
raw evidence and stage 1's guess + confidence.
"""
import gc
from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf

from config import Config
from dataset import Subject, _extract_patch, _sample_patch_origin
from predict import sliding_window_predict


@dataclass
class Stage2Subject:
    """A training subject augmented with stage 1's output and error map."""
    subject: Subject
    stage1_probs: np.ndarray   # (D, H, W, num_classes) float32
    error_map: np.ndarray      # (D, H, W) float32 in [0, 1]


def compute_error_map(stage1_probs: np.ndarray, label: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Per-voxel "residual": 1 - (probability stage 1 assigned to the CORRECT class).

    0.0 = stage 1 was confident and right; 1.0 = confidently wrong.
    This is the direct analogue of a boosting residual, computed with the
    training labels we do have.
    """
    onehot = np.eye(num_classes, dtype=np.float32)[label]
    prob_of_true = np.sum(stage1_probs * onehot, axis=-1)
    return (1.0 - prob_of_true).astype(np.float32)


def build_stage2_subjects(stage1_model, subjects: List[Subject], cfg: Config) -> List[Stage2Subject]:
    """Runs stage 1 over each labeled subject and records its probs + error map."""
    out = []
    for subject in subjects:
        probs = sliding_window_predict(stage1_model, subject, cfg).astype(np.float32)
        err = compute_error_map(probs, subject.label, cfg.num_classes)
        print(f"  subject {subject.subject_id}: mean stage-1 error {err.mean():.4f}, "
              f"voxels above threshold {int((err > cfg.error_threshold).sum()):,}")
        out.append(Stage2Subject(subject=subject, stage1_probs=probs, error_map=err))
        gc.collect()
    return out


def select_error_coords(s2: Stage2Subject, cfg: Config) -> np.ndarray:
    """
    Picks the voxels stage 2 should focus on.

    Two guards matter here, both found by testing:
      1. Restrict to *brain* voxels. Background is ~87% of the volume; if we let
         background voxels count as "errors", the sampler just draws empty
         patches and learns nothing.
      2. Use an adaptive (percentile) threshold rather than only a fixed one.
         Early in training stage 1 is wrong nearly everywhere, so a fixed
         threshold selects every voxel and the "focus on errors" bias silently
         degenerates into uniform sampling.
    """
    brain = (s2.subject.label > 0) | (np.argmax(s2.stage1_probs, axis=-1) > 0)
    brain_coords = np.argwhere(brain)
    if len(brain_coords) == 0:
        return np.argwhere(s2.subject.label > 0)

    brain_errs = s2.error_map[brain]
    n_brain = len(brain_coords)
    max_keep = max(1, int(cfg.max_error_fraction * n_brain))

    above = brain_errs > cfg.error_threshold
    n_above = int(above.sum())

    if 0 < n_above <= max_keep:
        # The fixed threshold is informative: use it directly.
        return brain_coords[above]

    # Otherwise rank by error and keep the worst `max_keep`. argpartition is
    # tie-robust (a percentile cutoff silently selects nothing when many voxels
    # share the same error value, e.g. an untrained stage 1) and is O(n).
    worst = np.argpartition(brain_errs, n_brain - max_keep)[n_brain - max_keep:]
    return brain_coords[worst]


def stage2_patch_generator(s2_subjects: List[Stage2Subject], cfg: Config, rng_seed: int = 0):
    """
    Yields (input_patch, label_patch, weight_patch).

    input_patch: (P, P, P, 2 + num_classes) -- T1, T2, then stage 1's probabilities
    weight_patch: (P, P, P, 1) -- per-voxel loss weight derived from stage 1's error
    """
    rng = np.random.RandomState(rng_seed)
    # Precompute the high-error voxel coordinates once per subject (doing this
    # per-draw would be far too slow).
    error_coords = [select_error_coords(s2, cfg) for s2 in s2_subjects]

    while True:
        idx = rng.randint(0, len(s2_subjects))
        s2 = s2_subjects[idx]
        coords = error_coords[idx]

        # Bias -- but don't restrict -- toward stage 1's mistakes.
        center_voxel = None
        if rng.rand() < cfg.error_oversample_ratio and len(coords) > 0:
            center_voxel = coords[rng.randint(0, len(coords))]

        origin = _sample_patch_origin(s2.subject.t1.shape, cfg.patch_size, center_voxel, rng)

        t1_p = _extract_patch(s2.subject.t1, origin, cfg.patch_size)
        t2_p = _extract_patch(s2.subject.t2, origin, cfg.patch_size)
        prob_p = np.stack(
            [_extract_patch(s2.stage1_probs[..., c], origin, cfg.patch_size)
             for c in range(cfg.num_classes)],
            axis=-1,
        )
        x = np.concatenate([np.stack([t1_p, t2_p], axis=-1), prob_p], axis=-1).astype(np.float32)

        y = _extract_patch(s2.subject.label, origin, cfg.patch_size).astype(np.int32)[..., None]

        err_p = _extract_patch(s2.error_map, origin, cfg.patch_size)
        # Weight: every voxel counts at least 1.0; stage-1 mistakes count up to
        # (1 + error_weight_scale). Keeps easy voxels in the loss (so stage 2
        # doesn't forget what stage 1 already had right) while emphasizing errors.
        w = (1.0 + cfg.error_weight_scale * err_p).astype(np.float32)[..., None]

        yield x, y, w


def make_stage2_dataset(s2_subjects: List[Stage2Subject], cfg: Config, shuffle_buffer: int = 32):
    ps = cfg.patch_size
    sig = (
        tf.TensorSpec(shape=(ps, ps, ps, 2 + cfg.num_classes), dtype=tf.float32),
        tf.TensorSpec(shape=(ps, ps, ps, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(ps, ps, ps, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: stage2_patch_generator(s2_subjects, cfg), output_signature=sig
    )
    return ds.shuffle(shuffle_buffer).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def make_weighted_loss(cfg: Config):
    """Dice + per-voxel-weighted cross-entropy. Keras passes weights as sample_weight."""
    num_classes = cfg.num_classes
    dice_w, ce_w = cfg.dice_loss_weight, cfg.ce_loss_weight

    def loss_fn(y_true, y_pred, sample_weight=None):
        y_true_int = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, depth=num_classes)

        axes = tuple(range(1, 4))
        inter = tf.reduce_sum(y_true_onehot * y_pred, axis=axes)
        union = tf.reduce_sum(y_true_onehot, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
        dice = 1.0 - tf.reduce_mean((2.0 * inter + 1e-6) / (union + 1e-6))

        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # (B,D,H,W)
        if sample_weight is not None:
            w = tf.squeeze(sample_weight, axis=-1)
            ce = tf.reduce_sum(ce * w) / (tf.reduce_sum(w) + 1e-6)
        else:
            ce = tf.reduce_mean(ce)

        return dice_w * dice + ce_w * ce

    return loss_fn


def two_stage_predict(stage1_model, stage2_model, subject: Subject, cfg: Config) -> np.ndarray:
    """
    Full inference: stage 1 over the volume, then stage 2 conditioned on it.
    No ground truth needed -- exactly like applying trained boosting rounds.
    """
    probs1 = sliding_window_predict(stage1_model, subject, cfg).astype(np.float32)

    ps = cfg.patch_size
    step = max(1, int(ps * (1 - cfg.sliding_window_overlap)))
    shape = subject.t1.shape
    prob_sum = np.zeros(shape + (cfg.num_classes,), dtype=np.float32)
    counts = np.zeros(shape, dtype=np.float32)

    starts = []
    for dim in shape:
        if dim <= ps:
            starts.append([0])
        else:
            s = list(range(0, dim - ps + 1, step))
            if s[-1] != dim - ps:
                s.append(dim - ps)
            starts.append(s)

    for d0 in starts[0]:
        for h0 in starts[1]:
            for w0 in starts[2]:
                d1, h1, w1 = min(d0 + ps, shape[0]), min(h0 + ps, shape[1]), min(w0 + ps, shape[2])
                pad = [(0, ps - (d1 - d0)), (0, ps - (h1 - h0)), (0, ps - (w1 - w0))]
                t1p = np.pad(subject.t1[d0:d1, h0:h1, w0:w1], pad, mode="constant")
                t2p = np.pad(subject.t2[d0:d1, h0:h1, w0:w1], pad, mode="constant")
                pp = np.pad(probs1[d0:d1, h0:h1, w0:w1, :], pad + [(0, 0)], mode="constant")
                x = np.concatenate([np.stack([t1p, t2p], axis=-1), pp], axis=-1)[None, ...]

                pred = stage2_model(x, training=False).numpy()[0]
                prob_sum[d0:d1, h0:h1, w0:w1, :] += pred[: d1 - d0, : h1 - h0, : w1 - w0, :]
                counts[d0:d1, h0:h1, w0:w1] += 1.0

    return prob_sum / np.maximum(counts, 1e-6)[..., None]
