"""
Sliding-window inference over full volumes, Dice evaluation on held-out
subjects, and export of predictions in the original label value scheme
(0 / 10 / 150 / 250) for challenge submission.

A 144x192x256 volume doesn't fit through the network in one shot at a
reasonable patch size, so we tile it into overlapping patches, run each
through the model, and average overlapping predictions back into a full
probability volume before taking the argmax.
"""
import gc
import os
from typing import List, Tuple

import keras
import nibabel as nib
import numpy as np
import tensorflow as tf

from config import Config
from dataset import Subject, load_all_subjects, load_subject, discover_subject_ids, split_train_val
from losses import make_combined_loss, make_metrics


def _patch_origins(volume_shape, patch_size: int, step: int) -> List[Tuple[int, int, int]]:
    origins_per_dim = []
    for dim in volume_shape:
        if dim <= patch_size:
            origins_per_dim.append([0])
            continue
        starts = list(range(0, dim - patch_size + 1, step))
        if starts[-1] != dim - patch_size:
            starts.append(dim - patch_size)  # make sure the tail of the volume is covered
        origins_per_dim.append(starts)

    origins = []
    for d0 in origins_per_dim[0]:
        for h0 in origins_per_dim[1]:
            for w0 in origins_per_dim[2]:
                origins.append((d0, h0, w0))
    return origins


def sliding_window_predict(model: keras.Model, subject: Subject, cfg: Config) -> np.ndarray:
    """Returns a (D, H, W, num_classes) probability volume."""
    ps = cfg.patch_size
    step = max(1, int(ps * (1 - cfg.sliding_window_overlap)))
    shape = subject.t1.shape

    prob_sum = np.zeros(shape + (cfg.num_classes,), dtype=np.float32)
    counts = np.zeros(shape, dtype=np.float32)

    origins = _patch_origins(shape, ps, step)
    for (d0, h0, w0) in origins:
        d1, h1, w1 = min(d0 + ps, shape[0]), min(h0 + ps, shape[1]), min(w0 + ps, shape[2])
        t1_patch = subject.t1[d0:d1, h0:h1, w0:w1]
        t2_patch = subject.t2[d0:d1, h0:h1, w0:w1]

        pad = [(0, ps - t1_patch.shape[i]) for i in range(3)]
        t1_pad = np.pad(t1_patch, pad, mode="constant")
        t2_pad = np.pad(t2_patch, pad, mode="constant")
        input_patch = np.stack([t1_pad, t2_pad], axis=-1)[np.newaxis, ...]

        # model(...) instead of model.predict(...): for many single-batch calls in a
        # tight loop (one per patch), calling the model directly avoids the extra
        # dataset/callback bookkeeping .predict() does on every call, which otherwise
        # accumulates overhead and memory across a full-volume sliding window.
        pred = model(input_patch, training=False).numpy()[0]  # (ps, ps, ps, num_classes)
        pred_cropped = pred[: d1 - d0, : h1 - h0, : w1 - w0, :]

        prob_sum[d0:d1, h0:h1, w0:w1, :] += pred_cropped
        counts[d0:d1, h0:h1, w0:w1] += 1.0

    counts = np.maximum(counts, 1e-6)[..., np.newaxis]
    return prob_sum / counts


def dice_score(pred_label: np.ndarray, true_label: np.ndarray, class_id: int, smooth: float = 1e-6) -> float:
    pred_c = (pred_label == class_id).astype(np.float32)
    true_c = (true_label == class_id).astype(np.float32)
    intersection = np.sum(pred_c * true_c)
    union = np.sum(pred_c) + np.sum(true_c)
    return float((2.0 * intersection + smooth) / (union + smooth))


def evaluate_subjects(model: keras.Model, subjects: List[Subject], cfg: Config) -> dict:
    """Runs sliding-window inference + Dice scoring on a list of labeled subjects."""
    results = {name: [] for name in cfg.class_names}
    for subject in subjects:
        probs = sliding_window_predict(model, subject, cfg)
        pred_label = np.argmax(probs, axis=-1)
        for class_id, name in enumerate(cfg.class_names):
            d = dice_score(pred_label, subject.label, class_id)
            results[name].append(d)
        print(f"Subject {subject.subject_id}: "
              + ", ".join(f"{n}={results[n][-1]:.4f}" for n in cfg.class_names))
        del probs, pred_label
        gc.collect()
    return results


def save_prediction(pred_label: np.ndarray, reference_hdr_path: str, out_path: str, cfg: Config):
    """Maps class indices {0,1,2,3} back to raw label values {0,10,150,250}
    and saves as a NIfTI file, reusing the affine of the original scan."""
    ref_img = nib.load(reference_hdr_path)
    raw = np.zeros_like(pred_label, dtype=np.int16)
    for idx, raw_val in enumerate(cfg.raw_label_values):
        raw[pred_label == idx] = raw_val
    out_img = nib.Nifti1Image(raw, affine=ref_img.affine)
    nib.save(out_img, out_path)


def main():
    cfg = Config()
    model_path = os.path.join(cfg.checkpoint_dir, "best_model.keras")
    model = keras.models.load_model(
        model_path,
        custom_objects={"loss_fn": make_combined_loss(cfg)},
        compile=False,
    )

    # --- Evaluate on held-out labeled subjects ---
    subjects = load_all_subjects(cfg.train_dir, cfg, has_labels=True)
    _, val_subjects = split_train_val(subjects, cfg)
    print("Evaluating on validation subjects...")
    results = evaluate_subjects(model, val_subjects, cfg)
    print("\nMean Dice per class:")
    for name, scores in results.items():
        print(f"  {name}: {np.mean(scores):.4f}")

    # --- Predict on the unlabeled test set and export ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    test_ids = discover_subject_ids(cfg.test_dir, has_labels=False)
    for sid in test_ids:
        subject = load_subject(cfg.test_dir, sid, cfg, has_labels=False)
        probs = sliding_window_predict(model, subject, cfg)
        pred_label = np.argmax(probs, axis=-1)
        ref_path = os.path.join(cfg.test_dir, f"subject-{sid}-T1.hdr")
        out_path = os.path.join(cfg.output_dir, f"subject-{sid}-prediction.nii.gz")
        save_prediction(pred_label, ref_path, out_path, cfg)
        print(f"Saved prediction for subject {sid} -> {out_path}")
        del subject, probs, pred_label
        gc.collect()


if __name__ == "__main__":
    main()
