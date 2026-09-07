"""
Dice loss and per-class Dice metrics.

We optimize Dice directly (rather than just cross-entropy) because:
  1. It's the metric the challenge actually scores you on.
  2. The classes are heavily imbalanced (~87% background, see EDA in the
     README) -- plain cross-entropy would let the model get a deceptively
     low loss by mostly predicting background.

We combine soft Dice with a bit of cross-entropy, which tends to give more
stable gradients early in training than pure Dice loss (Dice loss can plateau
badly when a class is entirely absent from a batch).
"""
import tensorflow as tf
from config import Config


def soft_dice_loss(y_true_onehot, y_pred, num_classes: int, smooth: float = 1e-6):
    """Mean soft Dice loss across all classes (including background)."""
    axes = tuple(range(1, y_true_onehot.shape.rank - 1))  # spatial dims only
    intersection = tf.reduce_sum(y_true_onehot * y_pred, axis=axes)
    union = tf.reduce_sum(y_true_onehot, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)


def make_combined_loss(cfg: Config):
    """Returns a loss function suitable for model.compile(loss=...)."""
    num_classes = cfg.num_classes
    dice_w = cfg.dice_loss_weight
    ce_w = cfg.ce_loss_weight

    def loss_fn(y_true, y_pred):
        # y_true: (B, D, H, W, 1) int labels. y_pred: (B, D, H, W, C) softmax probs.
        y_true_int = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, depth=num_classes)

        dice = soft_dice_loss(y_true_onehot, y_pred, num_classes)
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        )
        return dice_w * dice + ce_w * ce

    return loss_fn


class PerClassDice(tf.keras.metrics.Metric):
    """Tracks Dice for a single class index, e.g. CSF / GM / WM separately."""

    def __init__(self, class_id: int, num_classes: int, name: str, smooth: float = 1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.num_classes = num_classes
        self.smooth = smooth
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_int = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_true_onehot = tf.one_hot(y_true_int, depth=self.num_classes)

        true_c = y_true_onehot[..., self.class_id]
        pred_c = y_pred[..., self.class_id]

        self.intersection.assign_add(tf.reduce_sum(true_c * pred_c))
        self.union.assign_add(tf.reduce_sum(true_c) + tf.reduce_sum(pred_c))

    def result(self):
        return (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


def make_metrics(cfg: Config):
    return [
        PerClassDice(class_id=i, num_classes=cfg.num_classes, name=f"dice_{name}")
        for i, name in enumerate(cfg.class_names)
    ]
