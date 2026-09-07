"""
A frugal 3D U-Net for iSeg-2017 tissue segmentation.

Design rationale (tied to the hackathon's "Dice / network size" criterion):

1. Skip connections (the actual "U" in U-Net) are the main addition over a
   plain conv/deconv encoder-decoder: at each resolution level, the encoder
   feature map is concatenated into the decoder before the corresponding
   upsampling step. This lets fine tissue-boundary detail bypass the
   bottleneck, which matters a lot here since gray/white matter contrast in
   6-month-old infants is famously low.

2. Downsampling uses strided Conv3D rather than MaxPooling3D, for the same
   reason as the base DLwP chapter: max pooling discards *which* location in
   the window the strongest activation came from, and segmentation needs
   precise per-voxel spatial information. A learned strided conv keeps more
   of that information than an arg-max operation would.

3. GroupNormalization instead of BatchNormalization: 3D volumes at a
   reasonable patch size force a very small batch size (2-4), where batch
   statistics are unstable. GroupNorm normalizes within each sample, so it's
   insensitive to batch size.

4. The whole network is deliberately narrow (few filters per level, few
   levels) so total parameter count stays low -- the point isn't to maximize
   raw Dice, it's to maximize Dice *per parameter*. See count_params() below
   and tune `base_filters` / `n_levels` in config.py to explore that tradeoff.
"""
from typing import Tuple

import keras
from keras import layers

from config import Config


def _conv_block(x, filters: int, groups: int, name: str):
    x = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.GroupNormalization(groups=min(groups, filters), name=f"{name}_gn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv3D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.GroupNormalization(groups=min(groups, filters), name=f"{name}_gn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def _downsample(x, filters: int, groups: int, name: str):
    """Learned downsampling via a strided conv, instead of MaxPooling3D."""
    x = layers.Conv3D(filters, 3, strides=2, padding="same", name=f"{name}_stridedconv")(x)
    x = layers.GroupNormalization(groups=min(groups, filters), name=f"{name}_gn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def _upsample(x, filters: int, groups: int, name: str):
    x = layers.Conv3DTranspose(filters, 3, strides=2, padding="same", name=f"{name}_upconv")(x)
    x = layers.GroupNormalization(groups=min(groups, filters), name=f"{name}_gn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def build_unet3d(cfg: Config, input_shape: Tuple[int, int, int, int] = None) -> keras.Model:
    """
    Builds a small 3D U-Net.

    input_shape defaults to (patch_size, patch_size, patch_size, 2) -- the
    "2" is the T1+T2 channel stack. Using None for the spatial dims instead
    would also work at inference time on arbitrarily sized crops, but a
    fixed size keeps shape bookkeeping for the skip connections simple.
    """
    if input_shape is None:
        ps = cfg.patch_size
        input_shape = (ps, ps, ps, 2)

    inputs = keras.Input(shape=input_shape, name="t1_t2_input")
    x = inputs

    skips = []
    filters = cfg.base_filters
    # Encoder
    for level in range(cfg.n_levels):
        x = _conv_block(x, filters, cfg.groupnorm_groups, name=f"enc{level}")
        skips.append(x)
        x = _downsample(x, filters * 2, cfg.groupnorm_groups, name=f"down{level}")
        filters *= 2

    # Bottleneck
    x = _conv_block(x, filters, cfg.groupnorm_groups, name="bottleneck")

    # Decoder
    for level in reversed(range(cfg.n_levels)):
        filters //= 2
        x = _upsample(x, filters, cfg.groupnorm_groups, name=f"up{level}")
        x = layers.Concatenate(name=f"skip_concat{level}")([x, skips[level]])
        x = _conv_block(x, filters, cfg.groupnorm_groups, name=f"dec{level}")

    outputs = layers.Conv3D(
        cfg.num_classes, 1, padding="same", activation="softmax", name="segmentation_output"
    )(x)

    return keras.Model(inputs, outputs, name="frugal_unet3d")


def count_params(model: keras.Model) -> int:
    return model.count_params()


if __name__ == "__main__":
    cfg = Config()
    model = build_unet3d(cfg)
    model.summary()
    print(f"\nTotal parameters: {count_params(model):,}")
