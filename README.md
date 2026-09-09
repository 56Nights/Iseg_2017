# Two-Stage Boosted 3D U-Net — iSeg-2017

SCIA 2026 Hackathon, Sujet 2. Segmenting 6-month-old infant brain MRI into white matter,
gray matter and cerebrospinal fluid from T1 and T2 scans.

Graded on Dice / network size and on originality, so I tried to do something more
interesting than fine-tuning a big pretrained model.

## How I got here

My first idea was to use SAM with point and box prompts. I dropped it once I looked at
the data properly: SAM is over 600M parameters (bad for a frugality score), it was trained
on natural photos, it works in 2D, and an MRI is a 3D volume — 144×192×256 voxels per
subject. Segmenting it slice by slice throws away the continuity between slices, and I'd
have needed to generate prompts automatically for every tissue on every slice anyway.

So I went with a 3D U-Net instead. That's the obvious choice, which is also the problem:
I read the iSeg-2017 benchmark paper (Wang et al., IEEE TMI 2019) and 8 of the top 20
teams used a U-Net. Just building one isn't original, and it isn't especially frugal
either — the winning entry (MSL_SKKU) was only 1.55M parameters, smaller than my baseline.

What I found useful in that paper was the error analysis. All 8 top methods fail in the
same place: the cortical GM/WM boundary. At 6 months the brain is in the "isointense
phase" — myelination is halfway done, so WM and GM have almost the same intensity in both
T1 and T2. It's the lowest tissue contrast of the whole first year, which is exactly why
the organizers picked this age. Subcortical regions score around 0.94 Dice; the cortex is
where everyone loses points.

The paper also mentions that all the top teams sampled their training patches randomly
"without evaluating the importance of each sample", and suggests error maps could be used
to pick better samples. Nobody in the top 8 did it. That gave me an angle.

## The idea

Instead of one network trying to be equally good everywhere, I split it into two stages
and structured it like gradient boosting:

1. **Stage 1** is a normal small 3D U-Net. It makes a first attempt.
2. I compute a **residual map** on the training subjects, where I have labels:
   `error = 1 - P(correct class)` for every voxel. Same thing a boosting round fits against.
3. **Stage 2** is a second, smaller U-Net. It gets `[T1, T2, stage-1 probabilities]` as
   input and is trained specifically on stage 1's mistakes, using error-biased patch
   sampling and a loss weighted by the error map.

I was worried at first that this couldn't work at test time since I have no labels for the
test subjects, so no error map. But that's how XGBoost works too — each tree is fitted to
residuals during training, and at inference you just run the trees forward. Stage 2 learns
what stage-1 failure looks like *from the inputs* (low contrast, cortical ribbon, stage 1
being unconfident), so it doesn't need the labels once trained.

## Does it actually work? Yes and no — here's what I found

I ran the full pipeline (stage 1: 150 epochs, early-stopped at 71; stage 2: 100 epochs,
early-stopped at 66) and evaluated on the 2 held-out subjects. Three separate checks, and
they don't all point the same way:

**1. Stage 1 was already strong, so there wasn't much error left to fix.** Mean stage-1
residual across all subjects was only 0.009–0.013. Tissue Dice was already ~0.87–0.94 per
class before stage 2 touched anything. I'd expected more headroom.

**2. Aggregate Dice went up slightly.**

| class | stage 1 | two-stage | delta |
|---|---|---|---|
| CSF | 0.9390 | 0.9413 | +0.0022 |
| gray matter | 0.8995 | 0.9020 | +0.0026 |
| white matter | 0.8713 | 0.8732 | +0.0019 |
| **tissue mean** | **0.9033** | **0.9055** | **+0.0022** |

**3. But Dice-per-parameter went down**, which is the number the hackathon actually
grades on:

```
Parameters: stage 1 2,188,244 | stage 2 548,940 | combined 2,737,184
Tissue Dice per million params: stage 1 0.413 -> two-stage 0.331
```

Stage 2 cost 549K parameters (25% more) to buy +0.0022 Dice. On Dice/size, that's a loss.

**4. Per-subject, the correction isn't even consistent.** I checked which voxels stage 2
actually fixed versus broke, subject by subject:

```
Subject 9:  changed 10,021 | fixed 4,449 | broke 5,498 | net -1,049
Subject 10: changed 8,370  | fixed 4,477 | broke 3,825 | net +652
```

It helped one validation subject and hurt the other. With only 2 validation subjects that's
weak evidence of anything generalizable — the small positive average Dice is two roughly
opposite effects nearly canceling out, not a consistent correction.

## My actual conclusion

The hypothesis was reasonable and the implementation works correctly, but the evidence
says it doesn't pay for itself here: stage 1 already leaves very little residual error
(mean 0.01), and with only 8 training subjects, stage 2 doesn't learn a correction that
generalizes — it helps one validation brain and hurts the other, and whatever it does gain
in raw Dice doesn't cover its own parameter cost. I'm reporting this as a negative result
with a diagnosed cause, not as a win, because I think that's more honest and — I'd argue —
more useful to a jury than an unexamined positive number would have been.

## What would actually fix it, if I had more time

- **Out-of-fold residuals.** Right now residual maps for the training subjects are computed
  in-sample — stage 1 already fit those exact brains, so its errors there are artificially
  small and don't represent what it does on a brain it's never seen. The correct fix is
  training stage 1 eight times, leaving one subject out each time, and using the held-out
  prediction for that subject's residual map. I didn't have the compute budget to do this
  during the hackathon.
- **More training data or augmentation**, since 8 subjects is not enough for stage 2 to
  learn a pattern that isn't mostly noise.

## Parameters

| | Parameters |
|---|---|
| Stage 1 | 2,188,244 |
| Stage 2 | 548,940 |
| **Combined** | **2,737,184** |

For reference: the standard 3D U-Net is around 19M, and the iSeg-2017 winner was 1.55M.
I'm well under the textbook architecture but not smaller than the actual winner, which is
part of why I'm not pitching this on parameter count alone.

## Data

10 labeled training subjects, 13 unlabeled test subjects, as Analyze `.hdr`/`.img` pairs,
144×192×256 at 1mm isotropic:

- `subject-N-T1.hdr/.img`, `subject-N-T2.hdr/.img` — the two modalities
- `subject-N-label.hdr/.img` — ground truth for training subjects, values `{0, 10, 150, 250}`
  for background / CSF / GM / WM

I feed T1 and T2 as two input channels rather than picking one. They have complementary
contrast — tissue that's ambiguous in T1 is often clearer in T2 — which is the whole reason
the challenge gives you both.

Class balance is about 87% background, 6% GM, 4% WM, 3% CSF. That's why the loss is
Dice + cross-entropy rather than plain cross-entropy (a model predicting all background
scores 87% accuracy and is useless), and why patch sampling oversamples foreground.

## Setup — using uv, not pip

I switched to [uv](https://docs.astral.sh/uv/) for dependency management. It's faster than
pip and locks exact versions so the environment is reproducible.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the project root
uv sync
```

`uv sync` reads `pyproject.toml`, resolves everything against `uv.lock`, and creates a
`.venv` automatically — no manual venv creation, no `pip install -r requirements.txt`.
Every script below is run through `uv run`, which uses that environment without needing
to activate it manually.

Then get the data:

```bash
unzip iSeg-2017-Training.zip -d data/train_raw
unzip iSeg-2017-Testing.zip  -d data/test_raw
```

## Files

```
pyproject.toml            dependencies, resolved by uv
uv.lock                   exact locked versions, for a reproducible environment

src/
  config.py               every hyperparameter in one place
  dataset.py              loading, z-score normalization, label remapping, patch sampling
  model.py                the 3D U-Net (both stages use the same builder)
  losses.py               Dice + cross-entropy, per-class Dice metrics
  stage2.py               the actual idea: residual maps, error-region selection,
                          biased sampling, weighted loss, two-stage inference
  train_stage1.py         trains the baseline
  train_stage2.py         builds residual maps, trains the boosting stage
  evaluate.py             stage 1 vs two-stage, plus Dice per million parameters
  diagnose.py             per-subject fixed-vs-broke voxel breakdown
  export_predictions.py   writes .nii.gz files for the 13 test subjects
  visualize.py            figures for the report

notebook/
  iseg_boosted_unet.ipynb   same pipeline as one notebook, for Colab/Kaggle GPUs
                            (uses pip, since that's what those platforms expect)
```

## Running it

You need a GPU — 3D convolutions are painfully slow on CPU.

```bash
cd src
uv run python train_stage1.py        # baseline
uv run python train_stage2.py        # residuals + boosting stage
uv run python evaluate.py            # aggregate Dice, stage 1 vs two-stage
uv run python diagnose.py            # per-subject fixed vs broke voxels
uv run python visualize.py           # figures
uv run python export_predictions.py  # submission files
```

Everything reads from `config.py`. `evaluate.py` and `export_predictions.py` work with just
stage 1 if there's no stage-2 checkpoint yet, so you can look at the baseline first.
Subjects 9 and 10 are the validation split by default.

## Choices I made and why

- **T1 + T2 stacked at the input.** Using one modality wastes half the information.
- **Strided Conv3D instead of MaxPooling3D** for downsampling. Max pooling only keeps the
  strongest value in each window and forgets *where* in the window it came from, which
  matters when you're predicting a label per voxel and boundaries need to be precise.
- **GroupNormalization instead of BatchNorm.** 64³ patches in 3D force a batch size of 2–4,
  and batch statistics are unreliable that small. GroupNorm normalizes within each sample.
- **Dice + cross-entropy.** Dice is the metric being scored, so I optimize it directly, but
  pure Dice loss stalls when a class is missing from a batch, so the CE term keeps gradients
  sensible early on.
- **Skip connections.** This is what makes it a U-Net rather than a plain encoder-decoder.
  Without them, fine boundary detail has to survive being squeezed through the bottleneck,
  and the cortical GM/WM boundary is exactly the detail I can't afford to lose.
- **Stage 2 is smaller than stage 1.** It only has to fix residuals, not solve the whole
  problem, so it doesn't need full capacity. Didn't end up paying for itself, but the
  reasoning behind sizing it that way still stands.

## Two things that broke when I tested it (both fixed, both in the code)

**The error threshold has to be adaptive.** I started with a fixed cutoff — a voxel counts
as an error if `error > 0.3`. But an undertrained stage 1 is wrong almost everywhere, so
every voxel in the volume passed the threshold and "focus on the errors" silently became
plain uniform sampling. I now rank voxels by error and keep the worst 25%, using
`argpartition` rather than a percentile cutoff, because percentiles break on ties: when
stage 1 outputs the same error value everywhere, `error > percentile` selects nothing.

**Error voxels have to be restricted to the brain.** Background is 87% of the volume, so
when background counted as "error" every patch I drew came back completely empty.

## About overfitting

With only 8 training subjects, training stage 2 exclusively on error regions would overfit
badly — memorizing where the errors are in these specific brains rather than learning what
an ambiguous region looks like in general. So the sampling is biased, not restricted: about
65% of patches are centered on high-error voxels, but each patch is still a full 64³ cube
with normal anatomy around it, redrawn at a different random center every step. The other
35% are drawn uniformly. `error_oversample_ratio` and `error_weight_scale` control this.
As the results above show, this wasn't enough to make the boosting reliably generalize —
but the mechanism itself is sound; the bottleneck was the residual signal being both small
and computed in-sample.

## Limitations

- 8 training subjects and a 2-subject validation set. As shown above, that's thin enough
  that per-subject results disagree in sign.
- No data augmentation yet (flips, rotations, elastic deformation, intensity jitter). Given
  how little data there is, this is probably the biggest easy win left.
- Residuals for stage 2 are computed in-sample on training subjects, not out-of-fold. This
  is the most likely reason stage 2 didn't generalize (see "what would actually fix it" above).
- The benchmark paper notes every method degrades on the motion-corrupted test subjects.
  Boosting doesn't address that; simulated motion augmentation might.
- Stage 2 doubles inference time since stage 1 has to run over the full volume first.
