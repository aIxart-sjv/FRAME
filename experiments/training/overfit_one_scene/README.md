# Experiment `overfit_one_scene`

Status: **completed**. A training smoke experiment from `python -m frame.train`; numbers are means over the validation patches listed below, measured by this run, and are not benchmark results. No winner is declared.

> The data are FRAME's synthetic smoke scenes: these results are **not evidence about real** Sentinel-2 imagery or about any model's real-world quality.

## Data
- datasets: synthetic_smoke; manifest digest `c27cc7aa04d1f58a42490bab0d42186db61bb213e707d4435c09360319b945a5`
- train: 1 pairs from scenes ['R01_S00']
- validation: 1 pairs from scenes ['R00_S00']
- records in other splits (never read): none
- bands (by name): ('B04', 'B03', 'B02', 'B08') -> ('B04', 'B03', 'B02', 'B08'); LR train patch 32, validation patch 128

## Model and training
- model: `tiny_cnn` {'width': 32, 'depth': 4}; parameters 47424 (47424 trainable, 0.1809 MiB)
- optimizer adamw lr 0.001, scheduler none, loss l1 (+ spectral 0.0, consistency 0.0)
- steps 2000, batch 1 x accumulation 1, precision {'requested': 'off', 'effective': 'off', 'note': None}, seed 0
- training time 6.1907 s on cuda; peak VRAM 18.24853515625 MiB

## Validation (deterministic, same patches for every row)
| model | patches | loss | PSNR (dB) | SSIM | RMSE | SAM (deg) | ERGAS |
|---|---|---|---|---|---|---|---|
| tiny_cnn - step 0 (before training) | 1 | 0.0094 | 33.106 | 0.9128 | 0.0221 | 2.5379 | 4.7500 |
| tiny_cnn - trained, step 2000 | 1 | 0.0060 | 36.724 | 0.9565 | 0.0146 | 1.7992 | 2.8409 |
| baseline: bicubic | 1 | 0.0094 | 33.106 | 0.9128 | 0.0221 | 2.5379 | 4.7500 |

Rows are not comparable in kind: baselines may carry pretrained weights or a hard constraint the trained model does not.

## Training scenes (over-fit check)
The same evaluation on the scenes the model TRAINED on (no augmentation). A gap to the validation table shows how much of the fit is memorisation; this is a sanity check that the loop can fit data, not a quality result.
| model | patches | loss | PSNR (dB) | SSIM | RMSE | SAM (deg) | ERGAS |
|---|---|---|---|---|---|---|---|
| tiny_cnn - trained, step 2000 | 1 | 0.0055 | 37.748 | 0.9599 | 0.0130 | 1.3689 | 2.7305 |
| baseline: bicubic | 1 | 0.0088 | 33.738 | 0.9180 | 0.0206 | 2.1246 | 4.7794 |

## Environment
- python 3.11.9, torch 2.14.0+cu130, numpy 2.4.6, Linux-7.1.5-arch1-2-x86_64-with-glibc2.44
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 3759 MiB
- code: git revision `eb3769559525ea81e902422253f8950c36b075a5`, working tree dirty: True

## Reproduce
`python -m frame.train run <this directory>/config.json` (same data, seed and config; checkpoints in `checkpoint/` are not committed).
