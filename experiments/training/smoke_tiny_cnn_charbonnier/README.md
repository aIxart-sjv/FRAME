# Experiment `smoke_tiny_cnn_charbonnier`

Status: **completed**. A training smoke experiment from `python -m frame.train`; numbers are means over the validation patches listed below, measured by this run, and are not benchmark results. No winner is declared.

> The data are FRAME's synthetic smoke scenes: these results are **not evidence about real** Sentinel-2 imagery or about any model's real-world quality.

## Data
- datasets: synthetic_smoke; manifest digest `01bb0e2d18892c71bdf5228c9cb544b7e406eb49ee1bf9917519b1a2622949e1`
- train: 6 pairs from scenes ['R01_S00', 'R01_S01', 'R02_S00', 'R02_S01', 'R04_S00', 'R04_S01']
- validation: 2 pairs from scenes ['R03_S00', 'R03_S01']
- records in other splits (never read): {'test': 2}
- bands (by name): ('B04', 'B03', 'B02', 'B08') -> ('B04', 'B03', 'B02', 'B08'); LR train patch 32, validation patch 128

## Model and training
- model: `tiny_cnn` {'width': 32, 'depth': 4}; parameters 47424 (47424 trainable, 0.1809 MiB)
- optimizer adamw lr 0.001, scheduler cosine, loss charbonnier (+ spectral 0.0, consistency 0.0)
- steps 600, batch 4 x accumulation 2, precision {'requested': 'off', 'effective': 'off', 'note': None}, seed 0
- training time 4.9276 s on cuda; peak VRAM 25.47900390625 MiB

## Validation (deterministic, same patches for every row)
| model | patches | loss | PSNR (dB) | SSIM | RMSE | SAM (deg) | ERGAS |
|---|---|---|---|---|---|---|---|
| tiny_cnn - step 0 (before training) | 2 | 0.0062 | 33.267 | 0.9126 | 0.0217 | 2.4672 | 4.0013 |
| tiny_cnn - trained, step 600 | 2 | 0.0035 | 36.691 | 0.9515 | 0.0146 | 1.7231 | 2.4632 |
| baseline: bicubic | 2 | 0.0062 | 33.267 | 0.9126 | 0.0217 | 2.4672 | 4.0013 |
| baseline: lite | 2 | 0.0049 | 34.846 | 0.9327 | 0.0181 | 2.1607 | 3.1660 |

Rows are not comparable in kind: baselines may carry pretrained weights or a hard constraint the trained model does not.

## Training scenes (over-fit check)
The same evaluation on the scenes the model TRAINED on (no augmentation). A gap to the validation table shows how much of the fit is memorisation; this is a sanity check that the loop can fit data, not a quality result.
| model | patches | loss | PSNR (dB) | SSIM | RMSE | SAM (deg) | ERGAS |
|---|---|---|---|---|---|---|---|
| tiny_cnn - trained, step 600 | 6 | 0.0034 | 36.635 | 0.9524 | 0.0148 | 1.7375 | 2.5494 |
| baseline: bicubic | 6 | 0.0061 | 33.285 | 0.9150 | 0.0217 | 2.3929 | 4.1107 |
| baseline: lite | 6 | 0.0049 | 34.814 | 0.9341 | 0.0182 | 2.1377 | 3.2967 |

## Environment
- python 3.11.9, torch 2.14.0+cu130, numpy 2.4.6, Linux-7.1.5-arch1-2-x86_64-with-glibc2.44
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 3759 MiB
- code: git revision `eb3769559525ea81e902422253f8950c36b075a5`, working tree dirty: True

## Reproduce
`python -m frame.train run <this directory>/config.json` (same data, seed and config; checkpoints in `checkpoint/` are not committed).
