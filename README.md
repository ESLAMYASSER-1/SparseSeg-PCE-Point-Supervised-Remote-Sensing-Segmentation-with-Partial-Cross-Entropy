# SparseSeg-PCE: Partial Cross-Entropy for Sparse Point-Supervised Remote-Sensing Segmentation

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

Investigates how much sparse point-level supervision a segmentation model actually needs, using Partial Cross-Entropy (PCE) loss on the OpenEarthMap remote-sensing dataset.

## Table of Contents
- [Overview](#overview)
- [Method](#method)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Status](#status)

## Overview

Dense pixel-level annotation for semantic segmentation is expensive to collect. This project simulates **sparse point supervision** — labeling only a small fraction of pixels — and measures how a U-Net segmentation model's performance changes as point-label density drops from 20% down to just 1%.

## Method

- **Loss:** Partial Cross-Entropy — only labeled pixels contribute to the loss; unlabeled pixels use an ignore index (`-1`), implemented via `CrossEntropyLoss(ignore_index=-1)`.
- **Model:** Standard U-Net, RGB input → dense multi-class segmentation output.
- **Dataset:** [OpenEarthMap](https://open-earth-map.org/) — dense ground-truth masks are sub-sampled to simulate sparse point labels; full dense masks are kept for validation.

## Experimental Design

| Factor | Values |
|---|---|
| Point-label density | 1%, 5%, 10%, 20% |
| Random seeds per density | 42, 43, 44 (12 runs total) |
| Primary metric | mIoU |
| Secondary metrics | Pixel Accuracy, Macro Precision/Recall/F1, Per-Class IoU |

Model architecture, optimizer, crop size, batch size, augmentation, and early-stopping configuration (monitoring `val/mIoU`) are held constant across all runs, so point-label density is the only factor being varied.

## Results

> Full results across all 12 runs (4 densities × 3 seeds) are still being finalized. This table and the accompanying training-curve / per-class-IoU figures will be filled in as runs complete.

| Point Fraction | mIoU (mean ± std) | Pixel Accuracy | Macro F1 |
|---|---|---|---|
| 1% | _pending_ | _pending_ | _pending_ |
| 5% | _pending_ | _pending_ | _pending_ |
| 10% | _pending_ | _pending_ | _pending_ |
| 20% | _pending_ | _pending_ | _pending_ |

## Getting Started

```bash
git clone https://github.com/ESLAMYASSER-1/sparseseg-pce.git
cd sparseseg-pce
pip install -r requirements.txt
python src/main.py --point-density 0.05 --seed 42
```

## Limitations

- Point labels are simulated from dense annotations, not real human point-clicks.
- Point sampling is uniformly random, which may not reflect real annotation behavior.
- Only 3 random seeds are evaluated per density.
- RGB-only input; results may not generalize to other sensor modalities or geographies outside OpenEarthMap's coverage.

## Status

🚧 **Active / in progress.** The data pipeline, Partial Cross-Entropy loss, U-Net training loop, and Lightning metric logging are implemented. The full 12-run sweep and final analysis (results table + figures above) are in progress — check the [commit history](../../commits/main) for the latest state.
