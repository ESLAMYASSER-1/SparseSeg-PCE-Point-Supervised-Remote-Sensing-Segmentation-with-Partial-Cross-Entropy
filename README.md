# Partial Cross-Entropy for Sparse Point-Supervised Remote-Sensing Segmentation

## 1. Project Overview

This project investigates **Partial Cross-Entropy (PCE) Loss** for semantic segmentation when only a small number of labeled pixels are available.

The experiment uses the **OpenEarthMap** remote-sensing semantic-segmentation dataset. The original dense segmentation masks are used to simulate sparse point-level supervision by randomly selecting a fraction of labeled pixels and treating all remaining pixels as unlabeled.

A **U-Net** segmentation model is trained using only these sparse labels. The model is then evaluated against the complete dense validation masks.

The main objective is to determine how the amount of sparse supervision affects segmentation performance and how robust the results are to different random point selections.

---

# 2. Task Requirements

The implementation must satisfy the following requirements.

### Requirement 1 — Partial Cross-Entropy

Implement Partial Cross-Entropy such that:

- Labeled pixels contribute to the loss.
- Unlabeled pixels do not contribute to the loss.
- Unlabeled pixels are represented using an ignore label such as `-1`.

The loss is:

\[
L_{PCE}
=
-\frac{1}{|\Omega_L|}
\sum_{i\in\Omega_L}
\log p(y_i|x_i)
\]

where:

- \(\Omega_L\) is the set of labeled pixels.
- \(y_i\) is the ground-truth class at labeled pixel \(i\).
- \(p(y_i|x_i)\) is the predicted probability of the correct class.

In practice, this can be implemented using Cross Entropy with `ignore_index=-1`.

---

# 3. Dataset

## OpenEarthMap

The experiment uses the **OpenEarthMap** dataset for remote-sensing semantic segmentation.

The dataset contains high-resolution remote-sensing imagery from geographically diverse regions and provides dense land-cover segmentation annotations.

For this project:

- RGB imagery is used as input.
- Dense masks are used as the reference annotations.
- Dense training masks are converted into simulated sparse point labels.
- Dense validation masks remain unchanged for evaluation.

The purpose of using the dense masks during evaluation is to measure how well a model trained from sparse supervision can reconstruct the complete segmentation.

---

# 4. Sparse Point-Supervision Simulation

For every training mask, randomly sample a predefined fraction of valid pixels.

The following point-label densities must be evaluated:

```text
1%
5%
10%
20%
```

For example, at 5% supervision:

```text
Original mask:

[1 1 1 2 2]
[1 1 2 2 2]
[3 3 3 2 2]

Sparse mask:

[-1 1 -1 -1 2]
[-1 -1 2 -1 -1]
[3 -1 -1 -1 -1]
```

Here:

- `-1` = unlabeled pixel
- Any other value = labeled pixel

Only the labeled pixels contribute to Partial Cross-Entropy.

---

# 5. Segmentation Model

Use a standard **U-Net** architecture.

### Input

```text
RGB image
3 channels
```

### Output

```text
Dense semantic segmentation
C classes
```

The number of output classes must match the class configuration used by the selected OpenEarthMap preprocessing pipeline.

---

# 6. Training

Use the same training configuration for all experiments.

Recommended configuration:

```text
Model:              U-Net
Optimizer:          AdamW
Loss:               Partial Cross-Entropy
Input:              RGB
Validation:         Dense ground-truth masks
Maximum epochs:     100
Early stopping:     Enabled
```

The following parameters must remain constant when comparing point-label densities:

- Model architecture
- Dataset split
- Image preprocessing
- Crop size
- Batch size
- Optimizer
- Learning rate
- Data augmentation
- Maximum number of epochs
- Early-stopping configuration
- Evaluation procedure

Only the experimental factor being investigated should change.

---

# 7. Early Stopping

Early stopping should monitor validation mIoU.

Example:

```python
EarlyStopping(
    monitor="val/mIoU",
    mode="max",
    patience=10,
    min_delta=0.001,
)
```

The best checkpoint should also be saved using `ModelCheckpoint`.

The best validation mIoU should be used as the primary result for each run.

---

# 8. Experimental Design

## Experiment 1 — Point-Label Density

### Research Question

> How does the amount of sparse point supervision affect semantic-segmentation performance?

### Independent Variable

Point-label density:

```text
1%
5%
10%
20%
```

### Dependent Variables

Primary:

```text
mIoU
```

Secondary:

```text
Pixel Accuracy
Macro Precision
Macro Recall
Macro F1
Per-Class IoU
```

### Hypothesis

Increasing point-label density is expected to provide the model with more spatial supervision and may improve segmentation performance.

However, this hypothesis must be evaluated using the measured results.

**Do not claim an improvement unless the experimental results support it.**

---

# 9. Experiment 2 — Random-Seed Robustness

Each point-label density must be evaluated using three independent random seeds:

```text
42
43
44
```

This produces:

```text
4 point densities × 3 seeds = 12 runs
```

### Example

```text
1%:
    seed 42
    seed 43
    seed 44

5%:
    seed 42
    seed 43
    seed 44

10%:
    seed 42
    seed 43
    seed 44

20%:
    seed 42
    seed 43
    seed 44
```

The purpose is to determine whether performance depends strongly on which pixels are randomly selected as supervision points.

---

# 10. Evaluation Metrics

Every experiment must report:

### 10.1 Mean Intersection over Union

\[
IoU_c =
\frac{TP_c}
{TP_c + FP_c + FN_c}
\]

and:

\[
mIoU =
\frac{1}{C}
\sum_{c=1}^{C} IoU_c
\]

mIoU is the primary metric.

---

### 10.2 Pixel Accuracy

\[
Accuracy =
\frac{\sum_c TP_c}
{N}
\]

---

### 10.3 Macro Precision

Calculate precision independently for each class and then average across classes.

---

### 10.4 Macro Recall

Calculate recall independently for each class and then average across classes.

---

### 10.5 Macro F1

Calculate F1 independently for each class and then average across classes.

---

### 10.6 Per-Class IoU

Report IoU for every OpenEarthMap class.

This is important because overall mIoU can hide poor performance on individual classes.

---

# 11. Lightning Logging

Use PyTorch Lightning logging to record the metrics.

Example:

```python
self.log_dict(
    {
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mIoU": miou,
        "val/pixel_acc": pixel_acc,
        "val/macro_precision": precision,
        "val/macro_recall": recall,
        "val/macro_f1": f1,
    },
    on_step=False,
    on_epoch=True,
    prog_bar=True,
)
```

Per-class IoU should be logged individually:

```python
for i, iou in enumerate(val_per_class_iou):
    self.log(
        f"val/iou_class_{i}",
        float(iou),
        on_step=False,
        on_epoch=True,
    )
```


---

# 12. Experiment Directory Structure

Organize the experiments as follows:

```text
experiments/
│
├── 1pct/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
│
├── 5pct/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
│
├── 10pct/
│   ├── seed_42/
│   ├── seed_43/
│   └── seed_44/
│
└── 20pct/
    ├── seed_42/
    ├── seed_43/
    └── seed_44/
```

Each run should contain its Lightning logs and checkpoints.

Example:

```text
seed_42/
├── lightning_logs/
├── checkpoints/
└── configuration.json
```

---

# 13. Results Table

The final report must contain a summary table:

| Point Fraction | mIoU Mean | mIoU Std | Pixel Accuracy Mean | Macro F1 Mean |
|---------------:|----------:|---------:|---------------------:|--------------:|
| 1%             |           |          |                      |               |
| 5%             |           |          |                      |               |
| 10%            |           |          |                      |               |
| 20%            |           |          |                      |               |

The mean and standard deviation are calculated across seeds 42, 43 and 44.

---

# 14. Per-Seed Results

The report should also contain the individual results:

| Point Fraction | Seed | mIoU | Pixel Accuracy | Macro F1 |
|---------------:|-----:|-----:|---------------:|---------:|
| 1% | 42 | | | |
| 1% | 43 | | | |
| 1% | 44 | | | |
| 5% | 42 | | | |
| 5% | 43 | | | |
| 5% | 44 | | | |
| 10% | 42 | | | |
| 10% | 43 | | | |
| 10% | 44 | | | |
| 20% | 42 | | | |
| 20% | 43 | | | |
| 20% | 44 | | | |

---

# 15. Required Figures

The final report should contain at least the following figures.

### Figure 1 — Training Loss

Plot:

```text
Epoch vs Training Loss
```

---

### Figure 2 — Validation Loss

Plot:

```text
Epoch vs Validation Loss
```

---

### Figure 3 — Validation mIoU

Plot:

```text
Epoch vs Validation mIoU
```

Preferably show the four point-label densities for comparison.

---

### Figure 4 — Point Density vs mIoU

Plot:

```text
Point-label density vs Mean mIoU
```

Include error bars representing standard deviation across seeds.

Example:

```text
mIoU
 ^
 |
 |                 ●
 |          ●
 |      ●
 |  ●
 +----------------------> Point density
    1%   5%  10%  20%
```

The actual trend must come from the experimental results.

---

### Figure 5 — Per-Class IoU

Compare per-class IoU across the different point-label densities.

This can reveal whether sparse supervision disproportionately affects specific land-cover classes.

---

# 16. Analysis Requirements

The report must answer the following questions.

### Question 1

> Does increasing point-label density improve segmentation performance?

Use mIoU as the primary evidence.

---

### Question 2

> How stable are the results across random seeds?

Use:

```text
Mean ± Standard Deviation
```

A high standard deviation indicates greater sensitivity to the random point-selection process.

---

### Question 3

> Which semantic classes are most affected by sparse supervision?

Use per-class IoU.

---

### Question 4

> Does increasing point density show diminishing returns?

Compare the performance differences between:

```text
1% → 5%
5% → 10%
10% → 20%
```

For example, if the improvement from 10% to 20% is substantially smaller than the improvement from 1% to 5%, this may indicate diminishing returns.

The conclusion must be based on the measured results.

---

# 17. Factors Affecting Model Performance

The experiment should distinguish between controlled and investigated factors.

### Investigated factors

```text
1. Point-label density
2. Random seed
```

### Controlled factors

```text
Model architecture
Learning rate
Optimizer
Batch size
Crop size
Dataset split
Data augmentation
Training configuration
Early stopping
Evaluation procedure
```

The controlled factors should remain unchanged across the experiments.

---

# 18. Final Report Structure

The final technical report should contain:

```text
1. Introduction
2. Objective
3. Dataset
4. Methodology
5. Partial Cross-Entropy
6. Sparse Point-Label Generation
7. U-Net Architecture
8. Experimental Design
9. Training Configuration
10. Evaluation Metrics
11. Experimental Results
12. Point-Density Analysis
13. Random-Seed Analysis
14. Per-Class Analysis
15. Training Curves
16. Discussion
17. Limitations
18. Conclusion
```

---

# 19. Limitations

The report must acknowledge the following limitations:

- Point labels are simulated from dense annotations.
- The point-sampling strategy is uniformly random.
- Simulated annotations do not fully represent human point annotation.
- The experiment uses RGB imagery.
- Results may depend on crop size and preprocessing.
- Results may depend on model architecture and training configuration.
- Only three random seeds are evaluated.
- A development-scale subset should not be presented as a definitive benchmark.
- Geographic differences in OpenEarthMap may affect generalization.

---

# 20. Expected Deliverables

The completed project should provide:

```text
├── src
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── utils/
│   └── main.py
│
├── results/
│   ├── per_run.csv
│   └── summary.csv
│
├── report/
│   ├── final_report.md
│   └── image.png
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# 21. Success Criteria

The project is considered complete when:

- [ ] Partial Cross-Entropy is implemented.
- [ ] Unlabeled pixels are ignored during loss computation.
- [ ] OpenEarthMap is loaded correctly.
- [ ] Sparse point labels are generated.
- [ ] U-Net is trained using sparse supervision.
- [ ] Dense validation masks are used for evaluation.
- [ ] Early stopping is implemented.
- [ ] Lightning metrics are logged correctly.
- [ ] 1%, 5%, 10% and 20% point densities are evaluated.
- [ ] Seeds 42, 43 and 44 are evaluated.
- [ ] mIoU is calculated correctly.
- [ ] Pixel accuracy is calculated.
- [ ] Macro precision is calculated.
- [ ] Macro recall is calculated.
- [ ] Macro F1 is calculated.
- [ ] Per-class IoU is reported.
- [ ] Mean and standard deviation are reported.
- [ ] Training curves are generated.
- [ ] Point-density comparison is generated.
- [ ] Random-seed robustness is analyzed.
- [ ] Final conclusions are based on measured results.
- [ ] Limitations are documented.

---
