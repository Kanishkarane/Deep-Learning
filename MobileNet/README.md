# Experiment 5 — CNN Training, Regularization, Optimization & Transfer Learning (MobileNetV2 / Oxford-IIIT Pets)

CS3807 Deep Learning Laboratory — Experiment 5.

## Overview
Comprehensive study of CNN training using MobileNetV2 (ImageNet pretrained) on the Oxford-IIIT Pet dataset (37 breeds, 224×224×3). Covers weight initialization, regularization, batch normalization, optimizers, hyperparameter tuning, transfer learning / fine-tuning, and 5-fold cross-validation based model selection.

## Setup

```bash
pip install -r requirements.txt
```

Run `Experiment5_MobileNetV2_Pets.ipynb` top to bottom in Jupyter/Colab (GPU runtime recommended).

The Oxford-IIIT Pet dataset is downloaded automatically via direct tarball download (no `tensorflow_datasets` dependency, to avoid protobuf version conflicts).

## Contents

| Section | Topic |
|---|---|
| 4 | MobileNetV2 architecture |
| 5 | Weight initialization (Zero, Random, Xavier, He) |
| 6 | Regularization (None, L2, Dropout, BatchNorm) |
| 7 | Batch Normalization numerical example |
| 8 | Optimizers (SGD, Momentum, RMSProp, Adam) |
| 9 | Hyperparameter tuning (LR, batch size, dropout) |
| 10 | Transfer learning: feature extraction vs. fine-tuning |
| 11 | 5-fold cross-validation |
| 12 | Final model evaluation on held-out test set |
| 13 | Overall results summary |
| 14–17 | Inferences, discussion questions, additional exercise, expected outcome |

## Known Issue

`base_model` is left partially unfrozen after the Section 10 fine-tuning cell and is not re-frozen before Section 11/12, causing unintended further fine-tuning at too high a learning rate. This produces a large CV-vs-test accuracy gap (~93% CV vs. ~61% test). Fix: add `base_model.trainable = False` at the start of the Section 11 and Section 12 cells before rebuilding the model.

## Requirements
See `requirements.txt`.
