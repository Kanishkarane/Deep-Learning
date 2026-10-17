# Effect of Attention Type and Attention Position in a Pretrained ResNet50

**CS3807 – Deep Learning Laboratory · Experiments 8 & 9**
Shiv Nadar University Chennai · B.Tech AI & Data Science · Semester V · AY 2026–27

This repository contains the complete implementation and report for a controlled ablation study of
attention mechanisms inserted into an ImageNet-pretrained **ResNet50**, trained and evaluated on
**TensorFlow Flowers** (`tf_flowers`, 5 classes, 3,670 images).

The study covers:

- attention **types**: channel (SE), spatial, channel→spatial, self-attention, a combined module, ECA, CBAM and Coordinate Attention;
- attention **positions**: P1 (after conv2_x), P2 (conv3_x), P3 (conv4_x) and P4 (conv5_x);
- **multi-stage** attention, attention **order**, and spatial **kernel size**;
- **qualitative** analysis: Grad-CAM, internal attention maps and attention maps at each position.

---

## Repository structure

```
.
├── CS3807_Exp8_9_Attention_ResNet50_M4_Mac.ipynb   # the full experiment (runs top to bottom)
├── requirements.txt                                 # exact package versions used
├── README.md                 
├── CS3807_Exp8_9_Report.pdf                     # compiled report
└── images/                                      # all figures used in the report
```

When the notebook runs, it creates `lab_outputs_mac/` next to itself. This folder holds the dataset cache, the fixed split, `results.json`, the model weights and the figures. It is not part of the repository; see the notes below.

---

## Setup

The experiment was run on a **MacBook Air (Apple M4)** using the Apple GPU through `tensorflow-metal`.
The Metal plugin only works with the exact versions in `requirements.txt`, and it **requires Python 3.12**.

```bash
# macOS (Apple silicon), from the repository folder
/opt/homebrew/bin/python3.12 -m venv .venv     # install with: brew install python@3.12
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# check that the GPU is visible (must print one GPU device)
python -c "import setuptools, tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

**Linux / Windows / Colab:** the same `requirements.txt` works. `tensorflow-metal` is skipped automatically, and TensorFlow uses CUDA if it is available, otherwise the CPU.

---

## How to run

1. Open the notebook in VS Code or Jupyter and select the `.venv` (Python 3.12) kernel.
2. Run the cells **from top to bottom**. The first cells check the environment and run a quick GPU-versus-CPU self-test.
3. The first run downloads the flowers dataset (≈220 MB) and the ImageNet ResNet50 weights automatically.
4. To test the pipeline first, set `QUICK_MODE = True` in the configuration cell (2 + 2 epochs per model; results go to a separate sub-folder). Then set it back to `False`, restart the kernel and *Run All*.
5. Finished experiments are saved and **skipped on re-runs**, so you can stop and continue later. An interrupted experiment restarts from epoch 1.

On a MacBook Air M4 one experiment took about 13–30 min. All 17 trained configurations took roughly 6–7 hours, and the 7 mandatory ones about 2.5 hours. A fan-less laptop throttles when it gets hot, so keep it plugged in, on a hard surface, with the lid open. Run `caffeinate -dis` in a Terminal to stop the Mac from sleeping.

---

## Experimental protocol (identical for every model)

| Setting | Value |
|---|---|
| Backbone | Keras ResNet50, ImageNet weights, `include_top=False`, input 224×224×3 |
| Head | GlobalAveragePooling → Dropout(0.3) → Dense(5) → softmax |
| Split | stratified 70 / 15 / 15 (2,569 / 550 / 551), seed 42, saved and reused |
| Preprocessing | `resnet50.preprocess_input` |
| Augmentation (train only) | horizontal flip, rotation 0.05, zoom 0.1, translation 0.1, contrast 0.1 |
| Phase 1 | backbone frozen, 10 epochs, Adam, LR 1e-3 |
| Phase 2 | conv5_x unfrozen (BatchNorm frozen), 10 epochs, Adam, LR 1e-4 |
| LR schedule | ReduceLROnPlateau(val_loss, factor 0.5, patience 2) |
| Model selection | best-validation-loss weights restored; test set used once |
| Metrics | accuracy, macro precision / recall / F1, parameters, inference time per image |

---

## Results (test set, 551 images)

| Model | Position | Accuracy | Macro F1 | Attention params | Inference (ms/img) |
|---|---|---|---|---|---|
| Baseline ResNet50 | – | 0.9437 | 0.9436 | 0 | 12.85 |
| Channel (SE) | P3 | 0.9183 | 0.9192 | 132,160 | 18.86 |
| Spatial | P3 | 0.9347 | 0.9340 | 99 | 15.78 |
| Channel→Spatial | P2 | **0.9510** | **0.9510** | 33,411 | 16.57 |
| Channel→Spatial | P3 | 0.9238 | 0.9238 | 132,259 | 18.56 |
| Channel→Spatial | P4 | 0.9310 | 0.9317 | 526,563 | 16.67 |
| Self-Attention | P3 | 0.9419 | 0.9411 | 1,310,721 | 17.03 |
| ECA | P3 | 0.9474 | 0.9485 | 5 | 19.06 |
| CBAM | P3 | 0.9220 | 0.9214 | 132,259 | 18.37 |
| Coordinate Attention | P3 | 0.9129 | 0.9135 | 100,512 | 18.88 |
| Combined (Ch→Sp→Self) | P3 | 0.9111 | 0.9102 | 1,442,980 | 17.24 |
| Channel→Spatial | P1 | 0.9093 | 0.9079 | 8,563 | 16.85 |
| Channel→Spatial | P3+P4 | 0.9383 | 0.9382 | 658,822 | 17.48 |
| Channel→Spatial | P2+P3+P4 | 0.8947 | 0.8946 | 692,233 | 17.69 |
| Spatial→Channel | P3 | 0.9328 | 0.9335 | 132,259 | 17.72 |
| Channel→Spatial, k = 3 | P3 | 0.9056 | 0.9040 | 132,179 | 18.46 |
| Channel→Spatial, k = 5 | P3 | 0.9165 | 0.9162 | 132,211 | 17.30 |

With 551 test images, the 95 % confidence interval of an accuracy is about ±2.2 pp. Differences smaller than **≈3.1 pp** between two models are therefore not statistically meaningful.

### Key findings

- **No attention type at P3 beat the baseline.** Self-attention, spatial attention and ECA matched it within noise; the Combined module was clearly worse.
- **Position matters as much as type.** The same Channel→Spatial module ranged from 0.9093 (P1) to **0.9510 (P2, the best model)**.
- **More is not better.** P2+P3+P4 gave the worst model (0.8947), and the 5-parameter ECA matched or beat every heavier mechanism.
- **Order and kernel:** Spatial→Channel was slightly better than Channel→Spatial, and a 7×7 spatial kernel beat 5×5 and 3×3.
- **Qualitative:** Grad-CAM uses the flower when a prediction is correct and background colour cues when it is wrong. The internal attention maps were largely input-independent, which explains the small gains.
- **Cost:** only self-attention adds real computation (+7.4 % at P3). Measured inference times were inflated by thermal throttling of the fan-less laptop.

---

## Notes

- **Do not commit `lab_outputs_mac/`.** Each model's weights file is about 95 MB, close to GitHub's 100 MB per-file limit, and about 1.6 GB in total. Add it to `.gitignore`:
  ```
  lab_outputs_mac/
  .venv/
  ```
- Every configuration was trained once (seed 42). Repeated runs with other seeds would be needed to rank models whose results lie within the noise band.
- The optional configuration self-attention at P4 was not trained.

---

## References

1. K. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
2. J. Hu et al., "Squeeze-and-Excitation Networks," IEEE TPAMI 2020.
3. S. Woo et al., "CBAM: Convolutional Block Attention Module," ECCV 2018.
4. Q. Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks," CVPR 2020.
5. Q. Hou et al., "Coordinate Attention for Efficient Mobile Network Design," CVPR 2021.
6. A. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
7. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV 2017.
8. TensorFlow, Keras and TensorFlow Datasets documentation.
