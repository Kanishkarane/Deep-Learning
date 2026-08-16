## Comparative Study of Deep CNN Architectures Using Transfer Learning

## Overview

This experiment studies the evolution of deep CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet,
ResNet50) and applies transfer learning with a VGG16 backbone pretrained on ImageNet to classify
CIFAR-10 images. LeNet-5, AlexNet, and a scaled-down GoogleNet are also built and trained from scratch,
and ResNet50 transfer learning is repeated separately, so the final architecture comparison uses real
measured results rather than textbook figures alone.

## Files

| File                                     | Description                                                                                                          |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `CNN_TransferLearning_Experiment4.ipynb` | Full Jupyter notebook — all tasks, hyperparameter study, mandatory plots, discussion questions, additional exercises |
| `requirements.txt`                       | Python package dependencies                                                                                          |

## Notebook Contents

1. **Task 1 — Dataset Preparation**: load CIFAR-10, normalize to [0,1], display samples, print shapes
2. **Architecture Overview**: conceptual comparison of LeNet-5, AlexNet, VGG16, GoogleNet, ResNet50
3. **Task 2 — Transfer Learning Setup**: VGG16 backbone, frozen base, GAP + Dense + Softmax head
4. **Task 3 — Model Training**: Adam, lr=0.001, batch=32, 15 epochs
5. **Task 4 — Fine Tuning**: unfreeze `block5`, train 8 more epochs at lr=1e-5, before/after comparison
6. **Task 5 — Model Evaluation**: accuracy, precision, recall, F1, confusion matrix, classification report
7. **Hyperparameter Study**: learning rate, batch size, optimizer, dense units, frozen-vs-partial
8. **Mandatory Plots**: sample images, accuracy/loss curves, confusion matrix, misclassified images
9. **Discussion Questions**: all 10 answered
10. **Additional Exercises (Section 20)**:
    - LeNet-5 built and trained from scratch
    - AlexNet (CIFAR-adapted) built and trained from scratch
    - GoogleNet (scaled-down, 2 Inception modules) built and trained from scratch
    - ResNet50 transfer learning, mirroring the VGG16 pipeline
    - Adam vs SGD controlled comparison
    - Final 5-architecture comparison table with measured accuracy/parameters/training time

## How to Run

1. Open `CNN_TransferLearning_Experiment4.ipynb` in **Google Colab**.
2. Set the runtime to GPU: `Runtime → Change runtime type → T4 GPU`.
3. Run all cells top to bottom (`Runtime → Run all`). No local setup or `requirements.txt` install is
   needed on Colab — TensorFlow, numpy, matplotlib, seaborn, scikit-learn, and pandas are preinstalled.
4. Full run time is roughly 45–75 minutes on a T4 GPU, depending on Colab's current load, since the
   notebook trains six separate models (VGG16, ResNet50, LeNet-5, AlexNet, GoogleNet, plus the Adam/SGD
   and hyperparameter-study runs).

## Key Results

| Model                     | Parameters | Test Accuracy (%) | Training Time (s) |
| ------------------------- | ---------- | ----------------- | ----------------- |
| LeNet-5                   | 83,126     | 51.08             | 98.2              |
| AlexNet (CIFAR adapted)   | 6,976,842  | 71.36             | 297.6             |
| GoogleNet (CIFAR adapted) | 673,978    | 75.31             | 211.8             |
| VGG16 (transfer)          | 14,848,586 | 83.39             | 1062.4            |
| ResNet50 (transfer)       | 24,114,826 | 81.39             | 729.9             |

## Known Issues Fixed During Development

- **Double-normalization collapse**: feeding `[0,1]`-normalized images into Keras's `preprocess_input`
  (which expects raw `[0,255]` input) caused the model to collapse to predicting a single class for
  every input. Fixed with a `Rescaling(255.0)` layer before `preprocess_input` for both VGG16 and
  ResNet50 branches.
- **Blank plot exports**: calling `plt.show()` before `plt.savefig()` (or before a second `savefig` for
  a different format, e.g. `.eps`) produces a blank saved image. All `savefig` calls must precede
  `plt.show()`.
- **Full-base unfreezing collapse**: unfreezing the entire pretrained backbone with a relatively large
  learning rate (as in one hyperparameter-study configuration) destroyed the pretrained weights and
  collapsed validation accuracy to ~10% (random-guessing baseline). Proper fine tuning (Task 4) avoids
  this by unfreezing only the last block and using a much smaller learning rate (1e-5).
