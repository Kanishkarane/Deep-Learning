# CS3807 – Experiment 2: MLP for Fashion-MNIST Classification

Implementation of a Multi-Layer Perceptron (MLP) for multi-class image
classification on the Fashion-MNIST dataset, including automated
hyperparameter optimization using RandomizedSearchCV + SciKeras.

## Contents

- `CS3807_Experiment2_MLP_FashionMNIST.ipynb` — main notebook (all tasks: dataset exploration, preprocessing, model construction, training, evaluation, hyperparameter search, plots, results, discussion)
- `requirements.txt` — Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

Then launch:

```bash
jupyter notebook CS3807_Experiment2_MLP_FashionMNIST.ipynb
```

(Google Colab also works — just upload the notebook and run; Colab already has TensorFlow installed, so you only need `pip install scikeras`.)

## What the notebook does

1. **Dataset Exploration** – loads Fashion-MNIST, prints shapes, displays sample images, plots class distribution
2. **Preprocessing** – flattens images (28×28 → 784), normalizes pixels to [0,1], one-hot encodes labels
3. **Model Construction** – builds `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
4. **Training** – Adam optimizer, categorical cross-entropy loss, 20 epochs, batch size 32
5. **Evaluation** – accuracy, precision, recall, F1-score, confusion matrix, classification report
6. **Hyperparameter Optimization** – RandomizedSearchCV (5-fold CV) over hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, dropout
7. **Comparison** – baseline vs. optimized model, with all mandatory plots and inference notes
8. **Discussion & Conclusion** – written sections answering the manual's discussion questions

## Notes

- Hyperparameter search runs on a 10% subsample of the training set to keep runtime reasonable; the final optimized model is retrained on the full training set.
- The tunable model uses `sparse_categorical_crossentropy` (accepts integer labels) so it's compatible with `StratifiedKFold`'s cross-validation splitting; the baseline model uses `categorical_crossentropy` with one-hot labels as specified in the manual.
- Increase `N_ITER` and `CV_FOLDS` in the search cell for a more thorough (but slower) search if you have GPU access.
