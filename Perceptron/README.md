# Single Layer Perceptron — Banknote Authentication

**CS3807 – Deep Learning Laboratory, Experiment 1**
A single-layer perceptron implemented from scratch in NumPy to classify banknotes as authentic
or forged, following the lab manual's task structure.

---

## Files

| File | Description |
|---|---|
| `Experiment1_Perceptron.ipynb` | Main notebook — all tasks done, already executed with outputs |
| `data_banknote_authentication.txt` | Raw dataset (CSV, no header) |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook Experiment1_Perceptron.ipynb
```

Keep the `.txt` data file in the **same folder** as the notebook — it's loaded with a relative
path. Works the same way on Google Colab; just upload both files to the session.

---

## Dataset

Source: UCI ML Repository — Banknote Authentication Dataset.
1372 samples, 4 numeric features, no missing values.

| Column | Meaning |
|---|---|
| `variance`, `skewness`, `curtosis`, `entropy` | Statistical features of the Wavelet-transformed banknote image |
| `class` | 0 = Authentic, 1 = Forged |

Class balance: ~55.5% Authentic / ~44.5% Forged.

---

## What the notebook covers

1. **Dataset exploration** — shape, missing values, `.describe()`
2. **EDA** — histograms, correlation heatmap, scatter plot, boxplots, each with a short interpretation
3. **Preprocessing** — feature normalization (StandardScaler, fit on train only) + 80/20 stratified split
4. **Perceptron from scratch** — step activation, forward pass, weight/bias update rule, all in raw NumPy
5. **Training** — epoch-wise logging of error count, weights, bias; convergence + weight/bias evolution plots
6. **Evaluation** — accuracy, precision, recall, F1, confusion matrix
7. **Learning rate comparison** — 0.001 / 0.01 / 0.1
8. **Decision boundary plot** — 2-feature version, for visualization only
9. **Scikit-learn comparison** — `sklearn.linear_model.Perceptron` vs the from-scratch version
10. **Discussion** — Step vs Sigmoid, why XOR isn't solvable by a single-layer perceptron, why normalization helps convergence

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 98.55% |
| Precision | 96.83% |
| Recall | 100% |
| F1-score | 98.39% |

`variance` is the strongest predictor (correlation −0.725 with `class`); `entropy` is the
weakest (−0.023) — visible in both the correlation heatmap and boxplots.

---

## Notes

- Scaler is fit only on the training set, then applied to test — avoids data leakage.
- `stratify=y` in the split keeps the ~55/45 class ratio consistent across train/test.
- Learning rate has little effect here because the perceptron's update depends on the *sign*
  of the error, not its size — all three rates converge to similar error counts on this
  near-linearly-separable data.

---

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press, 2016.
3. UCI ML Repository — Banknote Authentication Dataset.
4. Scikit-learn Documentation — Perceptron.
