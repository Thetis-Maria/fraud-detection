# Credit Card Fraud Detection

Classification on a highly imbalanced dataset (0.17% fraud). The focus of this
project is **evaluation methodology**: why accuracy and ROC-AUC mislead on
imbalanced data, why the Precision-Recall curve is the right tool, and why
tuning the decision threshold can outperform resampling techniques like SMOTE.

## Dataset

[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions, of which only 492 are fraudulent (0.17%). Features
`V1`–`V28` are anonymised PCA components; `Time` and `Amount` are the only raw
columns.

## Key findings

**1. Accuracy is a trap.**
A plain Logistic Regression reaches 99.92% accuracy — but a model that always
predicts "not fraud" already scores 99.83% without looking at a single
feature. Behind that accuracy, the baseline misses 34 of 98 frauds
(recall = 0.65).

**2. ROC-AUC flatters; PR tells the truth.**
The same model scores ROC-AUC = 0.959 but Average Precision = 0.737. Because
FPR divides false positives by the ~57,000 normal transactions, even large
increases in false alarms barely move the ROC curve. Precision divides by the
model's own positive predictions, so it stays sensitive. On extreme imbalance,
the PR curve is the honest metric.

**3. Class weights and SMOTE are equivalent here.**
Both push recall to ~0.91 and precision down to ~0.055 — nearly identical
results. Both force the model to treat the classes as balanced, and for a
linear model they land in almost the same place. `class_weight="balanced"` is
preferable in practice: same result without doubling the training set size.

**4. Threshold tuning beats resampling (the headline result).**
Taking the *original, undistorted* baseline model and lowering its decision
threshold to 0.024 (target: recall ≥ 0.85) gives a far better operating point
than either resampling method:

| Approach                       | Recall | Precision | FP    | FN | F1 (fraud) |
|--------------------------------|:------:|:---------:|:-----:|:--:|:----------:|
| Baseline LR (threshold 0.5)    | 0.65   | 0.83      | 13    | 34 | 0.73       |
| LR class_weight                | 0.91   | 0.055     | 1,517 | 9  | 0.10       |
| LR + SMOTE                     | 0.92   | 0.054     | 1,576 | 8  | 0.10       |
| **Baseline + tuned threshold** | 0.88   | **0.558** | **68** | 13 | **0.68**   |

For essentially the same recall, threshold tuning produces **~22× fewer false
positives** than resampling (68 vs ~1,500). The reason: class weights and
SMOTE distort the model's probability estimates, while threshold tuning keeps
a well-calibrated model and simply moves the cut-off.

**Take-home:** on imbalanced problems, *where you set the threshold* often
matters more than *how you train*.

## Structure

```
├── data/               # creditcard.csv (not committed, ~150 MB)
├── notebook/
│   ├── 01_eda.ipynb    # class imbalance, amount & time patterns
│   ├── 02_baseline.ipynb  # accuracy trap, PR vs ROC
│   └── 03_imbalance.ipynb #class weights,SMOTE, threshold tuning
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate # Windows (source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

Download `creditcard.csv` from the Kaggle link above and place it in `data/`.

## EDA highlights

- **Fraud amounts are not large.** Median fraud amount (€9.25) is *lower* than
  normal (€22.00), and 25% of frauds are ≤ €1 — consistent with card-testing
  behaviour, where a stolen card is first probed with a tiny charge.
- **Frauds ignore the daily cycle.** Normal transactions dip sharply overnight
  (~2–7 AM); frauds do not, spiking around 2 AM when legitimate activity is at
  its lowest. (Caveat: the dataset spans only two days.)

## Roadmap

- [ ] Refactor notebook logic into a `src/` package with CLI and pytest tests
- [ ] Add non-linear models (Random Forest, XGBoost), where SMOTE may help more
- [ ] Stratified k-fold cross-validation instead of a single split
- [ ] Cost-based evaluation (assign € cost to FP vs FN)