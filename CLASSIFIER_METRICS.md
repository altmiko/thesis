# Classifier evaluation metrics — formulae

Scope: the victim classifiers (`SimpleMLP`, `CNNOnly`, `FTTransformer`; binary and category heads)
trained on CICIDS2017-DistriNet and CSE-CIC-IDS-2018-DistriNet.

Ground truth: `src/classifiers/cicids2017d_experiments.py:compute_metrics`. The CICIDS2018 multi-seed
driver (`scripts/train_cicids2018_classifiers_multiseed.py`) runs that same script per seed, so the
formulae are identical for both datasets.

## Notation

- Test set $\{(x_i, y_i)\}_{i=1}^{N}$, classes $c \in \{1,\dots,K\}$.
  - Category head: $K = 5$ (`Benign, DoS, DDoS, Recon, BruteForce`).
  - Binary head: $K = 2$ (`Benign, Attack`).
- Prediction: $\hat y_i = \arg\max_c \operatorname{softmax}(f(x_i))_c$ (argmax over the head's logits;
  the binary head is a 2-logit softmax too, so there is **no tuned decision threshold**).
- Per-class counts from the confusion matrix:
  - $TP_c = \sum_i \mathbb{1}[y_i = c \wedge \hat y_i = c]$
  - $FP_c = \sum_i \mathbb{1}[y_i \ne c \wedge \hat y_i = c]$
  - $FN_c = \sum_i \mathbb{1}[y_i = c \wedge \hat y_i \ne c]$
  - Support $n_c = TP_c + FN_c$.

## Test accuracy

`sklearn.metrics.accuracy_score(y_true, y_pred)`

$$
\mathrm{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat y_i = y_i] = \frac{\sum_{c} TP_c}{N}
$$

Dominated by the majority class (Benign) under class imbalance.

## Balanced accuracy

`sklearn.metrics.balanced_accuracy_score(y_true, y_pred)` (default `adjusted=False`)

$$
\mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \qquad
\mathrm{BalAcc} = \frac{1}{K}\sum_{c=1}^{K} \mathrm{Recall}_c
$$

i.e. the unweighted mean of per-class recall (= macro recall). Each class counts equally regardless of
support. For the binary head this is $\tfrac12(\mathrm{TPR} + \mathrm{TNR})$.

Edge case: sklearn averages only over classes with $n_c > 0$ in `y_true`; all classes are present in
every test split, so the mean is over all $K$ classes.

## Macro-F1

`sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)`

$$
P_c = \frac{TP_c}{TP_c + FP_c}, \qquad
R_c = \frac{TP_c}{TP_c + FN_c}, \qquad
F1_c = \frac{2 P_c R_c}{P_c + R_c} = \frac{2\,TP_c}{2\,TP_c + FP_c + FN_c}
$$

$$
\mathrm{Macro\text{-}F1} = \frac{1}{K}\sum_{c=1}^{K} F1_c
$$

- Mean of **per-class** F1 — not the harmonic mean of macro-precision and macro-recall.
- `zero_division=0`: an undefined ratio (e.g. class never predicted, $TP_c + FP_c = 0$) contributes
  $0$ rather than raising.
- Binary head: macro over **both** `Benign` and `Attack`, not the positive-class-only F1.
- Checkpoint selection uses **validation** macro-F1 (validation loss as tie-break); test metrics are
  computed once on the selected checkpoint.

## Multi-seed aggregation (CICIDS2018 victims)

Per metric $m$ over training seeds $s \in S$ (42/123/2024):

$$
\bar m = \frac{1}{|S|}\sum_{s} m_s, \qquad
\mathrm{sd}(m) = \sqrt{\frac{1}{|S|-1}\sum_{s}(m_s - \bar m)^2}
$$

(sample standard deviation, `ddof=1`), reported as mean ± sd.
