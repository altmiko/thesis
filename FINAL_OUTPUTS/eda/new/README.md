# PCA and t-SNE EDA

Deterministic descriptive embeddings for the two active DistriNet datasets.

## Stage definitions

- **Before preprocessing:** sampled directly from the corrected raw CSVs after only the five-category label mapping and exclusion of non-finite rows required by PCA/t-SNE. No production cleaning, deduplication, split, class-size reduction, or production scaler is applied.
- **After preprocessing:** sampled across the final train/validation/test `X_*.npy` arrays after production filtering, category mapping, float32 feature+category deduplication, chronological within-source-label splitting, train-fitted RobustScaler transformation, and (for 2018) class-size reduction.
- Sampling is deterministic bottom-K sampling, stratified by category. PCA uses at most 5,000 rows per category; t-SNE uses at most 1,500. Class-stratified scatter density therefore does **not** encode natural prevalence.
- Before and after embeddings are fitted independently. Their axes are not a shared coordinate system; compare class overlap/separation, not absolute coordinates.

## Outputs

Each `<dataset>/<stage>/` directory contains `pca.png`, `tsne.png`, PCA loadings, and compressed plotted coordinates.

- `cicids2017/{before_preprocessing,after_preprocessing}/`
- `cicids2018/{before_preprocessing,after_preprocessing}/`
- `cicids2018_data_reduction_showcase.png`: exact cleaned/deduplicated pre-reduction vs final counts (62,393,756 → 833,552; 1.34% retained).
- `cicids2018_reduction_counts.csv`: count source for the reduction figure.
- `metadata.json`: sample populations, PCA variance, t-SNE KL divergence, methods, and arguments.

## Reproduce

```powershell
$Env:PYTHONPATH = 'src'
python scripts/generate_final_eda_embeddings.py
```

These figures are descriptive only. They do not fit or alter any classifier, constraint, validator, or attack artifact.
