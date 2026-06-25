# SERVE-SC pipeline results

> SYNTHETIC DEMONSTRATION RESULTS -- computed on generated data. These are NOT the paper's numbers and do not reproduce them. Supply the real datasets (see DATA_SCHEMA.md) to reproduce reported tables.

- backend: `numpy-fallback`
- contracts: 400 (train/dev/test = 240/80/80)

## Detection
- macro-F1: 0.989
- macro-AUC: 1.000

## Prioritisation
- HR@5: 0.085  (95% CI [0.059, 0.122])
- HR@10: 0.170  (95% CI [0.111, 0.232])
- NDCG@5: 0.869  (95% CI [0.616, 1.000])
- NDCG@10: 0.849  (95% CI [0.599, 1.000])
- Spearman: 0.458  (95% CI [0.271, 0.645])

## Risk-model coefficients (bootstrap)
| term | median | 90% range | excludes 0 |
|------|--------|-----------|------------|
| intercept | -3.99 | [-6.91, -1.71] | True |
| u | +1.47 | [-0.55, +4.32] | False |
| e | -1.65 | [-3.68, +0.70] | False |
| chi | +0.70 | [-0.55, +2.05] | False |
| q | +0.90 | [-0.19, +2.08] | False |
| p | +0.89 | [-2.15, +4.18] | False |
| u*e | +3.69 | [+1.61, +6.35] | True |
| u*chi | +2.51 | [+1.02, +4.35] | True |
