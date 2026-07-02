# Supplementary Table — Unsupervised fusion-weight strategies (all 30 categories)

All strategies evaluated across 6 discovery cycles (30 categories). **102 distinct methods**; Fixed-α=0.65 baseline AUROC = 95.226. No unsupervised strategy beats the fixed weight by more than +0.004 pp. Precise definitions are in the cited cycle script.

| Rank | Method | Cycle | Mean AUROC | Mean regret (pp) | MAE α vs oracle | Margin vs Fixed-0.65 (pp) | Defined in |
|---:|---|:--:|---:|---:|---:|---:|---|
| 1 | Trimmed-95 | C6 | 95.230 | 0.373 | -- | +0.0040 | `alpha_discovery_cycle6.py` |
| 2 | Fixed-0.65 **(fixed)** | C1 | 95.226 | 0.376 | 0.145 | +0.0000 | `unsupervised_alpha_discovery.py` |
| 3 | ConfGated-95 | C5 | 95.226 | 0.376 | -- | +0.0000 | `alpha_discovery_cycle5.py` |
| 4 | Ensemble-5 | C6 | 95.226 | 0.376 | -- | +0.0000 | `alpha_discovery_cycle6.py` |
| 5 | Ensemble-narrow | C6 | 95.226 | 0.376 | -- | +0.0000 | `alpha_discovery_cycle6.py` |
| 6 | SmoothAdapt-s0.1 | C6 | 95.198 | 0.403 | -- | -0.0280 | `alpha_discovery_cycle6.py` |
| 7 | Trimmed-90 | C6 | 95.153 | 0.449 | -- | -0.0730 | `alpha_discovery_cycle6.py` |
| 8 | Ensemble-7 | C6 | 95.147 | 0.455 | -- | -0.0790 | `alpha_discovery_cycle6.py` |
| 9 | Z-0.65 | C5 | 95.134 | 0.468 | -- | -0.0920 | `alpha_discovery_cycle5.py` |
| 10 | CDF-0.65 | C5 | 95.055 | 0.548 | -- | -0.1710 | `alpha_discovery_cycle5.py` |
| 11 | RankW-0.65 | C6 | 95.055 | 0.548 | -- | -0.1710 | `alpha_discovery_cycle6.py` |
| 12 | Agreement-weighted | C1 | 95.029 | 0.573 | 0.163 | -0.1970 | `unsupervised_alpha_discovery.py` |
| 13 | WeightedPCA | C4 | 95.015 | 0.588 | 0.179 | -0.2110 | `alpha_discovery_cycle4.py` |
| 14 | NormalDeviation-90 | C1 | 94.974 | 0.628 | 0.174 | -0.2520 | `unsupervised_alpha_discovery.py` |
| 15 | AnomalyRate-10% | C2 | 94.957 | 0.646 | 0.174 | -0.2690 | `alpha_discovery_cycle2.py` |
| 16 | AnomalyRate-15% | C2 | 94.951 | 0.651 | 0.177 | -0.2750 | `alpha_discovery_cycle2.py` |
| 17 | SmoothAdapt-s0.2 | C6 | 94.951 | 0.651 | -- | -0.2750 | `alpha_discovery_cycle6.py` |
| 18 | AnomalyRate-20% | C2 | 94.939 | 0.663 | 0.179 | -0.2870 | `alpha_discovery_cycle2.py` |
| 19 | NormalDeviation-80 | C1 | 94.930 | 0.672 | 0.180 | -0.2960 | `unsupervised_alpha_discovery.py` |
| 20 | AnomalyRate-25% | C2 | 94.922 | 0.679 | 0.182 | -0.3040 | `alpha_discovery_cycle2.py` |
| 21 | PowerMean-p0.5 | C6 | 94.917 | 0.685 | -- | -0.3090 | `alpha_discovery_cycle6.py` |
| 22 | ConfGated-90 | C5 | 94.906 | 0.696 | -- | -0.3200 | `alpha_discovery_cycle5.py` |
| 23 | NormalDeviation-70 | C1 | 94.902 | 0.701 | 0.186 | -0.3240 | `unsupervised_alpha_discovery.py` |
| 24 | AnomalyRate-30% | C2 | 94.901 | 0.702 | 0.186 | -0.3250 | `alpha_discovery_cycle2.py` |
| 25 | Bimodality-ratio | C1 | 94.897 | 0.705 | 0.186 | -0.3290 | `unsupervised_alpha_discovery.py` |
| 26 | PCA-weights | C4 | 94.890 | 0.713 | 0.198 | -0.3360 | `alpha_discovery_cycle4.py` |
| 27 | CondVar-50% | C4 | 94.880 | 0.722 | 0.182 | -0.3460 | `alpha_discovery_cycle4.py` |
| 28 | CondVar-30% | C4 | 94.820 | 0.782 | 0.189 | -0.4060 | `alpha_discovery_cycle4.py` |
| 29 | Concordance-20% | C2 | 94.819 | 0.783 | 0.197 | -0.4070 | `alpha_discovery_cycle2.py` |
| 30 | ScoreGradient | C4 | 94.816 | 0.786 | 0.214 | -0.4100 | `alpha_discovery_cycle4.py` |
| 31 | Concordance-30% | C2 | 94.749 | 0.854 | 0.218 | -0.4770 | `alpha_discovery_cycle2.py` |
| 32 | CrossPseudo-25% | C2 | 94.738 | 0.864 | 0.201 | -0.4880 | `alpha_discovery_cycle2.py` |
| 33 | RankCross-25% | C2 | 94.738 | 0.864 | 0.201 | -0.4880 | `alpha_discovery_cycle2.py` |
| 34 | MutualAUROC-direct | C2 | 94.736 | 0.866 | 0.191 | -0.4900 | `alpha_discovery_cycle2.py` |
| 35 | CrossReversed-10% | C2 | 94.733 | 0.869 | 0.187 | -0.4930 | `alpha_discovery_cycle2.py` |
| 36 | CrossReversed-15% | C2 | 94.729 | 0.873 | 0.188 | -0.4970 | `alpha_discovery_cycle2.py` |
| 37 | CrossReversed-30% | C2 | 94.724 | 0.878 | 0.195 | -0.5020 | `alpha_discovery_cycle2.py` |
| 38 | CrossReversed-20% | C2 | 94.721 | 0.880 | 0.192 | -0.5050 | `alpha_discovery_cycle2.py` |
| 39 | CrossPseudo-20% | C2 | 94.718 | 0.884 | 0.199 | -0.5080 | `alpha_discovery_cycle2.py` |
| 40 | CrossPseudo-30% | C2 | 94.718 | 0.884 | 0.195 | -0.5080 | `alpha_discovery_cycle2.py` |
| 41 | RankCross-20% | C2 | 94.718 | 0.884 | 0.199 | -0.5080 | `alpha_discovery_cycle2.py` |
| 42 | CrossReversed-25% | C2 | 94.710 | 0.892 | 0.195 | -0.5160 | `alpha_discovery_cycle2.py` |
| 43 | MutualAUROC-inv | C2 | 94.696 | 0.907 | 0.194 | -0.5300 | `alpha_discovery_cycle2.py` |
| 44 | PseudoAUROC-otsu | C1 | 94.679 | 0.923 | 0.197 | -0.5470 | `unsupervised_alpha_discovery.py` |
| 45 | PseudoAUROC-quantile | C1 | 94.679 | 0.923 | 0.197 | -0.5470 | `unsupervised_alpha_discovery.py` |
| 46 | PseudoAUROC-kmeans | C1 | 94.679 | 0.923 | 0.197 | -0.5470 | `unsupervised_alpha_discovery.py` |
| 47 | Fixed-0.50 **(fixed)** | C1 | 94.679 | 0.923 | 0.197 | -0.5470 | `unsupervised_alpha_discovery.py` |
| 48 | Concordance-40% | C2 | 94.679 | 0.923 | 0.226 | -0.5470 | `alpha_discovery_cycle2.py` |
| 49 | CrossPseudo-15% | C2 | 94.671 | 0.930 | 0.200 | -0.5550 | `alpha_discovery_cycle2.py` |
| 50 | RankCross-15% | C2 | 94.671 | 0.930 | 0.200 | -0.5550 | `alpha_discovery_cycle2.py` |
| 51 | RankAverage | C5 | 94.607 | 0.995 | -- | -0.6190 | `alpha_discovery_cycle5.py` |
| 52 | CDF-Equal | C5 | 94.607 | 0.995 | -- | -0.6190 | `alpha_discovery_cycle5.py` |
| 53 | GeometricMean | C5 | 94.596 | 1.006 | -- | -0.6300 | `alpha_discovery_cycle5.py` |
| 54 | MaxThen065 | C5 | 94.584 | 1.019 | -- | -0.6420 | `alpha_discovery_cycle5.py` |
| 55 | CrossPseudo-10% | C2 | 94.572 | 1.030 | 0.197 | -0.6540 | `alpha_discovery_cycle2.py` |
| 56 | RankCross-10% | C2 | 94.572 | 1.030 | 0.197 | -0.6540 | `alpha_discovery_cycle2.py` |
| 57 | PowerMean-p2 | C6 | 94.553 | 1.049 | -- | -0.6730 | `alpha_discovery_cycle6.py` |
| 58 | SmoothAdapt-s0.3 | C6 | 94.516 | 1.086 | -- | -0.7100 | `alpha_discovery_cycle6.py` |
| 59 | Z-Equal | C5 | 94.482 | 1.120 | -- | -0.7440 | `alpha_discovery_cycle5.py` |
| 60 | RawGMM | C2 | 94.447 | 1.154 | 0.212 | -0.7790 | `alpha_discovery_cycle2.py` |
| 61 | TopkContrib-10% | C4 | 94.440 | 1.162 | 0.226 | -0.7860 | `alpha_discovery_cycle4.py` |
| 62 | ScoreRatio-10% | C4 | 94.440 | 1.162 | 0.226 | -0.7860 | `alpha_discovery_cycle4.py` |
| 63 | GeomMean-0.65 | C6 | 94.412 | 1.190 | -- | -0.8140 | `alpha_discovery_cycle6.py` |
| 64 | PseudoAUROC-gmm | C1 | 94.399 | 1.203 | 0.214 | -0.8270 | `unsupervised_alpha_discovery.py` |
| 65 | PseudoAUROC-ensemble | C1 | 94.399 | 1.203 | 0.214 | -0.8270 | `unsupervised_alpha_discovery.py` |
| 66 | TopkContrib-20% | C4 | 94.274 | 1.328 | 0.244 | -0.9520 | `alpha_discovery_cycle4.py` |
| 67 | ScoreRatio-20% | C4 | 94.274 | 1.328 | 0.244 | -0.9520 | `alpha_discovery_cycle4.py` |
| 68 | RankConcentration | C4 | 94.192 | 1.410 | 0.253 | -1.0340 | `alpha_discovery_cycle4.py` |
| 69 | ElbowClarity | C4 | 94.183 | 1.418 | 0.243 | -1.0430 | `alpha_discovery_cycle4.py` |
| 70 | TopkContrib-30% | C4 | 94.143 | 1.459 | 0.257 | -1.0830 | `alpha_discovery_cycle4.py` |
| 71 | ScoreSep-70 | C3 | 94.080 | 1.522 | 0.303 | -1.1460 | `alpha_discovery_cycle3.py` |
| 72 | ScoreSep-80 | C3 | 93.878 | 1.723 | 0.339 | -1.3480 | `alpha_discovery_cycle3.py` |
| 73 | MedianScore-ratio | C1 | 93.872 | 1.730 | 0.282 | -1.3540 | `unsupervised_alpha_discovery.py` |
| 74 | PowerMean-p-1 | C6 | 93.788 | 1.814 | -- | -1.4380 | `alpha_discovery_cycle6.py` |
| 75 | KL-ratio | C1 | 93.703 | 1.899 | 0.274 | -1.5230 | `unsupervised_alpha_discovery.py` |
| 76 | ProbOR | C5 | 93.639 | 1.963 | -- | -1.5870 | `alpha_discovery_cycle5.py` |
| 77 | ProdComplement | C5 | 93.639 | 1.963 | -- | -1.5870 | `alpha_discovery_cycle5.py` |
| 78 | Fisher | C3 | 93.633 | 1.970 | 0.302 | -1.5930 | `alpha_discovery_cycle3.py` |
| 79 | RankMax | C5 | 93.564 | 2.039 | -- | -1.6620 | `alpha_discovery_cycle5.py` |
| 80 | Combined-GFS | C3 | 93.540 | 2.062 | 0.306 | -1.6860 | `alpha_discovery_cycle3.py` |
| 81 | L2-norm | C5 | 93.534 | 2.068 | -- | -1.6920 | `alpha_discovery_cycle5.py` |
| 82 | PowerMean-p3 | C6 | 93.496 | 2.107 | -- | -1.7300 | `alpha_discovery_cycle6.py` |
| 83 | ScoreSep-90 | C3 | 93.449 | 2.152 | 0.368 | -1.7770 | `alpha_discovery_cycle3.py` |
| 84 | CalinskiHarabasz | C3 | 93.445 | 2.157 | 0.335 | -1.7810 | `alpha_discovery_cycle3.py` |
| 85 | GMM-AshmanD | C3 | 93.276 | 2.326 | 0.333 | -1.9500 | `alpha_discovery_cycle3.py` |
| 86 | TopkGap-10% | C3 | 93.225 | 2.377 | 0.354 | -2.0010 | `alpha_discovery_cycle3.py` |
| 87 | LogSumExp-T3 | C5 | 93.170 | 2.431 | -- | -2.0560 | `alpha_discovery_cycle5.py` |
| 88 | TopkGap-20% | C3 | 93.047 | 2.555 | 0.371 | -2.1790 | `alpha_discovery_cycle3.py` |
| 89 | Silhouette | C3 | 92.978 | 2.625 | 0.383 | -2.2480 | `alpha_discovery_cycle3.py` |
| 90 | Clayton-Copula | C6 | 92.753 | 2.849 | -- | -2.4730 | `alpha_discovery_cycle6.py` |
| 91 | Bimodality | C3 | 92.313 | 3.289 | 0.386 | -2.9130 | `alpha_discovery_cycle3.py` |
| 92 | SoftmaxAdapt-T2 | C5 | 92.279 | 3.323 | -- | -2.9470 | `alpha_discovery_cycle5.py` |
| 93 | LogSumExp-T5 | C5 | 92.176 | 3.425 | -- | -3.0500 | `alpha_discovery_cycle5.py` |
| 94 | ExtremityWeighted | C5 | 91.808 | 3.794 | -- | -3.4180 | `alpha_discovery_cycle5.py` |
| 95 | LogSumExp-T10 | C5 | 90.825 | 4.777 | -- | -4.4010 | `alpha_discovery_cycle5.py` |
| 96 | DipStat | C3 | 90.258 | 5.344 | 0.488 | -4.9680 | `alpha_discovery_cycle3.py` |
| 97 | Max | C5 | 89.882 | 5.720 | -- | -5.3440 | `alpha_discovery_cycle5.py` |
| 98 | Negentropy | C3 | 89.563 | 6.039 | 0.496 | -5.6630 | `alpha_discovery_cycle3.py` |
| 99 | SoftmaxAdapt-T5 | C5 | 89.501 | 6.101 | -- | -5.7250 | `alpha_discovery_cycle5.py` |
| 100 | ScoreProportional | C5 | 89.495 | 6.107 | -- | -5.7310 | `alpha_discovery_cycle5.py` |
| 101 | QuantileAdaptive | C5 | 89.491 | 6.111 | -- | -5.7350 | `alpha_discovery_cycle5.py` |
| 102 | DisagreeSwitch | C5 | 85.094 | 10.508 | -- | -10.1320 | `alpha_discovery_cycle5.py` |

## Cycle / paradigm map

- **C1** (14 methods) — Individual-model signals & pseudo-labeling. Source: `src/unsupervised_alpha_discovery.py`.
- **C2** (25 methods) — Cross-model pseudo-labeling & anomaly-rate priors. Source: `src/alpha_discovery_cycle2.py`.
- **C3** (13 methods) — Direct unsupervised metric optimization (separation/Fisher/silhouette). Source: `src/alpha_discovery_cycle3.py`.
- **C4** (12 methods) — Conditional score analysis (gradient/top-k/PCA/cond-variance). Source: `src/alpha_discovery_cycle4.py`.
- **C5** (23 methods) — Non-linear fusion, sample-adaptive weighting & score calibration. Source: `src/alpha_discovery_cycle5.py`.
- **C6** (15 methods) — Power-mean family, alpha-ensembles & trimmed fusion. Source: `src/alpha_discovery_cycle6.py`.
