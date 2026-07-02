"""Compute paradigm-level statistics for the exhaustive search table."""
import json
import os
import numpy as np

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

paradigm_assignment = {
    # Cycle 1
    "Agreement-weighted": "Cross-model agreement",
    "NormalDeviation-90": "Distributional signals",
    "NormalDeviation-80": "Distributional signals",
    "NormalDeviation-70": "Distributional signals",
    "Bimodality-ratio": "Distributional signals",
    "PseudoAUROC-otsu": "Pseudo-labeling",
    "PseudoAUROC-quantile": "Pseudo-labeling",
    "PseudoAUROC-kmeans": "Pseudo-labeling",
    "PseudoAUROC-gmm": "Pseudo-labeling",
    "PseudoAUROC-ensemble": "Pseudo-labeling",
    "MedianScore-ratio": "Distributional signals",
    "KL-ratio": "Distributional signals",
    # Cycle 2
    "AnomalyRate-15%": "Anomaly rate prior",
    "AnomalyRate-10%": "Anomaly rate prior",
    "AnomalyRate-20%": "Anomaly rate prior",
    "AnomalyRate-25%": "Anomaly rate prior",
    "AnomalyRate-30%": "Anomaly rate prior",
    "Concordance-20%": "Cross-model agreement",
    "Concordance-30%": "Cross-model agreement",
    "Concordance-40%": "Cross-model agreement",
    "MutualAUROC-direct": "Cross-model agreement",
    "MutualAUROC-inv": "Cross-model agreement",
    "CrossPseudo-25%": "Cross-model agreement",
    "CrossPseudo-20%": "Cross-model agreement",
    "CrossPseudo-30%": "Cross-model agreement",
    "CrossPseudo-15%": "Cross-model agreement",
    "CrossPseudo-10%": "Cross-model agreement",
    "RankCross-25%": "Cross-model agreement",
    "RankCross-20%": "Cross-model agreement",
    "RankCross-15%": "Cross-model agreement",
    "RankCross-10%": "Cross-model agreement",
    "CrossReversed-15%": "Cross-model agreement",
    "CrossReversed-30%": "Cross-model agreement",
    "CrossReversed-10%": "Cross-model agreement",
    "CrossReversed-20%": "Cross-model agreement",
    "CrossReversed-25%": "Cross-model agreement",
    "RawGMM": "Pseudo-labeling",
    # Cycle 3
    "ScoreSep-70": "Direct metric optimization",
    "ScoreSep-80": "Direct metric optimization",
    "ScoreSep-90": "Direct metric optimization",
    "Fisher": "Direct metric optimization",
    "Combined-GFS": "Direct metric optimization",
    "CalinskiHarabasz": "Direct metric optimization",
    "TopkGap-20%": "Direct metric optimization",
    "TopkGap-10%": "Direct metric optimization",
    "GMM-AshmanD": "Direct metric optimization",
    "Silhouette": "Direct metric optimization",
    "Bimodality": "Direct metric optimization",
    "DipStat": "Direct metric optimization",
    "Negentropy": "Direct metric optimization",
    # Cycle 4
    "WeightedPCA": "Conditional score analysis",
    "PCA-weights": "Conditional score analysis",
    "CondVar-50%": "Conditional score analysis",
    "ScoreGradient": "Conditional score analysis",
    "CondVar-30%": "Conditional score analysis",
    "TopkContrib-10%": "Conditional score analysis",
    "ScoreRatio-10%": "Conditional score analysis",
    "TopkContrib-20%": "Conditional score analysis",
    "ScoreRatio-20%": "Conditional score analysis",
    "RankConcentration": "Conditional score analysis",
    "TopkContrib-30%": "Conditional score analysis",
    "ElbowClarity": "Conditional score analysis",
    # Cycle 5
    "ConfGated-95": "Sample-level adaptive",
    "ConfGated-90": "Sample-level adaptive",
    "ScoreProportional": "Sample-level adaptive",
    "SoftmaxAdapt-T2": "Sample-level adaptive",
    "SoftmaxAdapt-T5": "Sample-level adaptive",
    "QuantileAdaptive": "Sample-level adaptive",
    "ExtremityWeighted": "Sample-level adaptive",
    "Z-0.65": "Score calibration",
    "Z-Equal": "Score calibration",
    "CDF-0.65": "Score calibration",
    "CDF-Equal": "Score calibration",
    "Max": "Non-linear fusion",
    "GeometricMean": "Non-linear fusion",
    "ProbOR": "Non-linear fusion",
    "L2-norm": "Non-linear fusion",
    "LogSumExp-T3": "Non-linear fusion",
    "LogSumExp-T5": "Non-linear fusion",
    "LogSumExp-T10": "Non-linear fusion",
    "RankAverage": "Non-linear fusion",
    "RankMax": "Non-linear fusion",
    "ProdComplement": "Non-linear fusion",
    "MaxThen065": "Other (copula, hybrid)",
    "DisagreeSwitch": "Other (copula, hybrid)",
    # Cycle 6
    "PowerMean-p0.5": "Power mean family",
    "PowerMean-p2": "Power mean family",
    "PowerMean-p-1": "Power mean family",
    "PowerMean-p3": "Power mean family",
    "GeomMean-0.65": "Power mean family",
    "Ensemble-5": "Ensemble of alphas",
    "Ensemble-narrow": "Ensemble of alphas",
    "Ensemble-7": "Ensemble of alphas",
    "Trimmed-95": "Trimmed fusion",
    "Trimmed-90": "Trimmed fusion",
    "RankW-0.65": "Other (copula, hybrid)",
    "SmoothAdapt-s0.1": "Other (copula, hybrid)",
    "SmoothAdapt-s0.2": "Other (copula, hybrid)",
    "SmoothAdapt-s0.3": "Other (copula, hybrid)",
    "Clayton-Copula": "Other (copula, hybrid)",
}

paradigm_results = {}
total = 0
for fname in [
    "unsupervised_alpha_discovery.json",
    "alpha_discovery_cycle2.json",
    "alpha_discovery_cycle3.json",
    "alpha_discovery_cycle4.json",
    "alpha_discovery_cycle5.json",
    "alpha_discovery_cycle6.json",
]:
    path = os.path.join(results_dir, fname)
    with open(path) as f:
        data = json.load(f)
    d = data.get("summary", data)
    for mname, mdata in d.items():
        if mname.startswith("Fixed") or mname in ("best_method", "best_non_fixed"):
            continue
        auroc = mdata.get("mean_auroc", 0)
        pname = paradigm_assignment.get(mname, "UNKNOWN")
        if pname == "UNKNOWN":
            print(f"UNKNOWN: {mname}")
        if pname not in paradigm_results:
            paradigm_results[pname] = []
        paradigm_results[pname].append((mname, auroc))
        total += 1

order = [
    "Distributional signals",
    "Pseudo-labeling",
    "Cross-model agreement",
    "Anomaly rate prior",
    "Direct metric optimization",
    "Conditional score analysis",
    "Non-linear fusion",
    "Sample-level adaptive",
    "Score calibration",
    "Power mean family",
    "Ensemble of alphas",
    "Trimmed fusion",
    "Other (copula, hybrid)",
]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsupervised_strategies_full.json")) as _f:
    fixed_base = json.load(_f)["fixed_065_auroc"]
print(f"{'Paradigm':<35} {'N':>3} {'Best':>10} {'Delta':>8}")
print("-" * 60)
for pname in order:
    if pname in paradigm_results:
        ml = paradigm_results[pname]
        n = len(ml)
        best = max(m[1] for m in ml)
        delta = best - fixed_base
        print(f"{pname:<35} {n:>3} {best:>10.3f} {delta:>+8.3f}")
print(f"{'TOTAL':<35} {total:>3}")

# CV for Fixed
fixed_vals = [97.85, 92.55, 95.34]
mean_f = np.mean(fixed_vals)
std_f = np.std(fixed_vals, ddof=0)
cv_f = std_f / mean_f * 100
print(f"\nFixed CV: mean={mean_f:.2f}, std={std_f:.2f}, CV={cv_f:.1f}%")
