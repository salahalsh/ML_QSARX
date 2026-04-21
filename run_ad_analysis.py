"""
Experiment 9: Applicability Domain Analysis
=============================================
Implements distance-based AD using Tanimoto similarity to k-nearest
training neighbors. Compares prediction quality inside vs outside AD.

Generates Williams plot (leverage vs standardized residuals) and
AD-stratified performance comparison.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

from run_case_study import (
    calc_ecfp, calc_rdkit_2d, clean_features, get_algorithms,
    RANDOM_STATE, TEST_SIZE
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

TARGETS = OrderedDict([
    ('EGFR', 'egfr_clean.csv'),
    ('DRD2', 'drd2_clean.csv'),
    ('BACE1', 'bace1_clean.csv'),
    ('hERG', 'herg_clean.csv'),
])

K_NEIGHBORS = 5
AD_Z_THRESHOLD = 1.5  # compounds with mean distance > mean + Z*std are outside AD


def tanimoto_distance_matrix(X1, X2):
    """Compute Tanimoto distance between binary fingerprint matrices."""
    # Tanimoto similarity = |A&B| / |A|B| = |A&B| / (|A| + |B| - |A&B|)
    # For binary vectors: intersection = dot product, union = sum_a + sum_b - dot
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)

    dot = X1 @ X2.T
    sum1 = X1.sum(axis=1).reshape(-1, 1)
    sum2 = X2.sum(axis=1).reshape(1, -1)
    union = sum1 + sum2 - dot

    # Avoid division by zero
    union = np.maximum(union, 1e-10)
    similarity = dot / union
    distance = 1.0 - similarity
    return distance


def compute_ad_tanimoto(X_train_fp, X_test_fp, k=K_NEIGHBORS, z=AD_Z_THRESHOLD):
    """Compute AD boundaries based on Tanimoto distance to k-nearest training neighbors."""
    # Training set: compute intra-training distances for AD boundary
    print("    Computing training set distances...", end=' ', flush=True)
    dist_train = tanimoto_distance_matrix(X_train_fp, X_train_fp)
    np.fill_diagonal(dist_train, np.inf)  # exclude self

    # k-nearest distances for each training compound
    train_knn_dists = np.sort(dist_train, axis=1)[:, :k]
    train_mean_dists = train_knn_dists.mean(axis=1)

    ad_mean = train_mean_dists.mean()
    ad_std = train_mean_dists.std()
    ad_threshold = ad_mean + z * ad_std
    print(f"threshold={ad_threshold:.4f} (mean={ad_mean:.4f}, std={ad_std:.4f})")

    # Test set: compute distance to k-nearest training neighbors
    print("    Computing test set distances...", end=' ', flush=True)
    dist_test = tanimoto_distance_matrix(X_test_fp, X_train_fp)
    test_knn_dists = np.sort(dist_test, axis=1)[:, :k]
    test_mean_dists = test_knn_dists.mean(axis=1)

    inside_ad = test_mean_dists <= ad_threshold
    n_inside = inside_ad.sum()
    n_outside = (~inside_ad).sum()
    pct_inside = 100 * n_inside / len(inside_ad)
    print(f"inside={n_inside} ({pct_inside:.1f}%), outside={n_outside}")

    return {
        'ad_threshold': ad_threshold,
        'ad_mean': ad_mean,
        'ad_std': ad_std,
        'train_mean_dists': train_mean_dists,
        'test_mean_dists': test_mean_dists,
        'inside_ad': inside_ad,
        'n_inside': int(n_inside),
        'n_outside': int(n_outside),
        'pct_inside': round(pct_inside, 1),
    }


def compute_leverage(X_train, X_test):
    """Compute leverage (hat matrix diagonal) for Williams plot."""
    # h_i = x_i^T (X^T X)^-1 x_i
    X = np.asarray(X_train, dtype=np.float64)
    X_t = np.asarray(X_test, dtype=np.float64)

    # Use pseudoinverse for stability (fingerprints are high-dimensional)
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        h_train = np.diag(X @ XtX_inv @ X.T)
        h_test = np.diag(X_t @ XtX_inv @ X_t.T)
    except np.linalg.LinAlgError:
        h_train = np.zeros(X.shape[0])
        h_test = np.zeros(X_t.shape[0])

    # Warning threshold: 3p/n (p = features, n = samples)
    h_star = 3 * X.shape[1] / X.shape[0]
    return h_train, h_test, h_star


def run_ad_for_target(target_name, csv_file):
    """Run AD analysis for one target."""
    print(f"\n{'='*60}")
    print(f"  AD Analysis: {target_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    smiles = df['canonical_smiles'].tolist()
    y = df['pIC50'].values

    # ECFP4
    print("  Calculating ECFP4...", end=' ', flush=True)
    X_fp = calc_ecfp(smiles, radius=2, n_bits=2048)
    print("done")

    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = X_fp.iloc[train_idx].reset_index(drop=True).values
    X_test = X_fp.iloc[test_idx].reset_index(drop=True).values
    y_train, y_test = y[train_idx], y[test_idx]

    # Train RF model
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=500, min_samples_split=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    r2_all = r2_score(y_test, y_pred)
    print(f"  RF R² (all test): {r2_all:.4f}")

    # AD computation
    ad_info = compute_ad_tanimoto(X_train, X_test)

    # Performance inside vs outside AD
    inside = ad_info['inside_ad']
    results = {
        'n_train': len(y_train),
        'n_test': len(y_test),
        'R2_all': round(r2_all, 4),
        'ad_threshold': round(ad_info['ad_threshold'], 4),
        'pct_inside': ad_info['pct_inside'],
        'n_inside': ad_info['n_inside'],
        'n_outside': ad_info['n_outside'],
    }

    if inside.sum() > 10:
        r2_in = r2_score(y_test[inside], y_pred[inside])
        rmse_in = np.sqrt(mean_squared_error(y_test[inside], y_pred[inside]))
        mae_in = mean_absolute_error(y_test[inside], y_pred[inside])
        results['R2_inside'] = round(r2_in, 4)
        results['RMSE_inside'] = round(rmse_in, 4)
        results['MAE_inside'] = round(mae_in, 4)
        print(f"  Inside AD:  R²={r2_in:.4f}, RMSE={rmse_in:.4f}, MAE={mae_in:.4f} (n={inside.sum()})")

    if (~inside).sum() > 10:
        r2_out = r2_score(y_test[~inside], y_pred[~inside])
        rmse_out = np.sqrt(mean_squared_error(y_test[~inside], y_pred[~inside]))
        mae_out = mean_absolute_error(y_test[~inside], y_pred[~inside])
        results['R2_outside'] = round(r2_out, 4)
        results['RMSE_outside'] = round(rmse_out, 4)
        results['MAE_outside'] = round(mae_out, 4)
        print(f"  Outside AD: R²={r2_out:.4f}, RMSE={rmse_out:.4f}, MAE={mae_out:.4f} (n={(~inside).sum()})")

    # Compute residuals and leverage for Williams plot
    residuals = y_test - y_pred
    std_residuals = (residuals - residuals.mean()) / residuals.std()

    results['residuals'] = residuals.tolist()
    results['std_residuals'] = std_residuals.tolist()
    results['test_mean_dists'] = ad_info['test_mean_dists'].tolist()
    results['y_test'] = y_test.tolist()
    results['y_pred'] = y_pred.tolist()
    results['inside_mask'] = inside.tolist()

    return results


def generate_ad_figures(all_ad_results):
    """Generate Figures 12-13: Williams plot and AD comparison."""
    targets = list(all_ad_results.keys())

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
    })

    # Figure 12: Williams-style plot (distance vs standardized residuals)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, target in zip(axes, targets):
        res = all_ad_results[target]
        dists = np.array(res['test_mean_dists'])
        std_res = np.array(res['std_residuals'])
        inside = np.array(res['inside_mask'])
        threshold = res['ad_threshold']

        ax.scatter(dists[inside], std_res[inside], alpha=0.3, s=8,
                   color='#2196F3', label='Inside AD')
        ax.scatter(dists[~inside], std_res[~inside], alpha=0.5, s=12,
                   color='#F44336', label='Outside AD', marker='x')

        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'AD boundary ({threshold:.3f})')
        ax.axhline(y=3, color='gray', linestyle=':', linewidth=0.8)
        ax.axhline(y=-3, color='gray', linestyle=':', linewidth=0.8)

        ax.set_xlabel('Mean Tanimoto Distance to k-NN')
        ax.set_ylabel('Standardized Residual')
        ax.set_title(f'{target} (n_in={res["n_inside"]}, n_out={res["n_outside"]})')
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Applicability Domain: Distance vs. Prediction Error', fontsize=13)
    fig.tight_layout()

    png_path = os.path.join(FIGURES_DIR, 'Figure_12_williams_plot.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # Figure 13: R² inside vs outside AD (bar chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(targets))
    width = 0.25

    r2_all = [all_ad_results[t]['R2_all'] for t in targets]
    r2_in = [all_ad_results[t].get('R2_inside', 0) for t in targets]
    r2_out = [all_ad_results[t].get('R2_outside', 0) for t in targets]

    ax.bar(x - width, r2_all, width, label='All Test', color='#9E9E9E', edgecolor='gray')
    ax.bar(x, r2_in, width, label='Inside AD', color='#4CAF50', edgecolor='gray')
    ax.bar(x + width, r2_out, width, label='Outside AD', color='#F44336', edgecolor='gray')

    # Annotate values
    for i in range(len(targets)):
        ax.text(i - width, r2_all[i] + 0.01, f'{r2_all[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r2_in[i] + 0.01, f'{r2_in[i]:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, r2_out[i] + 0.01, f'{r2_out[i]:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Target')
    ax.set_ylabel('R² (test)')
    ax.set_title('Prediction Quality: Inside vs. Outside Applicability Domain')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()
    ax.set_ylim(0, 0.95)
    fig.tight_layout()

    png_path = os.path.join(FIGURES_DIR, 'Figure_13_ad_comparison.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")


def main():
    print("=" * 60)
    print("  EXPERIMENT 9: Applicability Domain Analysis")
    print(f"  k-NN: {K_NEIGHBORS}, AD threshold: mean + {AD_Z_THRESHOLD}*SD")
    print("=" * 60)

    all_ad_results = OrderedDict()
    total_t0 = time.time()

    for target_name, csv_file in TARGETS.items():
        results = run_ad_for_target(target_name, csv_file)
        # Strip large arrays for JSON (keep summary stats)
        all_ad_results[target_name] = results

    # Save results (strip large arrays)
    save_results = {}
    for target, res in all_ad_results.items():
        save_results[target] = {k: v for k, v in res.items()
                                 if k not in ('residuals', 'std_residuals', 'test_mean_dists',
                                             'y_test', 'y_pred', 'inside_mask')}

    results_path = os.path.join(RESULTS_DIR, 'ad_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved: {results_path}")

    # Generate figures
    generate_ad_figures(all_ad_results)

    # Summary
    print("\n" + "=" * 60)
    print("  AD SUMMARY")
    print("=" * 60)
    for target, res in save_results.items():
        r2_in = res.get('R2_inside', 'N/A')
        r2_out = res.get('R2_outside', 'N/A')
        print(f"  {target}: {res['pct_inside']:.1f}% inside AD, "
              f"R²_in={r2_in}, R²_out={r2_out}, R²_all={res['R2_all']}")

    total_time = time.time() - total_t0
    print(f"\n  Total AD time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
