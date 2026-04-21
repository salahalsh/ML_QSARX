"""
Multi-Target 2D-QSAR Benchmarking
==================================
Runs Experiments 1-5 on all 4 targets (EGFR, DRD2, BACE-1, hERG) and
Experiment 6 (cross-target consistency analysis).

Reuses core functions from run_case_study.py.
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
import matplotlib.gridspec as gridspec
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Import core functions from the original case study
from run_case_study import (
    calc_rdkit_2d, calc_ecfp, calc_maccs,
    scaffold_split, clean_features, get_algorithms, evaluate_model, y_scrambling,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS
)
from sklearn.model_selection import train_test_split

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
SUPP_DIR = os.path.join(BASE_DIR, 'supplementary')

for d in [RESULTS_DIR, FIGURES_DIR, SUPP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Target definitions ─────────────────────────────────
TARGETS = OrderedDict([
    ('EGFR', {'file': 'egfr_clean.csv', 'family': 'Kinase', 'activity': 'IC50'}),
    ('DRD2', {'file': 'drd2_clean.csv', 'family': 'GPCR', 'activity': 'Ki'}),
    ('BACE1', {'file': 'bace1_clean.csv', 'family': 'Protease', 'activity': 'IC50'}),
    ('hERG', {'file': 'herg_clean.csv', 'family': 'Ion Channel', 'activity': 'IC50'}),
])

# Algorithms to test in Exp4 (scaffold split) — top performers only
SCAFFOLD_ALGOS = ['RF', 'XGB', 'LGBM', 'GBR']

# ═══════════════════════════════════════════════════════════
# HELPER: save figure
# ═══════════════════════════════════════════════════════════

def save_fig(fig, name, dpi=300):
    """Save figure as PNG and SVG."""
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    svg_path = os.path.join(FIGURES_DIR, f'{name}.svg')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")


def strip_arrays(results_dict):
    """Remove numpy arrays from results dict for JSON serialization."""
    return {
        k: {kk: vv for kk, vv in v.items()
            if kk not in ('y_test', 'y_pred_test', 'y_pred_train', 'y_train', 'fitted_model')}
        for k, v in results_dict.items()
    }


# ═══════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS FOR ONE TARGET
# ═══════════════════════════════════════════════════════════

def run_target_experiments(target_name, target_info):
    """Run Exp 1-5 for a single target. Returns results dict."""
    csv_path = os.path.join(DATA_DIR, target_info['file'])
    df = pd.read_csv(csv_path)
    smiles = df['canonical_smiles'].tolist()
    y = df['pIC50'].values
    n = len(df)

    print(f"\n{'='*70}")
    print(f"  TARGET: {target_name} ({target_info['family']}, {target_info['activity']})")
    print(f"  Compounds: {n}, pIC50 range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"{'='*70}")

    target_results = {
        'target': target_name,
        'family': target_info['family'],
        'activity_type': target_info['activity'],
        'n_compounds': n,
        'pIC50_range': [round(y.min(), 2), round(y.max(), 2)],
        'pIC50_mean': round(y.mean(), 2),
        'pIC50_std': round(y.std(), 2),
    }

    # ── Calculate descriptors ────────────────────────
    print("\n  Calculating descriptors...")
    t0 = time.time()

    print("    RDKit 2D...", end=' ', flush=True)
    desc_rdkit = calc_rdkit_2d(smiles)
    print(f"{desc_rdkit.shape[1]} descriptors")

    print("    ECFP4...", end=' ', flush=True)
    desc_ecfp4 = calc_ecfp(smiles, radius=2, n_bits=2048)
    print(f"{desc_ecfp4.shape[1]} bits")

    print("    ECFP6...", end=' ', flush=True)
    desc_ecfp6 = calc_ecfp(smiles, radius=3, n_bits=2048)
    print(f"{desc_ecfp6.shape[1]} bits")

    print("    MACCS...", end=' ', flush=True)
    desc_maccs = calc_maccs(smiles)
    print(f"{desc_maccs.shape[1]} bits")

    desc_time = time.time() - t0
    print(f"  Descriptors computed in {desc_time:.1f}s")

    # ── Random split ─────────────────────────────────
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\n  Random split: train={len(train_idx)}, test={len(test_idx)}")

    # ── Scaffold split ───────────────────────────────
    scaffold_train_idx, scaffold_test_idx = scaffold_split(smiles, y, test_size=TEST_SIZE)
    print(f"  Scaffold split: train={len(scaffold_train_idx)}, test={len(scaffold_test_idx)}")

    target_results['n_train_random'] = len(train_idx)
    target_results['n_test_random'] = len(test_idx)
    target_results['n_train_scaffold'] = len(scaffold_train_idx)
    target_results['n_test_scaffold'] = len(scaffold_test_idx)

    # ═════════════════════════════════════════════════
    # EXPERIMENT 1: Algorithm Comparison (ECFP4, random)
    # ═════════════════════════════════════════════════
    print(f"\n  --- Exp 1: Algorithm Comparison ({target_name}) ---")
    X_ecfp4 = desc_ecfp4
    X_train_e1 = X_ecfp4.iloc[train_idx].reset_index(drop=True)
    X_test_e1 = X_ecfp4.iloc[test_idx].reset_index(drop=True)

    algos = get_algorithms()
    exp1 = {}
    for algo_key, (estimator, algo_name) in algos.items():
        print(f"    {algo_name}...", end=' ', flush=True)
        try:
            res = evaluate_model(estimator, X_train_e1, X_test_e1, y_train, y_test)
            exp1[algo_key] = res
            print(f"R²={res['R2_test']:.4f}, RMSE={res['RMSE_test']:.4f}, "
                  f"CV={res['R2_CV']:.4f} ({res['time_s']:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    target_results['exp1_algorithm_comparison'] = strip_arrays(exp1)

    # ═════════════════════════════════════════════════
    # EXPERIMENT 2: Descriptor Comparison (RF, random)
    # ═════════════════════════════════════════════════
    print(f"\n  --- Exp 2: Descriptor Comparison ({target_name}) ---")
    rf_model = algos['RF'][0]
    exp2 = {}

    desc_sets = OrderedDict([
        ('ECFP4', desc_ecfp4),
        ('ECFP6', desc_ecfp6),
        ('RDKit-2D', desc_rdkit),
        ('MACCS', desc_maccs),
    ])

    for desc_name, X_desc in desc_sets.items():
        print(f"    {desc_name}...", end=' ', flush=True)
        X_tr = X_desc.iloc[train_idx].reset_index(drop=True)
        X_te = X_desc.iloc[test_idx].reset_index(drop=True)

        if desc_name == 'RDKit-2D':
            X_tr, X_te = clean_features(X_tr, X_te)

        try:
            res = evaluate_model(rf_model, X_tr, X_te, y_train, y_test)
            res['n_features'] = X_tr.shape[1]
            exp2[desc_name] = res
            print(f"R²={res['R2_test']:.4f}, n_feat={res['n_features']} ({res['time_s']:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Combined: RDKit-2D + ECFP4
    print(f"    RDKit-2D + ECFP4...", end=' ', flush=True)
    X_rdkit_tr = desc_rdkit.iloc[train_idx].reset_index(drop=True)
    X_rdkit_te = desc_rdkit.iloc[test_idx].reset_index(drop=True)
    X_rdkit_tr, X_rdkit_te = clean_features(X_rdkit_tr, X_rdkit_te)
    X_ecfp4_tr = desc_ecfp4.iloc[train_idx].reset_index(drop=True)
    X_ecfp4_te = desc_ecfp4.iloc[test_idx].reset_index(drop=True)
    X_comb_tr = pd.concat([X_rdkit_tr, X_ecfp4_tr], axis=1)
    X_comb_te = pd.concat([X_rdkit_te, X_ecfp4_te], axis=1)
    try:
        res = evaluate_model(rf_model, X_comb_tr, X_comb_te, y_train, y_test)
        res['n_features'] = X_comb_tr.shape[1]
        exp2['RDKit-2D + ECFP4'] = res
        print(f"R²={res['R2_test']:.4f}, n_feat={res['n_features']} ({res['time_s']:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")

    target_results['exp2_descriptor_comparison'] = strip_arrays(exp2)

    # ═════════════════════════════════════════════════
    # EXPERIMENT 3: Feature Selection (RDKit-2D, RF)
    # Only for EGFR (target-specific, sufficient for one target)
    # ═════════════════════════════════════════════════
    if target_name == 'EGFR':
        print(f"\n  --- Exp 3: Feature Selection ({target_name}) ---")
        print("    (Skipping — already computed in original case study)")
        # Load from existing results
        try:
            with open(os.path.join(RESULTS_DIR, 'case_study_results.json')) as f:
                existing = json.load(f)
            target_results['exp3_feature_selection'] = existing.get('exp3_feature_selection', {})
        except FileNotFoundError:
            target_results['exp3_feature_selection'] = {}
    else:
        target_results['exp3_feature_selection'] = None  # Not applicable

    # ═════════════════════════════════════════════════
    # EXPERIMENT 4: Scaffold vs Random Split (ECFP4)
    # ═════════════════════════════════════════════════
    print(f"\n  --- Exp 4: Scaffold vs Random Split ({target_name}) ---")
    exp4 = {}
    y_scaf_train = y[scaffold_train_idx]
    y_scaf_test = y[scaffold_test_idx]
    X_scaf_train = desc_ecfp4.iloc[scaffold_train_idx].reset_index(drop=True)
    X_scaf_test = desc_ecfp4.iloc[scaffold_test_idx].reset_index(drop=True)

    for algo_key in SCAFFOLD_ALGOS:
        if algo_key not in algos:
            continue
        estimator, algo_name = algos[algo_key]
        print(f"    {algo_name} (scaffold)...", end=' ', flush=True)
        try:
            res_scaf = evaluate_model(estimator, X_scaf_train, X_scaf_test,
                                       y_scaf_train, y_scaf_test)
            # Get random split result from exp1
            res_rand = exp1.get(algo_key, {})
            exp4[algo_key] = {
                'random': {k: v for k, v in res_rand.items()
                          if k not in ('y_test', 'y_pred_test', 'y_pred_train', 'y_train', 'fitted_model')},
                'scaffold': {k: v for k, v in res_scaf.items()
                            if k not in ('y_test', 'y_pred_test', 'y_pred_train', 'y_train', 'fitted_model')},
            }
            r2_rand = res_rand.get('R2_test', 0)
            r2_scaf = res_scaf['R2_test']
            print(f"R²_rand={r2_rand:.4f}, R²_scaf={r2_scaf:.4f}, "
                  f"Delta={r2_rand - r2_scaf:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    target_results['exp4_split_comparison'] = exp4

    # ═════════════════════════════════════════════════
    # EXPERIMENT 5: Y-Scrambling (LGBM, ECFP4)
    # ═════════════════════════════════════════════════
    print(f"\n  --- Exp 5: Y-Scrambling ({target_name}) ---")
    lgbm_model = algos.get('LGBM', algos.get('RF'))[0]

    # Original CV R²
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.base import clone
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clone(lgbm_model), X_train_e1.values, y_train,
                                cv=kf, scoring='r2', n_jobs=-1)
    original_cv_r2 = cv_scores.mean()

    print(f"    Original CV R²: {original_cv_r2:.4f}")
    print(f"    Running 30 scrambling iterations...", end=' ', flush=True)
    scrambled_values = y_scrambling(lgbm_model, X_train_e1, pd.Series(y_train), n_iter=30)
    print(f"done. Mean scrambled R²: {scrambled_values.mean():.4f}")

    target_results['exp5_y_scrambling'] = {
        'original_cv_r2': round(original_cv_r2, 4),
        'scrambled_values': [round(v, 4) for v in scrambled_values],
        'scrambled_mean': round(scrambled_values.mean(), 4),
        'scrambled_std': round(scrambled_values.std(), 4),
        'scrambled_max': round(scrambled_values.max(), 4),
        'scrambled_min': round(scrambled_values.min(), 4),
    }

    return target_results


# ═══════════════════════════════════════════════════════════
# EXPERIMENT 6: Cross-Target Consistency Analysis
# ═══════════════════════════════════════════════════════════

def cross_target_analysis(all_target_results):
    """Analyze whether algorithm/descriptor rankings are consistent across targets."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 6: Cross-Target Consistency Analysis")
    print("=" * 70)

    exp6 = {}

    # 6a: Algorithm ranking consistency
    print("\n  --- 6a: Algorithm Rankings ---")
    algo_r2 = {}  # algo -> {target: R2_test}
    for target_name, tres in all_target_results.items():
        exp1 = tres.get('exp1_algorithm_comparison', {})
        for algo, metrics in exp1.items():
            algo_r2.setdefault(algo, {})[target_name] = metrics['R2_test']

    # Build ranking matrix
    targets = list(all_target_results.keys())
    algo_names = list(algo_r2.keys())

    print(f"\n  {'Algorithm':<15}", end='')
    for t in targets:
        print(f"  {t:>8}", end='')
    print(f"  {'Mean':>8}  {'Rank':>4}")
    print("  " + "-" * (15 + 10 * len(targets) + 14))

    algo_means = {}
    for algo in algo_names:
        vals = [algo_r2[algo].get(t, float('nan')) for t in targets]
        mean_val = np.nanmean(vals)
        algo_means[algo] = mean_val
        print(f"  {algo:<15}", end='')
        for v in vals:
            print(f"  {v:>8.4f}", end='')
        print(f"  {mean_val:>8.4f}")

    # Rank by mean
    ranked = sorted(algo_means.items(), key=lambda x: -x[1])
    print(f"\n  Overall ranking: {' > '.join([f'{a}({v:.3f})' for a, v in ranked])}")

    # Kendall's tau between target pairs (rank correlation)
    from scipy.stats import kendalltau, spearmanr
    print("\n  Rank correlation (Spearman) between target pairs:")
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            t1, t2 = targets[i], targets[j]
            r1 = [algo_r2[a].get(t1, 0) for a in algo_names]
            r2 = [algo_r2[a].get(t2, 0) for a in algo_names]
            rho, p = spearmanr(r1, r2)
            print(f"    {t1} vs {t2}: rho={rho:.3f}, p={p:.4f}")

    exp6['algorithm_rankings'] = {
        'r2_matrix': algo_r2,
        'mean_r2': algo_means,
        'overall_ranking': [a for a, _ in ranked],
    }

    # 6b: Descriptor ranking consistency
    print("\n  --- 6b: Descriptor Rankings ---")
    desc_r2 = {}
    for target_name, tres in all_target_results.items():
        exp2 = tres.get('exp2_descriptor_comparison', {})
        for desc, metrics in exp2.items():
            desc_r2.setdefault(desc, {})[target_name] = metrics['R2_test']

    desc_names = list(desc_r2.keys())
    desc_means = {}
    for desc in desc_names:
        vals = [desc_r2[desc].get(t, float('nan')) for t in targets]
        mean_val = np.nanmean(vals)
        desc_means[desc] = mean_val
        print(f"    {desc:<20} mean R²={mean_val:.4f}  ({', '.join(f'{t}={v:.3f}' for t, v in zip(targets, vals))})")

    desc_ranked = sorted(desc_means.items(), key=lambda x: -x[1])
    print(f"\n  Descriptor ranking: {' > '.join([f'{d}({v:.3f})' for d, v in desc_ranked])}")

    exp6['descriptor_rankings'] = {
        'r2_matrix': desc_r2,
        'mean_r2': desc_means,
        'overall_ranking': [d for d, _ in desc_ranked],
    }

    # 6c: Scaffold gap consistency
    print("\n  --- 6c: Scaffold Gap Consistency ---")
    scaffold_gaps = {}
    for target_name, tres in all_target_results.items():
        exp4 = tres.get('exp4_split_comparison', {})
        gaps = {}
        for algo, splits in exp4.items():
            r2_rand = splits.get('random', {}).get('R2_test', 0)
            r2_scaf = splits.get('scaffold', {}).get('R2_test', 0)
            gaps[algo] = round(r2_rand - r2_scaf, 4)
        scaffold_gaps[target_name] = gaps
        mean_gap = np.mean(list(gaps.values())) if gaps else 0
        print(f"    {target_name}: mean gap={mean_gap:.3f} ({', '.join(f'{a}={g:.3f}' for a, g in gaps.items())})")

    exp6['scaffold_gaps'] = scaffold_gaps

    return exp6


# ═══════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════

def generate_cross_target_figures(all_target_results, exp6):
    """Generate Figure 9: cross-target heatmap."""
    print("\n  Generating cross-target figures...")

    # Set style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
    })

    # Figure 9: Algorithm × Target R² heatmap
    algo_r2 = exp6['algorithm_rankings']['r2_matrix']
    targets = list(all_target_results.keys())
    algo_order = exp6['algorithm_rankings']['overall_ranking']

    matrix = []
    for algo in algo_order:
        row = [algo_r2[algo].get(t, 0) for t in targets]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.8)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_yticks(range(len(algo_order)))
    ax.set_yticklabels(algo_order, fontsize=10)
    ax.set_xlabel('Target')
    ax.set_ylabel('Algorithm')
    ax.set_title('Test Set R² Across Targets and Algorithms')

    # Annotate cells
    for i in range(len(algo_order)):
        for j in range(len(targets)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='R² (test)')
    fig.tight_layout()
    save_fig(fig, 'Figure_9_cross_target_heatmap')

    # Figure 9b: Scaffold gap comparison
    scaffold_gaps = exp6['scaffold_gaps']
    algos_in_gaps = list(next(iter(scaffold_gaps.values())).keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(targets))
    width = 0.18
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, algo in enumerate(algos_in_gaps):
        gaps = [scaffold_gaps[t].get(algo, 0) for t in targets]
        ax.bar(x + i * width - width * 1.5, gaps, width, label=algo, color=colors[i % len(colors)])

    ax.set_xlabel('Target')
    ax.set_ylabel('R² Drop (Random − Scaffold)')
    ax.set_title('Generalization Gap: Random vs. Scaffold Split')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend()
    ax.axhline(y=0, color='gray', linewidth=0.5)
    fig.tight_layout()
    save_fig(fig, 'Figure_9b_scaffold_gap_comparison')


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MULTI-TARGET 2D-QSAR BENCHMARKING")
    print("  Targets: EGFR, DRD2, BACE-1, hERG")
    print("=" * 70)

    all_target_results = OrderedDict()
    total_t0 = time.time()

    for target_name, target_info in TARGETS.items():
        t0 = time.time()
        results = run_target_experiments(target_name, target_info)
        elapsed = time.time() - t0
        print(f"\n  [{target_name}] Completed in {elapsed:.0f}s")
        all_target_results[target_name] = results

    # ── Experiment 6: Cross-target analysis ──────────
    exp6 = cross_target_analysis(all_target_results)

    # ── Save results ─────────────────────────────────
    output = {
        'targets': {k: v for k, v in all_target_results.items()},
        'exp6_cross_target': exp6,
    }

    results_path = os.path.join(RESULTS_DIR, 'multi_target_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved results: {results_path}")

    # ── Generate figures ─────────────────────────────
    generate_cross_target_figures(all_target_results, exp6)

    # ── Summary table ────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: Best Algorithm per Target (ECFP4, random split)")
    print("=" * 70)
    for target_name, tres in all_target_results.items():
        exp1 = tres['exp1_algorithm_comparison']
        best = max(exp1.items(), key=lambda x: x[1]['R2_test'])
        print(f"  {target_name:>6}: {best[0]} R²={best[1]['R2_test']:.4f} "
              f"(n={tres['n_compounds']}, {tres['family']})")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total time: {total_elapsed / 60:.1f} minutes")
    print("  DONE.")


if __name__ == '__main__':
    main()
