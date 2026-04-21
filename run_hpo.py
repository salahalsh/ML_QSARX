"""
Experiment 7: Hyperparameter Optimization
==========================================
Uses Optuna (Bayesian optimization, TPE sampler) to tune the top-4 algorithms
(RF, XGBoost, LightGBM, SVR) and compare default vs optimized performance.

Key question: Does tuning change the algorithm ranking?

Publication settings:
  - 30 Optuna trials per algorithm (TPE sampler, seed=42)
  - 5-fold CV as optimization objective
  - n_jobs=12 (75% of 16 cores to avoid oversubscription)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from collections import OrderedDict

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from run_case_study import (
    calc_ecfp, evaluate_model,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# ── Publication parameters ──
N_TRIALS = 30
N_JOBS = 12  # 75% of 16 cores

TARGETS = OrderedDict([
    ('EGFR', 'egfr_clean.csv'),
    ('DRD2', 'drd2_clean.csv'),
    ('BACE1', 'bace1_clean.csv'),
    ('hERG', 'herg_clean.csv'),
])


def get_default_algorithms():
    """Return default (untuned) algorithms matching Table 1."""
    return OrderedDict([
        ('RF', RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=N_JOBS)),
        ('XGB', XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)),
        ('LGBM', LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)),
        ('SVR', Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10, gamma='scale'))
        ])),
    ])


def create_objective(algo_name, X_train, y_train, cv):
    """Create an Optuna objective function for each algorithm.

    Uses 5-fold CV R² as the optimization metric — the gold standard
    for HPO in QSAR because it prevents overfitting to a single split.
    """

    def objective(trial):
        if algo_name == 'RF':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
                max_depth=trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50]),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
                random_state=RANDOM_STATE, n_jobs=N_JOBS
            )
        elif algo_name == 'XGB':
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0
            )
        elif algo_name == 'LGBM':
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000, step=100),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int('num_leaves', 15, 255),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1
            )
        elif algo_name == 'SVR':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(
                    kernel='rbf',
                    C=trial.suggest_float('C', 0.1, 100, log=True),
                    gamma=trial.suggest_float('gamma', 1e-4, 1.0, log=True),
                    epsilon=trial.suggest_float('epsilon', 0.01, 0.5, log=True),
                ))
            ])

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        return scores.mean()

    return objective


def run_hpo_for_target(target_name, csv_file):
    """Run HPO for all 4 algorithms on one target."""
    print(f"\n{'='*60}")
    print(f"  HPO: {target_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    smiles = df['canonical_smiles'].tolist()
    y = df['pIC50'].values

    # Calculate ECFP4
    print("  Calculating ECFP4...", end=' ', flush=True)
    X_all = calc_ecfp(smiles, radius=2, n_bits=2048)
    print("done")

    # Split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = X_all.iloc[train_idx].reset_index(drop=True).values
    X_test = X_all.iloc[test_idx].reset_index(drop=True).values
    y_train, y_test = y[train_idx], y[test_idx]

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Default performance
    defaults = get_default_algorithms()
    results = {}

    for algo_name, default_model in defaults.items():
        print(f"\n  [{algo_name}] Default evaluation...", end=' ', flush=True)
        m = clone(default_model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        default_r2 = r2_score(y_test, y_pred)
        default_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Default 5-fold CV
        default_cv = cross_val_score(clone(default_model), X_train, y_train,
                                      cv=cv, scoring='r2', n_jobs=1).mean()
        print(f"R²={default_r2:.4f}, CV={default_cv:.4f}")

        # Optuna optimization
        print(f"  [{algo_name}] Optuna ({N_TRIALS} trials)...", end=' ', flush=True)
        t0 = time.time()
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        objective = create_objective(algo_name, X_train, y_train, cv)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        opt_time = time.time() - t0

        best_params = study.best_params
        best_cv = study.best_value
        print(f"Best CV={best_cv:.4f} ({opt_time:.0f}s)")

        # Evaluate optimized model on test set
        if algo_name == 'RF':
            opt_model = RandomForestRegressor(
                **best_params, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        elif algo_name == 'XGB':
            opt_model = XGBRegressor(
                **best_params, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
        elif algo_name == 'LGBM':
            opt_model = LGBMRegressor(
                **best_params, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
        elif algo_name == 'SVR':
            opt_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', **best_params))
            ])

        opt_model_fitted = clone(opt_model)
        opt_model_fitted.fit(X_train, y_train)
        y_pred_opt = opt_model_fitted.predict(X_test)
        opt_r2 = r2_score(y_test, y_pred_opt)
        opt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_opt))

        delta_r2 = opt_r2 - default_r2
        print(f"  [{algo_name}] Optimized: R²={opt_r2:.4f} (Delta={delta_r2:+.4f})")

        results[algo_name] = {
            'default': {
                'R2_test': round(default_r2, 4),
                'RMSE_test': round(default_rmse, 4),
                'R2_CV': round(default_cv, 4),
            },
            'optimized': {
                'R2_test': round(opt_r2, 4),
                'RMSE_test': round(opt_rmse, 4),
                'R2_CV': round(best_cv, 4),
                'best_params': {k: (str(v) if v is None else v) for k, v in best_params.items()},
                'n_trials': N_TRIALS,
                'optimization_time_s': round(opt_time, 1),
            },
            'delta_R2': round(delta_r2, 4),
        }

    return results


def generate_hpo_figures(all_hpo_results):
    """Generate Figure 10: Default vs Optimized performance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets = list(all_hpo_results.keys())
    algos = ['RF', 'XGB', 'LGBM', 'SVR']
    colors_default = '#90CAF9'
    colors_optimized = '#1565C0'

    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        results = all_hpo_results[target]
        x = np.arange(len(algos))
        width = 0.35

        default_vals = [results[a]['default']['R2_test'] for a in algos]
        opt_vals = [results[a]['optimized']['R2_test'] for a in algos]

        bars1 = ax.bar(x - width/2, default_vals, width, label='Default',
                       color=colors_default, edgecolor='gray')
        bars2 = ax.bar(x + width/2, opt_vals, width, label='Optimized',
                       color=colors_optimized, edgecolor='gray')

        # Annotate deltas
        for i, algo in enumerate(algos):
            delta = results[algo]['delta_R2']
            y_pos = max(default_vals[i], opt_vals[i]) + 0.01
            ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                    fontsize=8, color='red' if delta < 0 else 'green', fontweight='bold')

        ax.set_xlabel('Algorithm')
        ax.set_title(target)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=9)
        ax.set_ylim(0.3, 0.85)
        if ax == axes[0]:
            ax.set_ylabel('R² (test)')
            ax.legend(fontsize=8)

    fig.suptitle(f'Default vs. Optuna-Optimized Performance ({N_TRIALS} trials, {CV_FOLDS}-fold CV)',
                 fontsize=13)
    fig.tight_layout()

    png_path = os.path.join(FIGURES_DIR, 'Figure_10_hpo_comparison.png')
    svg_path = os.path.join(FIGURES_DIR, 'Figure_10_hpo_comparison.svg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default=None,
                        help='Run single target (EGFR, DRD2, BACE1, hERG)')
    args = parser.parse_args()

    # Determine which targets to run
    if args.target:
        if args.target not in TARGETS:
            print(f"Unknown target: {args.target}. Choose from {list(TARGETS.keys())}")
            sys.exit(1)
        run_targets = OrderedDict([(args.target, TARGETS[args.target])])
    else:
        run_targets = TARGETS

    print("=" * 60)
    print("  EXPERIMENT 7: Hyperparameter Optimization (Optuna)")
    print(f"  Trials: {N_TRIALS}, CV: {CV_FOLDS}-fold, Sampler: TPE")
    print(f"  Algorithms: RF, XGB, LGBM, SVR")
    print(f"  Targets: {list(run_targets.keys())}")
    print(f"  CPU: {N_JOBS} cores (75%)")
    print("=" * 60)

    # Load existing results if running single target (incremental mode)
    results_path = os.path.join(RESULTS_DIR, 'hpo_results.json')
    if args.target and os.path.exists(results_path):
        with open(results_path) as f:
            all_hpo_results = OrderedDict(json.load(f))
    else:
        all_hpo_results = OrderedDict()

    total_t0 = time.time()

    for target_name, csv_file in run_targets.items():
        results = run_hpo_for_target(target_name, csv_file)
        all_hpo_results[target_name] = results

        # Save after each target (incremental safety)
        with open(results_path, 'w') as f:
            json.dump(all_hpo_results, f, indent=2, default=str)
        print(f"  [checkpoint] Saved {target_name} results")

    # Generate figures (only if all 4 targets are available)
    if len(all_hpo_results) == 4:
        generate_hpo_figures(all_hpo_results)

    # Summary
    print("\n" + "=" * 60)
    print("  HPO SUMMARY: Default vs Optimized R² (test)")
    print("=" * 60)
    for target, results in all_hpo_results.items():
        print(f"\n  {target}:")
        for algo in ['RF', 'XGB', 'LGBM', 'SVR']:
            if algo in results:
                r = results[algo]
                print(f"    {algo}: {r['default']['R2_test']:.4f} → {r['optimized']['R2_test']:.4f} "
                      f"({r['delta_R2']:+.4f})")

    # Key finding: does ranking change?
    if len(all_hpo_results) >= 2:
        print("\n  Ranking stability:")
        for target, results in all_hpo_results.items():
            default_rank = sorted(results.keys(), key=lambda a: -results[a]['default']['R2_test'])
            opt_rank = sorted(results.keys(), key=lambda a: -results[a]['optimized']['R2_test'])
            changed = default_rank != opt_rank
            print(f"    {target}: Default={'>'.join(default_rank)}, "
                  f"Optimized={'>'.join(opt_rank)} {'*** CHANGED ***' if changed else '(stable)'}")

    total_time = time.time() - total_t0
    print(f"\n  Total HPO time: {total_time / 60:.1f} minutes")


if __name__ == '__main__':
    main()
