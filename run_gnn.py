"""
Experiment 8: Graph Neural Network Baseline
=============================================
Compares a 3-layer GCN against the best 2D method (RF + ECFP4)
on all 4 targets with both random and scaffold splits.

Uses PyTorch Geometric for graph construction and GNN training.
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

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

warnings.filterwarnings('ignore')

from run_case_study import (
    calc_ecfp, scaffold_split, evaluate_model,
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

# GNN hyperparameters
HIDDEN_DIM = 128
NUM_LAYERS = 3
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 20  # early stopping patience

# Atom features — compact encoding (28 dims) for drug-like molecules
COMMON_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I


def atom_features(atom):
    """Compute compact atom feature vector (28 dimensions)."""
    # Atomic number one-hot (9 common + 1 other = 10)
    anum = atom.GetAtomicNum()
    feat = [1 if anum == a else 0 for a in COMMON_ATOMS]
    feat.append(1 if anum not in COMMON_ATOMS else 0)
    # Degree one-hot (6)
    deg = atom.GetDegree()
    feat += [1 if deg == d else 0 for d in range(6)]
    # Formal charge one-hot (5)
    fc = atom.GetFormalCharge()
    feat += [1 if fc == c else 0 for c in [-2, -1, 0, 1, 2]]
    # Hybridization one-hot (4)
    hyb = atom.GetHybridization()
    for h in [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D]:
        feat.append(1 if hyb == h else 0)
    # Boolean features (3)
    feat.append(1 if atom.GetIsAromatic() else 0)
    feat.append(atom.GetTotalNumHs())
    feat.append(atom.GetNumRadicalElectrons())
    return feat  # 10 + 6 + 5 + 4 + 3 = 28 dims


def mol_to_graph(smiles, y_val):
    """Convert a SMILES to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    x = torch.tensor(node_feats, dtype=torch.float)

    # Edge index (undirected)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    y = torch.tensor([y_val], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


class GCNRegressor(torch.nn.Module):
    """3-layer GCN with global mean pooling for regression."""

    def __init__(self, in_channels, hidden_channels=HIDDEN_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x.squeeze(-1)


def train_gcn(train_loader, val_loader, in_channels, epochs=EPOCHS, patience=PATIENCE):
    """Train GCN with early stopping on validation loss."""
    device = torch.device('cpu')
    model = GCNRegressor(in_channels).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        n_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs
        train_loss = total_loss / n_samples

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += F.mse_loss(pred, batch.y).item() * batch.num_graphs
                n_val += batch.num_graphs
        val_loss /= n_val

        if epoch % 20 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_gcn(model, loader):
    """Evaluate GCN model on a data loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            y_true.extend(batch.y.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'R2_test': round(r2_score(y_true, y_pred), 4),
        'RMSE_test': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'MAE_test': round(mean_absolute_error(y_true, y_pred), 4),
        'y_true': y_true,
        'y_pred': y_pred,
    }


def run_gnn_for_target(target_name, csv_file):
    """Run GCN experiment for one target (random + scaffold splits)."""
    print(f"\n{'='*60}")
    print(f"  GNN: {target_name}")
    print(f"{'='*60}")

    df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
    smiles = df['canonical_smiles'].tolist()
    y = df['pIC50'].values
    n = len(df)

    # Convert all molecules to graphs
    print("  Converting molecules to graphs...", end=' ', flush=True)
    graphs = []
    valid_idx = []
    for i, (smi, yv) in enumerate(zip(smiles, y)):
        g = mol_to_graph(smi, yv)
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)
    print(f"{len(graphs)}/{n} valid")

    in_channels = graphs[0].x.shape[1]
    valid_idx = np.array(valid_idx)
    y_valid = y[valid_idx]
    smiles_valid = [smiles[i] for i in valid_idx]

    results = {'n_compounds': len(graphs), 'n_atom_features': in_channels}

    # ── Random split ──────────────────────────────
    print("\n  Random split:")
    train_idx, test_idx = train_test_split(
        np.arange(len(graphs)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Further split train into train/val (90/10)
    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=RANDOM_STATE
    )

    train_graphs = [graphs[i] for i in train_sub_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    print(f"    Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Train GCN
    t0 = time.time()
    torch.manual_seed(RANDOM_STATE)
    model = train_gcn(train_loader, val_loader, in_channels)
    gcn_time = time.time() - t0

    # Evaluate
    gcn_res = evaluate_gcn(model, test_loader)
    gcn_res['time_s'] = round(gcn_time, 1)
    print(f"    GCN: R²={gcn_res['R2_test']:.4f}, RMSE={gcn_res['RMSE_test']:.4f} ({gcn_time:.0f}s)")

    # RF baseline (same split)
    print("    RF baseline...", end=' ', flush=True)
    X_fp = calc_ecfp(smiles_valid, radius=2, n_bits=2048)
    X_train_rf = X_fp.iloc[train_idx].values
    X_test_rf = X_fp.iloc[test_idx].values
    y_train_rf = y_valid[train_idx]
    y_test_rf = y_valid[test_idx]

    rf = RandomForestRegressor(n_estimators=500, min_samples_split=5,
                                random_state=RANDOM_STATE, n_jobs=12)
    t0 = time.time()
    rf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf.predict(X_test_rf)
    rf_time = time.time() - t0

    rf_r2 = r2_score(y_test_rf, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    print(f"R²={rf_r2:.4f}, RMSE={rf_rmse:.4f} ({rf_time:.0f}s)")

    results['random'] = {
        'GCN': {k: v for k, v in gcn_res.items() if k not in ('y_true', 'y_pred')},
        'RF': {'R2_test': round(rf_r2, 4), 'RMSE_test': round(rf_rmse, 4), 'time_s': round(rf_time, 1)},
        'delta_R2': round(gcn_res['R2_test'] - rf_r2, 4),
    }

    # ── Scaffold split ────────────────────────────
    print("\n  Scaffold split:")
    scaf_train_idx, scaf_test_idx = scaffold_split(smiles_valid, y_valid, test_size=TEST_SIZE)

    # Map to graph indices
    scaf_train_sub, scaf_val = train_test_split(
        scaf_train_idx, test_size=0.1, random_state=RANDOM_STATE
    )

    scaf_train_graphs = [graphs[i] for i in scaf_train_sub]
    scaf_val_graphs = [graphs[i] for i in scaf_val]
    scaf_test_graphs = [graphs[i] for i in scaf_test_idx]

    scaf_train_loader = DataLoader(scaf_train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    scaf_val_loader = DataLoader(scaf_val_graphs, batch_size=BATCH_SIZE)
    scaf_test_loader = DataLoader(scaf_test_graphs, batch_size=BATCH_SIZE)

    print(f"    Train: {len(scaf_train_graphs)}, Val: {len(scaf_val_graphs)}, Test: {len(scaf_test_graphs)}")

    # Train GCN (scaffold)
    t0 = time.time()
    torch.manual_seed(RANDOM_STATE)
    model_scaf = train_gcn(scaf_train_loader, scaf_val_loader, in_channels)
    gcn_scaf_time = time.time() - t0

    gcn_scaf_res = evaluate_gcn(model_scaf, scaf_test_loader)
    gcn_scaf_res['time_s'] = round(gcn_scaf_time, 1)
    print(f"    GCN: R²={gcn_scaf_res['R2_test']:.4f}, RMSE={gcn_scaf_res['RMSE_test']:.4f}")

    # RF scaffold baseline
    print("    RF scaffold...", end=' ', flush=True)
    X_train_scaf = X_fp.iloc[scaf_train_idx].values
    X_test_scaf = X_fp.iloc[scaf_test_idx].values
    y_train_scaf = y_valid[scaf_train_idx]
    y_test_scaf = y_valid[scaf_test_idx]

    rf_scaf = RandomForestRegressor(n_estimators=500, min_samples_split=5,
                                     random_state=RANDOM_STATE, n_jobs=12)
    rf_scaf.fit(X_train_scaf, y_train_scaf)
    y_pred_scaf = rf_scaf.predict(X_test_scaf)
    rf_scaf_r2 = r2_score(y_test_scaf, y_pred_scaf)
    rf_scaf_rmse = np.sqrt(mean_squared_error(y_test_scaf, y_pred_scaf))
    print(f"R²={rf_scaf_r2:.4f}")

    results['scaffold'] = {
        'GCN': {k: v for k, v in gcn_scaf_res.items() if k not in ('y_true', 'y_pred')},
        'RF': {'R2_test': round(rf_scaf_r2, 4), 'RMSE_test': round(rf_scaf_rmse, 4)},
        'delta_R2': round(gcn_scaf_res['R2_test'] - rf_scaf_r2, 4),
    }

    return results


def generate_gnn_figures(all_gnn_results):
    """Generate Figure 11: GCN vs RF comparison."""
    targets = list(all_gnn_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, split_name in zip(axes, ['random', 'scaffold']):
        x = np.arange(len(targets))
        width = 0.35

        gcn_vals = [all_gnn_results[t][split_name]['GCN']['R2_test'] for t in targets]
        rf_vals = [all_gnn_results[t][split_name]['RF']['R2_test'] for t in targets]

        bars1 = ax.bar(x - width/2, rf_vals, width, label='RF + ECFP4',
                       color='#4CAF50', edgecolor='gray')
        bars2 = ax.bar(x + width/2, gcn_vals, width, label='GCN (3-layer)',
                       color='#9C27B0', edgecolor='gray')

        # Annotate deltas
        for i in range(len(targets)):
            delta = all_gnn_results[targets[i]][split_name]['delta_R2']
            y_pos = max(gcn_vals[i], rf_vals[i]) + 0.01
            ax.text(i, y_pos, f'{delta:+.3f}', ha='center', va='bottom',
                    fontsize=9, color='red' if delta < 0 else 'green', fontweight='bold')

        ax.set_xlabel('Target')
        ax.set_ylabel('R² (test)')
        ax.set_title(f'{split_name.capitalize()} Split')
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 0.85)

    fig.suptitle('GCN vs. RF + ECFP4 Across Targets', fontsize=13)
    fig.tight_layout()

    png_path = os.path.join(FIGURES_DIR, 'Figure_11_gnn_comparison.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default=None,
                        help='Run single target (EGFR, DRD2, BACE1, hERG)')
    args = parser.parse_args()

    if args.target:
        if args.target not in TARGETS:
            print(f"Unknown target: {args.target}. Choose from {list(TARGETS.keys())}")
            sys.exit(1)
        run_targets = OrderedDict([(args.target, TARGETS[args.target])])
    else:
        run_targets = TARGETS

    print("=" * 60)
    print("  EXPERIMENT 8: Graph Neural Network Baseline")
    print(f"  Architecture: {NUM_LAYERS}-layer GCN, {HIDDEN_DIM}-dim hidden")
    print(f"  Training: {EPOCHS} epochs max, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"  Targets: {list(run_targets.keys())}")
    print("=" * 60)

    # Load existing results if running single target (incremental mode)
    results_path = os.path.join(RESULTS_DIR, 'gnn_results.json')
    if args.target and os.path.exists(results_path):
        with open(results_path) as f:
            all_gnn_results = OrderedDict(json.load(f))
    else:
        all_gnn_results = OrderedDict()

    total_t0 = time.time()

    for target_name, csv_file in run_targets.items():
        results = run_gnn_for_target(target_name, csv_file)
        all_gnn_results[target_name] = results

    # Save results
    with open(results_path, 'w') as f:
        json.dump(all_gnn_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    # Generate figures
    generate_gnn_figures(all_gnn_results)

    # Summary
    print("\n" + "=" * 60)
    print("  GNN SUMMARY: GCN vs RF+ECFP4")
    print("=" * 60)
    print(f"\n  {'Target':<8} {'Split':<10} {'RF R²':>8} {'GCN R²':>8} {'Delta':>8}")
    print("  " + "-" * 44)
    for target, res in all_gnn_results.items():
        for split in ['random', 'scaffold']:
            rf = res[split]['RF']['R2_test']
            gcn = res[split]['GCN']['R2_test']
            delta = res[split]['delta_R2']
            print(f"  {target:<8} {split:<10} {rf:>8.4f} {gcn:>8.4f} {delta:>+8.4f}")

    total_time = time.time() - total_t0
    print(f"\n  Total GNN time: {total_time / 60:.1f} minutes")


if __name__ == '__main__':
    main()
