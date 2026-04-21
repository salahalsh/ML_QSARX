"""
Download and clean bioactivity data from ChEMBL for multi-target 2D-QSAR benchmarking.
Targets: EGFR (kinase), DRD2 (GPCR), BACE-1 (protease), hERG (ion channel)
Uses IC50 for kinases/proteases/ion channels, Ki for GPCRs (dominant assay type).
Both are converted to pActivity = 9 - log10(value_nM) for unified regression.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    'EGFR': {
        'chembl_id': 'CHEMBL203',
        'family': 'Kinase',
        'full_name': 'Epidermal Growth Factor Receptor',
        'activity_type': 'IC50',
    },
    'DRD2': {
        'chembl_id': 'CHEMBL217',
        'family': 'GPCR',
        'full_name': 'Dopamine D2 Receptor',
        'activity_type': 'Ki',  # GPCRs predominantly use Ki from radioligand binding assays
    },
    'BACE1': {
        'chembl_id': 'CHEMBL4822',
        'family': 'Protease',
        'full_name': 'Beta-Secretase 1',
        'activity_type': 'IC50',
    },
    'hERG': {
        'chembl_id': 'CHEMBL240',
        'family': 'Ion Channel',
        'full_name': 'hERG Potassium Channel',
        'activity_type': 'IC50',
    },
}


def download_target_data(chembl_target_id, target_name, activity_type='IC50'):
    """Download bioactivity data for a given target from ChEMBL REST API."""
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        'target_chembl_id': chembl_target_id,
        'standard_type': activity_type,
        'standard_units': 'nM',
        'standard_relation': '=',
        'assay_type': 'B',
        'limit': 1000,
        'offset': 0,
    }
    print(f"  [{target_name}] Activity type: {activity_type}")

    all_activities = []
    page = 1

    while True:
        print(f"  [{target_name}] Fetching page {page} (offset={params['offset']})...")
        try:
            resp = requests.get(base_url, params=params, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  [{target_name}] Request error on page {page}: {e}")
            if page > 1:
                print(f"  [{target_name}] Continuing with {len(all_activities)} records collected so far")
                break
            raise

        data = resp.json()
        activities = data.get('activities', [])
        if not activities:
            break

        all_activities.extend(activities)

        next_url = data.get('page_meta', {}).get('next')
        if not next_url:
            break

        params['offset'] += params['limit']
        page += 1
        time.sleep(0.5)

    print(f"  [{target_name}] Downloaded {len(all_activities)} activity records")
    return all_activities


def clean_data(activities, target_name, activity_type='IC50'):
    """Clean and process raw ChEMBL activity data."""
    from rdkit import Chem

    val_col = f'{activity_type}_nM'
    records = []
    for act in activities:
        smiles = act.get('canonical_smiles')
        value = act.get('standard_value')
        relation = act.get('standard_relation')
        chembl_id = act.get('molecule_chembl_id')
        assay_id = act.get('assay_chembl_id')

        if smiles and value and relation == '=':
            try:
                val_nm = float(value)
                if val_nm > 0:
                    records.append({
                        'molecule_chembl_id': chembl_id,
                        'canonical_smiles': smiles,
                        val_col: val_nm,
                        'assay_chembl_id': assay_id,
                    })
            except (ValueError, TypeError):
                continue

    df = pd.DataFrame(records)
    n_valid = len(df)
    print(f"  [{target_name}] Valid records: {n_valid}")

    if len(df) == 0:
        return df

    # Deduplicate: median value per unique SMILES
    df_dedup = df.groupby('canonical_smiles').agg({
        'molecule_chembl_id': 'first',
        val_col: 'median',
        'assay_chembl_id': 'first',
    }).reset_index()
    n_dedup = len(df_dedup)
    print(f"  [{target_name}] After dedup (median): {n_dedup}")

    # Convert to pActivity (pIC50 or pKi) — same transform: 9 - log10(nM)
    df_dedup['pIC50'] = 9 - np.log10(df_dedup[val_col])

    # Filter range [3, 12]
    df_clean = df_dedup[(df_dedup['pIC50'] >= 3.0) & (df_dedup['pIC50'] <= 12.0)].copy()
    n_range = len(df_clean)
    print(f"  [{target_name}] After pIC50 filter [3,12]: {n_range}")

    # RDKit validation
    valid_mask = []
    clean_smiles = []
    for smi in df_clean['canonical_smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mask.append(True)
            clean_smiles.append(Chem.MolToSmiles(mol))
        else:
            valid_mask.append(False)
            clean_smiles.append(None)

    df_clean = df_clean[valid_mask].copy()
    df_clean['canonical_smiles'] = [s for s, v in zip(clean_smiles, valid_mask) if v]
    print(f"  [{target_name}] After RDKit validation: {len(df_clean)}")

    # Final dedup on canonical SMILES
    df_clean = df_clean.drop_duplicates(subset='canonical_smiles', keep='first')

    # Remove salts/mixtures
    df_clean = df_clean[~df_clean['canonical_smiles'].str.contains(r'\.')].copy()
    print(f"  [{target_name}] Final clean: {len(df_clean)}")

    df_clean = df_clean.sort_values('pIC50', ascending=False).reset_index(drop=True)
    return df_clean


def summarize_dataset(df, target_name):
    """Print and return dataset summary."""
    n = len(df)
    stats = {
        'target': target_name,
        'n_compounds': n,
        'pIC50_min': round(df['pIC50'].min(), 2),
        'pIC50_max': round(df['pIC50'].max(), 2),
        'pIC50_mean': round(df['pIC50'].mean(), 2),
        'pIC50_std': round(df['pIC50'].std(), 2),
        'pIC50_median': round(df['pIC50'].median(), 2),
        'high_potency_pct': round(100 * (df['pIC50'] >= 7).sum() / n, 1),
        'medium_potency_pct': round(100 * ((df['pIC50'] >= 5) & (df['pIC50'] < 7)).sum() / n, 1),
        'low_potency_pct': round(100 * (df['pIC50'] < 5).sum() / n, 1),
    }
    print(f"\n  === {target_name} Summary ===")
    print(f"  Compounds: {n}")
    print(f"  pIC50: [{stats['pIC50_min']}, {stats['pIC50_max']}], mean={stats['pIC50_mean']} +/- {stats['pIC50_std']}")
    return stats


if __name__ == '__main__':
    all_summaries = []

    for target_name, target_info in TARGETS.items():
        csv_path = os.path.join(OUTPUT_DIR, f'{target_name.lower()}_clean.csv')

        # Skip EGFR if already downloaded
        if target_name == 'EGFR' and os.path.exists(os.path.join(OUTPUT_DIR, 'egfr_clean.csv')):
            print(f"\n{'='*50}")
            print(f"  {target_name} ({target_info['full_name']}) — ALREADY EXISTS, loading...")
            df = pd.read_csv(os.path.join(OUTPUT_DIR, 'egfr_clean.csv'))
            stats = summarize_dataset(df, target_name)
            stats['family'] = target_info['family']
            stats['chembl_id'] = target_info['chembl_id']
            all_summaries.append(stats)
            continue

        print(f"\n{'='*50}")
        print(f"  Downloading {target_name} ({target_info['full_name']}, {target_info['chembl_id']})")
        print(f"{'='*50}")

        act_type = target_info.get('activity_type', 'IC50')
        activities = download_target_data(target_info['chembl_id'], target_name, act_type)

        # Save raw
        raw_path = os.path.join(OUTPUT_DIR, f'{target_name.lower()}_raw.json')
        with open(raw_path, 'w') as f:
            json.dump(activities, f)
        print(f"  Saved raw data: {raw_path}")

        # Clean
        df_clean = clean_data(activities, target_name, act_type)

        if len(df_clean) < 500:
            print(f"  WARNING: {target_name} has only {len(df_clean)} compounds — may be too small")

        # Save
        df_clean.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        stats = summarize_dataset(df_clean, target_name)
        stats['family'] = target_info['family']
        stats['chembl_id'] = target_info['chembl_id']
        all_summaries.append(stats)

    # Save summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(os.path.dirname(OUTPUT_DIR), 'supplementary', 'Table_S7_multi_target_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved multi-target summary: {summary_path}")

    print("\n" + "=" * 50)
    print("ALL TARGETS COMPLETE")
    print("=" * 50)
    print(summary_df[['target', 'n_compounds', 'pIC50_mean', 'pIC50_std', 'family']].to_string(index=False))
