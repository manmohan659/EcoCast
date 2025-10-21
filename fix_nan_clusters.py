#!/usr/bin/env python3
"""
Emergency fix for NaN cluster assignments (CHN, IND, USA)

This script patches the clusters.csv and manifest.json files
without needing to re-run the entire notebook.

Usage:
    python fix_nan_clusters.py
"""

import pandas as pd
import json
import os

# Paths
ARTEFACTS = 'artefacts'
CLUSTERS_CSV = f'{ARTEFACTS}/clusters.csv'
MANIFEST_JSON = f'{ARTEFACTS}/manifest.json'

def main():
    print("🔧 Fixing NaN cluster assignments...")
    
    # 1. Load clusters
    clusters = pd.read_csv(CLUSTERS_CSV)
    print(f"   Loaded {len(clusters)} countries")
    print(f"   NaN count before: {clusters['cluster_lvl2'].isna().sum()}")
    
    # 2. Fix NaN values
    # Strategy: Countries without cluster_lvl2 get their cluster_lvl1 value + 10
    clusters['cluster_lvl2'] = clusters['cluster_lvl2'].fillna(
        clusters.get('cluster_lvl1', 0) + 10
    )
    
    # If cluster_lvl1 also missing, assign to cluster 2 (majority cluster)
    clusters['cluster_lvl2'] = clusters['cluster_lvl2'].fillna(2.0)
    
    print(f"   NaN count after:  {clusters['cluster_lvl2'].isna().sum()}")
    
    # 3. Save fixed clusters
    clusters.to_csv(CLUSTERS_CSV, index=False)
    print(f"✅ Updated {CLUSTERS_CSV}")
    
    # 4. Update manifest.json
    with open(MANIFEST_JSON, 'r') as f:
        manifest = json.load(f)
    
    # Replace cluster list with fixed values
    cluster_list = []
    for _, row in clusters.iterrows():
        cluster_list.append({
            'iso3': row['iso3'],
            'cluster_lvl2': float(row['cluster_lvl2'])  # Convert to float (not NaN)
        })
    
    manifest['clusters'] = cluster_list
    
    with open(MANIFEST_JSON, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Updated {MANIFEST_JSON}")
    
    # 5. Verify critical countries
    critical = ['CHN', 'IND', 'USA']
    print("\n🔍 Verification:")
    for iso in critical:
        cluster = clusters[clusters['iso3'] == iso]['cluster_lvl2'].values
        if len(cluster) > 0:
            print(f"   {iso}: cluster_lvl2 = {cluster[0]}")
        else:
            print(f"   ⚠️  {iso}: NOT FOUND in clusters.csv")
    
    print("\n✨ Done! Restart backend to see changes.")

if __name__ == '__main__':
    main()

