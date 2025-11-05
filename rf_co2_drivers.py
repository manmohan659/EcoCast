#!/usr/bin/env python3
"""
Random Forest CO₂ Driver Analysis

Identifies which factors (GDP, renewables, protected areas, etc.)
have the strongest relationship with CO₂ emissions across countries.

This is a CROSS-SECTIONAL analysis (snapshot of all countries),
NOT time-series forecasting (where RF fails).

Usage:
    python rf_co2_drivers.py

Outputs:
    artefacts/co2_drivers.csv
    artefacts/co2_drivers_chart.png
    models/co2_drivers_rf.pkl
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Paths
DATA_WORK = 'data_work'
ARTEFACTS = 'artefacts'
MODELS = 'models'

os.makedirs(ARTEFACTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

def main():
    print("🌲 Random Forest CO₂ Driver Analysis")
    print("=" * 50)
    
    # Load panel data
    panel = pd.read_csv(f'{DATA_WORK}/features_full.csv')
    print(f"✅ Loaded panel: {panel.shape}")
    
    # Get latest year per country (cross-sectional snapshot)
    latest_year = panel.groupby('iso3')['year'].max().reset_index()
    latest = panel.merge(latest_year, on=['iso3', 'year'])
    print(f"✅ Latest snapshot: {len(latest)} countries")
    
    # Define target
    target = 'co2_pc'
    
    # Feature selection: Exclude identifiers, target, and cluster labels
    exclude_cols = ['year', 'iso3', target]
    if 'cluster_lvl1' in latest.columns:
        exclude_cols.append('cluster_lvl1')
    if 'cluster_lvl2' in latest.columns:
        exclude_cols.append('cluster_lvl2')
    
    feature_cols = [c for c in latest.select_dtypes('number').columns 
                    if c not in exclude_cols]
    
    print(f"📊 Initial features: {len(feature_cols)}")
    
    # Prepare data
    X = latest[feature_cols].fillna(0)
    y = latest[target].fillna(0)
    
    # Remove low-variance features (constants or near-constants)
    selector = VarianceThreshold(threshold=0.01)
    X_filtered = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"📊 After variance filter: {len(selected_features)} features")
    
    # Train Random Forest
    print("\n🔨 Training Random Forest...")
    rf_drivers = RandomForestRegressor(
        n_estimators=500,        # Many trees for stable importance
        max_depth=10,            # Prevent overfitting
        min_samples_leaf=5,      # Require at least 5 countries per leaf
        max_features='sqrt',     # Use sqrt(n) features per split
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        verbose=0
    )
    
    rf_drivers.fit(X[selected_features], y)
    print("✅ Model trained")
    
    # Calculate performance
    y_pred = rf_drivers.predict(X[selected_features])
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n📈 Model Performance (Cross-sectional CO₂ prediction):")
    print(f"   R² = {r2:.3f}")
    print(f"   MAE = {mae:.3f} tCO₂/capita")
    print(f"   (Prophet time-series MAE = 0.012 for comparison)")
    
    # Extract feature importance
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_drivers.feature_importances_,
        'importance_pct': rf_drivers.feature_importances_ * 100
    }).sort_values('importance', ascending=False)
    
    # Top 20
    top20 = importance_df.head(20)
    
    print(f"\n🎯 Top 20 CO₂ Emission Drivers:")
    print(top20[['feature', 'importance_pct']].to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top20)), top20['importance_pct'], color='#ef4444')
    plt.yticks(range(len(top20)), top20['feature'])
    plt.xlabel('Importance (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Top 20 Factors Driving CO₂ Emissions\n(Random Forest Feature Importance)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    chart_path = f'{ARTEFACTS}/co2_drivers_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved chart: {chart_path}")
    
    # Save CSV for API
    csv_path = f'{ARTEFACTS}/co2_drivers.csv'
    importance_df.to_csv(csv_path, index=False)
    print(f"✅ Saved data: {csv_path}")
    
    # Save model for API use
    model_path = f'{MODELS}/co2_drivers_rf.pkl'
    joblib.dump(rf_drivers, model_path)
    print(f"✅ Saved model: {model_path}")
    
    # Save metadata
    metadata = {
        'model': 'RandomForestRegressor',
        'n_estimators': 500,
        'n_features': len(selected_features),
        'n_countries': len(latest),
        'r2_score': float(r2),
        'mae': float(mae),
        'top_3_drivers': top20['feature'].head(3).tolist()
    }
    
    import json
    with open(f'{ARTEFACTS}/co2_drivers_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✨ Done! Use this in:")
    print(f"   1. Backend: Add GET /co2-drivers endpoint")
    print(f"   2. Frontend: Visualize top drivers as bar chart")
    print(f"   3. Insights: 'What drives emissions?' section")

if __name__ == '__main__':
    main()

