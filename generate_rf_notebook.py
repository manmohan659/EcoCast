#!/usr/bin/env python3
"""
Generate comprehensive Random Forest analysis notebook for EcoCast

This creates rf_sustainability_analysis.ipynb with all 5 use cases
following the structure and rigor of the reference ai_ethics_rf.ipynb
"""
import json

def create_cells():
    """Generate all notebook cells"""
    
    cells = [
        # Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# 🌲 Random Forest Applications for Sustainability Forecasting\n\n**Author:** EcoCast Project  \n**Date:** 2025-01-17  \n**Reference:** ai_ethics_rf.ipynb structure  \n\n---\n\n## 📋 Analysis Objectives\n\n1. **CO₂ Driver Analysis** → Feature importance for emissions\n2. **Policy Impact Simulator** → Counterfactual renewable scenarios  \n3. **Trajectory Prediction** → ML-based Improving/Declining classification\n4. **Cluster Explainability** → What defines sustainability clusters?\n5. **Anomaly Detection** → Identify unusual country profiles\n\n**Why Random Forest here (vs Prophet)?**\n- Prophet: Time-series forecasting (predict future from past)\n- RF: Cross-sectional regression (predict Y from features at one point in time)\n\n---"]
        },
        
        # Cell 0: Data Import
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# ═══════════════════════════════════════════════════════════\n# CELL 0: Data Import and Setup\n# ═══════════════════════════════════════════════════════════\n\nimport os, sys, json, joblib\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest\nfrom sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, cross_val_predict\nfrom sklearn.metrics import (\n    r2_score, mean_absolute_error, mean_squared_error,\n    accuracy_score, precision_score, recall_score, f1_score,\n    confusion_matrix, classification_report\n)\nfrom sklearn.feature_selection import VarianceThreshold\nfrom sklearn.inspection import partial_dependence\nfrom math import sqrt\n\n# Paths\nPROJECT_ROOT = Path.cwd()\nDATA_WORK = PROJECT_ROOT / 'data_work'\nARTEFACTS = PROJECT_ROOT / 'artefacts'\nMODELS = PROJECT_ROOT / 'models'\nIMAGES = PROJECT_ROOT / 'artefacts' / 'rf_images'\n\nfor d in [ARTEFACTS, MODELS, IMAGES]:\n    d.mkdir(exist_ok=True)\n\nFEATURES_PATH = DATA_WORK / 'features_full.csv'\nassert FEATURES_PATH.exists(), f\"Missing: {FEATURES_PATH}\"\n\npanel = pd.read_csv(FEATURES_PATH)\nprint('✅ Panel:', panel.shape)\nprint('✅ Years:', panel['year'].min(), '→', panel['year'].max())\nprint('✅ Countries:', panel['iso3'].nunique())\n\nimport sklearn\nprint('\\n📦 Versions: numpy', np.__version__, '| pandas', pd.__version__, '| sklearn', sklearn.__version__)\n\nmissing = panel.isna().sum().sum()\nprint(f'🔍 Missing: {missing:,} ({missing/panel.size*100:.2f}%)')"]
        },
        
        # Cell 1: Snapshot Prep
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# ═══════════════════════════════════════════════════════════\n# CELL 1: Prepare Cross-Sectional Snapshot\n# ═══════════════════════════════════════════════════════════\n\nlatest_year = panel.groupby('iso3')['year'].max().reset_index()\nsnap = panel.merge(latest_year, on=['iso3', 'year'])\nprint(f'✅ Snapshot: {len(snap)} countries, year range {snap[\"year\"].min()}-{snap[\"year\"].max()}')\n\n# Load clusters\nclusters = pd.read_csv(ARTEFACTS / 'clusters.csv')\nsnap = snap.merge(clusters[['iso3', 'cluster_lvl2']], on='iso3', how='left')\nprint(f'✅ Clusters merged: {snap[\"cluster_lvl2\"].notna().sum()}/{len(snap)}')\n\n# Define target\nTARGET = 'co2_pc'\nexclude = ['year', 'iso3', TARGET, 'cluster_lvl1', 'cluster_lvl2']\nfeature_cols = [c for c in snap.select_dtypes('number').columns if c not in exclude]\n\nX = snap[feature_cols].fillna(0)\ny = snap[TARGET].fillna(0)\n\n# Variance filter\nselector = VarianceThreshold(threshold=0.01)\nX_filt = selector.fit_transform(X)\nselected = X.columns[selector.get_support()].tolist()\n\nprint(f'\\n📊 Features: {len(feature_cols)} → {len(selected)} (after variance filter)')\nprint(f'   Target ({TARGET}): mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}')"]
        },
        
        # UC1: CO₂ Drivers - Grid Search
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\n\n## 🎯 USE CASE 1: CO₂ Driver Analysis\n\n**Question:** What factors drive CO₂ emission differences?\n\n**Method:** RF regressor on cross-sectional data → feature importance\n\n---"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 1.1: Grid Search with OOB\nn_est_grid = [300, 500, 800]\nmtry_grid = sorted({max(1, min(len(selected), int(round(f*sqrt(len(selected)))))) for f in [0.5, 1.0, 2.0]})\n\nprint(f'Grid: n_estimators={n_est_grid}, max_features={mtry_grid}')\nprint(f'{\"NTREE\":>8} {\"MTRY\":>8} {\"OOB_R²\":>10} {\"OOB_MAE\":>10}')\nprint('-'*40)\n\nbest_r2 = -np.inf\nbest_model_uc1 = None\nbest_params_uc1 = None\ngrid_uc1 = []\n\nfor n in n_est_grid:\n    for m in mtry_grid:\n        rf = RandomForestRegressor(\n            n_estimators=n, max_features=m, max_depth=10,\n            min_samples_leaf=3, oob_score=True, bootstrap=True,\n            random_state=42, n_jobs=-1\n        )\n        rf.fit(X[selected], y)\n        r2, mae = rf.oob_score_, mean_absolute_error(y, rf.oob_prediction_)\n        print(f'{n:8d} {m:8d} {r2:10.5f} {mae:10.5f}')\n        grid_uc1.append({'NTREE':n, 'MTRY':m, 'OOB_R²':round(r2,5), 'OOB_MAE':round(mae,5)})\n        if r2 > best_r2:\n            best_r2, best_model_uc1, best_params_uc1 = r2, rf, {'n_estimators':n, 'max_features':m}\n\nprint(f'\\n✅ Best OOB R²={best_r2:.5f}, params={best_params_uc1}')"]
        },
        
        # UC1: Feature Importance
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 1.2: Feature Importance\nimport_df = pd.DataFrame({\n    'feature': selected,\n    'importance': best_model_uc1.feature_importances_,\n    'importance_pct': best_model_uc1.feature_importances_ * 100\n}).sort_values('importance', ascending=False)\n\nprint('🎯 Top 20 CO₂ Drivers:')\nfor i, r in import_df.head(20).iterrows():\n    print(f'{i+1:3d}. {r[\"feature\"]:.<55} {r[\"importance_pct\"]:>6.3f}%')\n\nimport_df.to_csv(ARTEFACTS/'co2_drivers.csv', index=False)\njoblib.dump(best_model_uc1, MODELS/'rf_co2_drivers.pkl')\n\n# Visualization\nplt.figure(figsize=(10,8))\nplt.barh(range(15), import_df.head(15)['importance_pct'], color='#ef4444')\nplt.yticks(range(15), import_df.head(15)['feature'])\nplt.xlabel('Importance (%)', fontweight='bold')\nplt.title(f'Top 15 CO₂ Drivers (R²={best_r2:.3f})', fontweight='bold')\nplt.gca().invert_yaxis()\nplt.grid(axis='x', alpha=0.3)\nplt.tight_layout()\nplt.savefig(IMAGES/'co2_drivers.png', dpi=300)\nplt.show()\nprint('✅ Saved: artefacts/co2_drivers.csv, models/rf_co2_drivers.pkl, images/co2_drivers.png')"]
        },
        
        # UC2: Policy Simulator
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\n\n## 🎯 USE CASE 2: Policy Impact Simulator\n\n**Question:** If renewables increase by X%, how does CO₂ change?\n\n**Method:** Partial dependence + counterfactual predictions\n\n---"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 2.1: Partial Dependence Plot (Renewables → CO₂)\nif 'renew_pct' in selected:\n    idx = selected.index('renew_pct')\n    pdp = partial_dependence(best_model_uc1, X[selected], features=[idx], grid_resolution=50)\n    \n    pdp_df = pd.DataFrame({'renew_pct': pdp['grid_values'][0], 'co2_expected': pdp['average'][0]})\n    pdp_df.to_csv(ARTEFACTS/'pdp_renewables.csv', index=False)\n    \n    plt.figure(figsize=(10,6))\n    plt.plot(pdp['grid_values'][0], pdp['average'][0], lw=3, color='#10b981', label='Expected CO₂')\n    plt.axhline(2.0, color='#ef4444', ls='--', lw=2, label='Paris 2030 target')\n    plt.xlabel('Renewable % (holding other features constant)')\n    plt.ylabel('Expected CO₂ (tCO₂/capita)')\n    plt.title('Policy Impact: Renewables → CO₂ (Partial Dependence)')\n    plt.legend()\n    plt.grid(alpha=0.3)\n    plt.tight_layout()\n    plt.savefig(IMAGES/'pdp_renewables.png', dpi=300)\n    plt.show()\n    print('✅ Saved: artefacts/pdp_renewables.csv, images/pdp_renewables.png')\nelse:\n    print('⚠️  renew_pct not in features, skip PDP')"]
        },
        
        # UC2: Counterfactual Function
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 2.2: Counterfactual Policy Simulator\ndef simulate_policy(iso3, renew_increase=10):\n    row = snap[snap['iso3']==iso3]\n    if row.empty or 'renew_pct' not in selected:\n        return None\n    \n    X_base = row[selected].fillna(0)\n    co2_base = best_model_uc1.predict(X_base)[0]\n    renew_curr = row['renew_pct'].iloc[0]\n    \n    X_cf = X_base.copy()\n    X_cf.loc[:, 'renew_pct'] = min(100, renew_curr + renew_increase)\n    co2_cf = best_model_uc1.predict(X_cf)[0]\n    \n    delta = co2_cf - co2_base\n    pct = (delta/co2_base*100) if co2_base>0 else 0\n    \n    return {\n        'iso3':iso3, 'curr_renew':renew_curr, 'curr_co2':co2_base,\n        'new_renew':renew_curr+renew_increase, 'new_co2':co2_cf,\n        'delta_co2':delta, 'pct_reduction':abs(pct)\n    }\n\n# Test\nfor iso in ['USA','DEU','IND','CHN']:\n    res = simulate_policy(iso, 20)\n    if res:\n        print(f\"{iso}: {res['curr_renew']:.1f}% → {res['new_renew']:.1f}% renew | CO₂: {res['curr_co2']:.2f} → {res['new_co2']:.2f} ({res['pct_reduction']:.1f}% reduction)\")\n\n# Batch generate\nsims = []\nfor iso in snap['iso3']:\n    for inc in [10,20,30,50]:\n        r = simulate_policy(iso, inc)\n        if r:\n            sims.append(r)\npd.DataFrame(sims).to_csv(ARTEFACTS/'policy_simulations.csv', index=False)\nprint(f'✅ Saved {len(sims)} scenarios: artefacts/policy_simulations.csv')"]
        },
        
        # UC3: Trajectory Prediction
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\n\n## 🎯 USE CASE 3: Trajectory Prediction\n\n**Question:** Will emissions improve/decline/stabilize?\n\n**Method:** RF classifier trained on (features at year T) → (actual trajectory T to T+5)\n\n---"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 3.1: Create Training Labels\ntraj_train = []\nfor iso in panel['iso3'].unique():\n    ts = panel[panel['iso3']==iso].sort_values('year')\n    if len(ts)<20:\n        continue\n    for split in range(15, len(ts)-5):\n        hist, fut = ts.iloc[:split], ts.iloc[split:split+5]\n        if len(fut)<5:\n            continue\n        snap_row = hist.iloc[-1]\n        co2_fut = fut['co2_pc'].values\n        valid = ~np.isnan(co2_fut)\n        if sum(valid)<3:\n            continue\n        slope = np.polyfit(range(sum(valid)), co2_fut[valid], 1)[0]\n        label = 0 if slope<-0.1 else (1 if slope>0.1 else 2)\n        feats = snap_row[selected].fillna(0).to_dict()\n        feats.update({'iso3':iso, 'label':label, 'slope':slope})\n        traj_train.append(feats)\n\ntraj_df = pd.DataFrame(traj_train)\nX_traj = traj_df[selected].fillna(0)\ny_traj = traj_df['label']\nprint(f'✅ Trajectory training: {len(traj_df)} samples, {y_traj.nunique()} classes')\nprint('   Distribution:', y_traj.value_counts().to_dict())"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 3.2: Train Classifier with OOB\nn_grid = [200,300,500]\nm_grid = [int(sqrt(len(selected)))]\nprint(f'Grid: n={n_grid}, m={m_grid}')\n\nbest_acc = -1\nbest_traj = None\nbest_par_traj = None\n\nfor n in n_grid:\n    for m in m_grid:\n        clf = RandomForestClassifier(\n            n_estimators=n, max_features=m, max_depth=8,\n            min_samples_leaf=10, oob_score=True,\n            class_weight='balanced', random_state=42, n_jobs=-1\n        )\n        clf.fit(X_traj, y_traj)\n        acc = clf.oob_score_\n        print(f'n={n}, m={m}, OOB_Acc={acc:.5f}')\n        if acc>best_acc:\n            best_acc, best_traj, best_par_traj = acc, clf, {'n':n,'m':m}\n\nprint(f'\\n✅ Best OOB Acc={best_acc:.5f}, params={best_par_traj}')\n\n# CV validation\nskf = StratifiedKFold(3, shuffle=True, random_state=42)\ncv_scores = cross_val_score(best_traj, X_traj, y_traj, cv=skf, scoring='accuracy')\nprint(f'3-Fold CV: {[round(s,5) for s in cv_scores]}, mean={cv_scores.mean():.5f}')"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 3.3: Predict Current Countries\nX_curr = snap[selected].fillna(0)\ny_pred = best_traj.predict(X_curr)\ny_prob = best_traj.predict_proba(X_curr)\n\nlabels = ['📉 Improving', '📈 Declining', '➡️ Stable']\ntraj_res = pd.DataFrame({\n    'iso3': snap['iso3'],\n    'co2_pc': snap['co2_pc'],\n    'trajectory': [labels[i] for i in y_pred],\n    'confidence': y_prob.max(axis=1),\n    'prob_improving': y_prob[:,0],\n    'prob_declining': y_prob[:,1],\n    'prob_stable': y_prob[:,2]\n}).sort_values('confidence', ascending=False)\n\nprint('Top 10 predictions (highest confidence):')\nprint(traj_res.head(10)[['iso3','co2_pc','trajectory','confidence']].to_string(index=False))\nprint(f'\\nDistribution: {traj_res[\"trajectory\"].value_counts().to_dict()}')\n\ntraj_res.to_csv(ARTEFACTS/'trajectory_predictions.csv', index=False)\njoblib.dump(best_traj, MODELS/'rf_trajectory_classifier.pkl')\nprint('\\n✅ Saved: artefacts/trajectory_predictions.csv, models/rf_trajectory_classifier.pkl')"]
        },
        
        # UC4: Cluster Explainability
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\n\n## 🎯 USE CASE 4: Cluster Explainability\n\n**Question:** What characteristics define each cluster?\n\n**Method:** RF classifier (features → cluster) → importance shows cluster drivers\n\n---"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 4.1: Train Cluster Predictor\nsnap_c = snap[snap['cluster_lvl2'].notna()].copy()\nX_c = snap_c[selected].fillna(0)\ny_c = snap_c['cluster_lvl2'].astype(int)\n\nprint(f'Cluster training: {len(X_c)} countries, clusters={sorted(y_c.unique())}')\n\nn_grid_c = [200, 300]\nm_grid_c = [int(sqrt(len(selected)))]\n\nbest_oob_c = -1\nbest_c = None\nbest_par_c = None\n\nfor n in n_grid_c:\n    for m in m_grid_c:\n        clf = RandomForestClassifier(\n            n_estimators=n, max_features=m, max_depth=6,\n            min_samples_leaf=5, oob_score=True,\n            class_weight='balanced', random_state=42, n_jobs=-1\n        )\n        clf.fit(X_c, y_c)\n        acc = clf.oob_score_\n        print(f'n={n}, m={m}, OOB_Acc={acc:.5f}')\n        if acc>best_oob_c:\n            best_oob_c, best_c, best_par_c = acc, clf, {'n':n,'m':m}\n\nprint(f'\\n✅ Best OOB Acc={best_oob_c:.5f}, params={best_par_c}')\n\nc_import = pd.DataFrame({\n    'feature': selected,\n    'importance': best_c.feature_importances_,\n    'importance_pct': best_c.feature_importances_*100\n}).sort_values('importance', ascending=False)\n\nprint('\\nTop 15 Cluster Drivers:')\nprint(c_import.head(15)[['feature','importance_pct']].to_string(index=False))\n\nc_import.to_csv(ARTEFACTS/'cluster_drivers.csv', index=False)\njoblib.dump(best_c, MODELS/'rf_cluster_explainer.pkl')\nprint('\\n✅ Saved: artefacts/cluster_drivers.csv, models/rf_cluster_explainer.pkl')"]
        },
        
        # UC5: Anomaly Detection
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["---\n\n## 🎯 USE CASE 5: Anomaly Detection\n\n**Question:** Which countries have unusual sustainability profiles?\n\n**Method:** Isolation Forest (expects ~5% outliers)\n\n---"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# USE CASE 5: Isolation Forest\niso_f = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)\nlabels = iso_f.fit_predict(X[selected])\nscores = iso_f.score_samples(X[selected])\n\nanom = pd.DataFrame({\n    'iso3': snap['iso3'].values,\n    'is_outlier': labels==-1,\n    'anomaly_score': scores,\n    'co2_pc': snap['co2_pc'].values,\n    'gdp_pc': snap.get('Percapita GDP (2010 USD)_BiocapPerCap', pd.Series([np.nan]*len(snap))).values\n}).sort_values('anomaly_score')\n\noutliers = anom[anom['is_outlier']]\nprint(f'🚨 Outliers: {len(outliers)}/{len(snap)} ({len(outliers)/len(snap)*100:.1f}%)')\nprint('\\nTop 10 outliers:')\nfor i, r in outliers.head(10).iterrows():\n    print(f'{r[\"iso3\"]:>4} | CO₂={r[\"co2_pc\"]:>7.2f}t | GDP=${r[\"gdp_pc\"]:>10,.0f} | Score={r[\"anomaly_score\"]:>8.4f}')\n\nanom.to_csv(ARTEFACTS/'sustainability_outliers.csv', index=False)\njoblib.dump(iso_f, MODELS/'isolation_forest.pkl')\n\nplt.figure(figsize=(12,8))\nnorm = anom[~anom['is_outlier']]\nplt.scatter(norm['gdp_pc'], norm['co2_pc'], c='#3b82f6', s=40, alpha=0.5, label=f'Normal ({len(norm)})')\nplt.scatter(outliers['gdp_pc'], outliers['co2_pc'], c='#ef4444', s=150, alpha=0.9, marker='*', edgecolors='#000', linewidths=2, label=f'Outliers ({len(outliers)})')\nfor _, r in outliers.head(8).iterrows():\n    plt.annotate(r['iso3'], (r['gdp_pc'], r['co2_pc']), xytext=(5,5), textcoords='offset points', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', fc='yellow', alpha=0.7))\nplt.xlabel('GDP per capita (USD)')\nplt.ylabel('CO₂ per capita (t)')\nplt.title('Sustainability Outliers (Isolation Forest, 5% contamination)')\nplt.legend()\nplt.grid(alpha=0.3)\nplt.tight_layout()\nplt.savefig(IMAGES/'outliers.png', dpi=300)\nplt.show()\nprint('\\n✅ Saved: artefacts/sustainability_outliers.csv, models/isolation_forest.pkl, images/outliers.png')"]
        },
        
        # Final: Export Metadata
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["# FINAL: Export Consolidated Metadata for API\nmeta = {\n    'generated': pd.Timestamp.now().isoformat(),\n    'n_countries': len(snap),\n    'n_features': len(selected),\n    'uc1_co2_drivers': {\n        'model': 'models/rf_co2_drivers.pkl',\n        'params': best_params_uc1,\n        'r2': round(best_r2, 5),\n        'top3': import_df['feature'].head(3).tolist()\n    },\n    'uc2_policy_sim': {\n        'model': 'models/rf_co2_drivers.pkl',  # Reuse\n        'outputs': ['artefacts/pdp_renewables.csv', 'artefacts/policy_simulations.csv']\n    },\n    'uc3_trajectory': {\n        'model': 'models/rf_trajectory_classifier.pkl',\n        'params': best_par_traj,\n        'oob_acc': round(best_acc, 5)\n    },\n    'uc4_clusters': {\n        'model': 'models/rf_cluster_explainer.pkl',\n        'params': best_par_c,\n        'oob_acc': round(best_oob_c, 5)\n    },\n    'uc5_outliers': {\n        'model': 'models/isolation_forest.pkl',\n        'n_outliers': int(len(outliers))\n    }\n}\n\nwith open(ARTEFACTS/'rf_manifest.json', 'w') as f:\n    json.dump(meta, f, indent=2)\n\nprint('\\n'+'='*60)\nprint('🎉 ALL 5 USE CASES COMPLETE')\nprint('='*60)\nprint('\\nGenerated files:')\nfor f in ARTEFACTS.glob('*.csv'):\n    if 'rf' in f.name or f.name in ['co2_drivers.csv','policy_simulations.csv','trajectory_predictions.csv','cluster_drivers.csv','sustainability_outliers.csv']:\n        print(f'  ✅ {f.relative_to(PROJECT_ROOT)}')\nfor f in MODELS.glob('rf_*.pkl'):\n    print(f'  ✅ {f.relative_to(PROJECT_ROOT)}')\nfor f in IMAGES.glob('*.png'):\n    print(f'  ✅ {f.relative_to(PROJECT_ROOT)}')\n\nprint('\\n📡 Next: Add these API endpoints to backend/app/main.py:')\nprint('  • GET /co2-drivers')\nprint('  • POST /simulate-policy/{iso3}')\nprint('  • GET /trajectories')\nprint('  • GET /cluster-drivers')\nprint('  • GET /outliers')"]
        }
    ]
    
    return cells

def main():
    """Generate the complete notebook JSON"""
    from pathlib import Path
    
    notebook = {
        "cells": create_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    output_path = Path('rf_sustainability_analysis.ipynb')
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Created {output_path}")
    print(f"   Cells: {len(notebook['cells'])}")
    print(f"\n📖 To run:")
    print(f"   jupyter notebook {output_path}")
    print(f"\n🎯 Outputs will be saved to:")
    print(f"   • artefacts/co2_drivers.csv")
    print(f"   • artefacts/policy_simulations.csv")
    print(f"   • artefacts/trajectory_predictions.csv")
    print(f"   • artefacts/cluster_drivers.csv")
    print(f"   • artefacts/sustainability_outliers.csv")
    print(f"   • models/rf_*.pkl (5 models)")
    print(f"   • artefacts/rf_images/*.png (publication-ready charts)")

if __name__ == '__main__':
    main()

