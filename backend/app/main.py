# backend/app/main.py  ← keep ONLY this block here
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd, os, json, pathlib
import numpy as np

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Set absolute paths for artefacts and forecasts directories
ARTEFACTS = os.getenv("ARTEFACTS_DIR", os.path.join(PROJECT_ROOT, "artefacts"))
FORECASTS = os.getenv("FORECASTS_DIR", os.path.join(PROJECT_ROOT, "forecasts"))
INSIGHTS = os.path.join(PROJECT_ROOT, "insights")
DATA_WORK = os.path.join(PROJECT_ROOT, "data_work")

print(f"Using artefacts directory: {ARTEFACTS}")
print(f"Using forecasts directory: {FORECASTS}")
print(f"Using insights directory: {INSIGHTS}")
print(f"Using data_work directory: {DATA_WORK}")

# Ensure directories exist
pathlib.Path(ARTEFACTS).mkdir(exist_ok=True)
pathlib.Path(FORECASTS).mkdir(exist_ok=True)

def _csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found")
    return pd.read_csv(path)

# Helper function to clean dataframes for JSON serialization
def clean_for_json(df):
    # Replace NaN and Inf values with None
    return df.replace([np.nan, np.inf, -np.inf], None)

app = FastAPI(title="Sustainability API", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-Total-Count"]
)

# Mount insights directory for static file access
app.mount(
    "/insights",
    StaticFiles(directory=INSIGHTS),
    name="insights"
)

@app.get("/manifest")                # → targets & cluster list
def manifest():
    try:
        # Load the manifest file
        with open(f"{ARTEFACTS}/manifest.json") as f:
            data = json.load(f)
        
        # Process the data to replace NaN values with None
        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            else:
                return obj
        
        # Clean the data
        cleaned_data = clean_nans(data)
        return cleaned_data
    except Exception as e:
        print(f"Error loading manifest: {str(e)}")
        # Return a simple manifest if the file doesn't exist or has errors
        return {"targets": ["co2_pc", "forest_land_EFProdPerCap"], "clusters": []}

@app.get("/clusters")                # → all country ↔ cluster labels
def clusters():
    try:
        df = _csv(f"{ARTEFACTS}/clusters.csv")
        # Clean the dataframe for JSON serialization
        return clean_for_json(df).to_dict("records")
    except Exception as e:
        print(f"Error loading clusters: {str(e)}")
        return []

@app.get("/cluster-median/{target}") # → median forecast series
def median(target:str):
    try:
        df = _csv(f"{ARTEFACTS}/cluster_medians_{target}.csv")
        return clean_for_json(df).to_dict("records")
    except Exception as e:
        print(f"Error loading cluster medians for {target}: {str(e)}")
        return []

@app.get("/forecast/{iso}/{target}") # → per‑country forecast cone
def forecast(iso:str, target:str):
    try:
        df = _csv(f"{FORECASTS}/{iso}/{target}.csv")
        return clean_for_json(df).to_dict("records")
    except Exception as e:
        print(f"Error loading forecast for {iso}/{target}: {str(e)}")
        return []

@app.get("/model-scores/{iso}")      # → MAE table for that country
def scores(iso:str):
    try:
        df = _csv(f"{ARTEFACTS}/model_scores.csv")
        result_df = df[df.iso3==iso]
        return clean_for_json(result_df).to_dict("records")
    except Exception as e:
        print(f"Error loading model scores for {iso}: {str(e)}")
        return []

@app.get("/timeseries/{iso}")        # → timeseries for protected area reality chart
def get_ts(iso: str):
    try:
        # Load features_full.csv which contains all timeseries data
        features_path = os.path.join(DATA_WORK, "features_full.csv")
        features_df = _csv(features_path)
        
        # Filter for the requested ISO code
        country_df = features_df[features_df.iso3 == iso]
        
        # Select only the relevant columns: year, protected_pct, forest_land_BiocapPerCap
        if 'protected_pct' in country_df.columns and 'forest_land_BiocapPerCap' in country_df.columns:
            result_df = country_df[['year', 'protected_pct', 'forest_land_BiocapPerCap']]
            # Make sure values are sorted by year
            result_df = result_df.sort_values('year')
            # Clean and return the data
            return clean_for_json(result_df).to_dict("records")
        else:
            print(f"Missing required columns for {iso}. Available columns: {country_df.columns.tolist()}")
            return []
    except Exception as e:
        print(f"Error fetching timeseries data for {iso}: {str(e)}")
        return []

# ═══════════════════════════════════════════════════════════
# Random Forest Enhanced Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/co2-drivers")
def get_co2_drivers():
    """
    Get top factors driving CO₂ emissions globally
    
    Returns: Feature importance from Random Forest trained on
             cross-sectional data (latest year, all 193 countries)
             
    Use: Display as bar chart showing "What drives emissions?"
    """
    try:
        drivers = _csv(f"{ARTEFACTS}/co2_drivers.csv")
        # Return top 20 with clean names
        top20 = drivers.head(20).copy()
        
        # Simplify feature names for frontend display
        top20['display_name'] = top20['feature'].str.replace('_', ' ').str.title()
        
        return clean_for_json(top20).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "CO₂ drivers not generated yet. Run: python rf_co2_drivers.py")
    except Exception as e:
        print(f"Error loading CO₂ drivers: {str(e)}")
        return []

@app.get("/co2-drivers/metadata")
def get_drivers_metadata():
    """Get Random Forest model metadata (R², MAE, n_features, etc.)"""
    try:
        with open(f"{ARTEFACTS}/rf_manifest.json") as f:
            manifest = json.load(f)
            return {
                "model": "RandomForestRegressor",
                "r2_score": manifest["uc1_co2_drivers"]["r2"],
                "mae": 1.00,  # From notebook
                "n_countries": manifest["n_countries"],
                "n_features": manifest["n_features"],
                "top_3_drivers": manifest["uc1_co2_drivers"]["top3"]
            }
    except:
        return {
            "model": "RandomForestRegressor",
            "status": "not_trained",
            "message": "Run rf_sustainability_analysis.ipynb to generate"
        }

@app.get("/trajectories")
def get_trajectories():
    """
    Get ML predictions for future emission trajectories
    
    Returns: Classification (Improving/Declining/Stable) with confidence scores
    """
    try:
        trajectories = _csv(f"{ARTEFACTS}/trajectory_predictions.csv")
        return clean_for_json(trajectories).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "Trajectory predictions not generated yet")
    except Exception as e:
        print(f"Error loading trajectories: {str(e)}")
        return []

@app.get("/outliers")
def get_outliers():
    """
    Get countries with unusual sustainability profiles
    
    Returns: Anomaly detection results from Isolation Forest
    """
    try:
        outliers = _csv(f"{ARTEFACTS}/sustainability_outliers.csv")
        return clean_for_json(outliers).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "Outlier analysis not generated yet")
    except Exception as e:
        print(f"Error loading outliers: {str(e)}")
        return []

@app.get("/cluster-drivers")
def get_cluster_drivers():
    """
    Get features that define sustainability clusters
    
    Returns: Feature importance for cluster classification
    """
    try:
        drivers = _csv(f"{ARTEFACTS}/cluster_drivers.csv")
        return clean_for_json(drivers).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "Cluster drivers not generated yet")
    except Exception as e:
        print(f"Error loading cluster drivers: {str(e)}")
        return []

@app.get("/pdp-renewables")
def get_pdp_renewables():
    """
    Get Partial Dependence Plot data for renewables → CO₂
    
    Returns: Expected CO₂ at different renewable energy levels
    """
    try:
        pdp = _csv(f"{ARTEFACTS}/pdp_renewables.csv")
        return clean_for_json(pdp).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "PDP data not generated yet")
    except Exception as e:
        print(f"Error loading PDP data: {str(e)}")
        return []

@app.get("/policy-simulations")
def get_policy_simulations():
    """
    Get all pre-computed policy simulation scenarios
    
    Returns: 772 scenarios across all countries
    """
    try:
        sims = _csv(f"{ARTEFACTS}/policy_simulations.csv")
        return clean_for_json(sims).to_dict("records")
    except FileNotFoundError:
        raise HTTPException(404, "Policy simulations not generated yet")
    except Exception as e:
        print(f"Error loading policy simulations: {str(e)}")
        return []

@app.get("/simulate-policy/{iso3}")
def simulate_policy(iso3: str, renew_increase: float = 20):
    """
    Simulate impact of increasing renewable energy for a specific country
    
    Args:
        iso3: Country code
        renew_increase: Percentage increase in renewable energy (default: 20)
    
    Returns: Current vs predicted CO₂ levels
    """
    try:
        # Load pre-computed simulations
        sims = _csv(f"{ARTEFACTS}/policy_simulations.csv")
        
        # Find matching scenario (closest to requested increase)
        country_sims = sims[sims['iso3'] == iso3.upper()]
        if country_sims.empty:
            raise HTTPException(404, f"No simulation data for {iso3}")
        
        # Get closest scenario to requested increase
        # Assuming scenarios are at 10, 20, 30, 50
        available_increases = [10, 20, 30, 50]
        closest = min(available_increases, key=lambda x: abs(x - renew_increase))
        
        # Calculate the actual increase for each row
        country_sims = country_sims.copy()
        country_sims['actual_increase'] = country_sims['new_renew'] - country_sims['curr_renew']
        
        # Find closest match
        result = country_sims.iloc[(country_sims['actual_increase'] - closest).abs().argsort()[:1]]
        
        if result.empty:
            result = country_sims.iloc[[0]]  # Fallback to first scenario
        
        return clean_for_json(result.iloc[0]).to_dict()
    except FileNotFoundError:
        raise HTTPException(404, "Policy simulation data not generated yet")
    except Exception as e:
        print(f"Error simulating policy for {iso3}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.get("/rf-manifest")
def get_rf_manifest():
    """
    Get Random Forest analysis metadata
    
    Returns: Information about all 5 RF use cases
    """
    try:
        with open(f"{ARTEFACTS}/rf_manifest.json") as f:
            return json.load(f)
    except:
        return {"error": "RF analysis not complete. Run rf_sustainability_analysis.ipynb"}