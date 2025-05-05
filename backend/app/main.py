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

print(f"Using artefacts directory: {ARTEFACTS}")
print(f"Using forecasts directory: {FORECASTS}")
print(f"Using insights directory: {INSIGHTS}")

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