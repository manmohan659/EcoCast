# Claude Reference Information

## Sustainability Pipeline
sustainability_pipeline_v2.ipynb is the main orchestration notebook that:
- Processes raw data from data_raw/ into clean data_work/ files
- Performs country clustering and saves to artefacts/clusters.csv
- Generates forecasts for CO2 per capita and forest land productivity
- Trains a LightGBM model to predict HDI from sustainability metrics
- Consolidates model performance in artefacts/model_scores.csv

When working on this project, refer to this notebook for the complete data processing and modeling pipeline.