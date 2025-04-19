# backend/scripts/build_medians.py
import pandas as pd, glob, os, json, pathlib

ARTEFACTS, FORECASTS = "artefacts", "forecasts"
pathlib.Path(ARTEFACTS).mkdir(exist_ok=True)

clusters = pd.read_csv(f"{ARTEFACTS}/clusters.csv")[["iso3", "cluster_lvl2"]]
targets  = ["co2_pc", "forest_land_EFProdPerCap"]

for tgt in targets:
    dfs=[]
    for f in glob.glob(f"{FORECASTS}/*/{tgt}.csv"):
        iso=f.split(os.sep)[1]
        df=pd.read_csv(f)[["year","yhat"]]; df["iso3"]=iso; dfs.append(df)
    if not dfs: continue
    (pd.concat(dfs).merge(clusters,on="iso3")
       .groupby(["cluster_lvl2","year"])["yhat"].median().reset_index()
       .to_csv(f"{ARTEFACTS}/cluster_medians_{tgt}.csv", index=False))
    print("✓ wrote cluster_medians_", tgt)

json.dump({"targets":targets,"clusters":clusters.to_dict("records")},
          open(f"{ARTEFACTS}/manifest.json","w"), indent=2)
print("✓ manifest refreshed")