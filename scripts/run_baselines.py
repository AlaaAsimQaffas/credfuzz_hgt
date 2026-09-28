import argparse
from pathlib import Path
import pandas as pd, yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import load_dataset

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); df,cols,_,target,_=load_dataset(cfg["dataset"],cfg["data_dir"]); y=df[target].astype(int)
num=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]; cat=[c for c in cols if c not in num]; pre=ColumnTransformer([("num",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),num),("cat",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),cat)])
models={"logistic_regression":LogisticRegression(max_iter=1000,class_weight="balanced",n_jobs=-1),"random_forest":RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",n_jobs=-1,random_state=cfg["seed"])}; rows=[]
for name,model in models.items():
  for fold,(tr,te) in enumerate(StratifiedKFold(cfg["folds"],shuffle=True,random_state=cfg["seed"]).split(df,y),1):
    pipe=make_pipeline(pre,model); pipe.fit(df.iloc[tr][cols],y.iloc[tr]); p1=pipe.predict_proba(df.iloc[te][cols])[:,1]; rows.append({"model":name,"fold":fold,"roc_auc":roc_auc_score(y.iloc[te],p1)})
out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/"baseline_metrics.csv",index=False)
