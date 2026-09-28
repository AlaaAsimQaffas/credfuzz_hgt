import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.data import load_dataset, temporal_array
from src.preprocessing import FoldPreprocessor, fit_temporal
from src.train import train_fold
from src.utils import seed_everything, save_json, versions

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--smoke-test",action="store_true"); p.add_argument("--flags",default="{}"); p.add_argument("--run-name",default="complete"); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text()); seed_everything(cfg["seed"]); frame, static_cols, groups, target, entity=load_dataset(cfg["dataset"],cfg["data_dir"])
if a.smoke_test: frame=frame.sample(min(1200,len(frame)),random_state=cfg["seed"]); cfg["folds"]=2; cfg["training"]["epochs"]=2; cfg["training"]["patience"]=2; cfg["model"]["hidden_dim"]=32; cfg["model"]["graph_layers"]=1; cfg["model"]["feedforward_dim"]=64
y=frame[target].astype(int).to_numpy(); raw_t, raw_m=temporal_array(frame,groups); splitter=StratifiedKFold(cfg["folds"],shuffle=True,random_state=cfg["seed"]); out=Path(cfg["output_dir"])/a.run_name; out.mkdir(parents=True,exist_ok=True); rows=[]
for fold,(dev,test) in enumerate(splitter.split(frame,y),1):
    train,val=train_test_split(dev,test_size=cfg["validation_fraction"],stratify=y[dev],random_state=cfg["seed"]+fold)
    prep=FoldPreprocessor().fit(frame.iloc[train][static_cols]); xtr=prep.transform(frame.iloc[train][static_cols]); xv=prep.transform(frame.iloc[val][static_cols]); xt=prep.transform(frame.iloc[test][static_cols])
    ttr,tv,tt=fit_temporal(raw_t[train],[raw_t[val],raw_t[test]])
    metrics,pred,ex=train_fold(xtr,ttr,raw_m[train],y[train],xv,tv,raw_m[val],y[val],xt,tt,raw_m[test],y[test],cfg,json.loads(a.flags))
    metrics["fold"]=fold; rows.append(metrics); pd.DataFrame({"index":test,"y_true":y[test],"probability":pred}).to_csv(out/f"fold_{fold}_predictions.csv",index=False)
    np.savez_compressed(out/f"fold_{fold}_explanations.npz",**ex); save_json(out/f"fold_{fold}_metrics.json",metrics)
pd.DataFrame(rows).to_csv(out/"fold_metrics.csv",index=False); summary={c:{"mean":float(np.mean([r[c] for r in rows])),"sd":float(np.std([r[c] for r in rows],ddof=1))} for c in rows[0] if c not in {"fold"}}
save_json(out/"summary.json",summary); save_json(out/"provenance.json",{"config":cfg,"versions":versions(),"device":str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}); print(json.dumps(summary,indent=2))
