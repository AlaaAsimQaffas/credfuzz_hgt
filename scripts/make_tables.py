import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument("--results-dir",default="results"); a=p.parse_args(); rows=[]
for f in Path(a.results_dir).glob("**/summary.json"):
    s=json.loads(f.read_text()); rows.append({"dataset":f.parent.parent.name,"run":f.parent.name,**{k:f"{v['mean']:.3f} ± {v['sd']:.3f}" for k,v in s.items() if isinstance(v,dict)}})
pd.DataFrame(rows).to_csv(Path(a.results_dir)/"publication_summary.csv",index=False)
