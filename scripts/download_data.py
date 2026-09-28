import argparse, io, zipfile
from pathlib import Path
import requests

p=argparse.ArgumentParser(); p.add_argument("--dataset", choices=["uci"], required=True); a=p.parse_args()
url="https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
out=Path("data/raw/uci"); out.mkdir(parents=True,exist_ok=True)
r=requests.get(url,timeout=120); r.raise_for_status(); zipfile.ZipFile(io.BytesIO(r.content)).extractall(out)
print(out.resolve())
