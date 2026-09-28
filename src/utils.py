import json, os, platform, random, sys
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def versions():
    import pandas, sklearn, yaml
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__, "pyyaml": yaml.__version__, "cuda": torch.version.cuda}

def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=float), encoding="utf-8")
