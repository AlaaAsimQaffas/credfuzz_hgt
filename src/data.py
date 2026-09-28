from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_STATUSES = {"charged off", "default", "late (31-120 days)", "does not meet the credit policy. status:charged off"}
NONDEFAULT_STATUSES = {"fully paid", "current", "does not meet the credit policy. status:fully paid"}

def _uci(root):
    files = list(Path(root).glob("*.xls*")) + list(Path(root).glob("*.csv"))
    if not files: raise FileNotFoundError("Run: python scripts/download_data.py --dataset uci")
    f = files[0]
    df = pd.read_excel(f, header=1) if f.suffix.lower().startswith(".xls") else pd.read_csv(f)
    target = next(c for c in df if "default payment" in c.lower())
    df = df.rename(columns={target: "target"})
    static = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    temporal = [[f"PAY_{x}" if x != 1 else "PAY_0", f"BILL_AMT{x}", f"PAY_AMT{x}"] for x in range(1, 7)]
    temporal[0][0] = "PAY_0"
    return df, static, temporal, "target", "ID"

def _home(root):
    root = Path(root); app = root / "application_train.csv"
    if not app.exists(): raise FileNotFoundError("Place Home Credit competition CSV files in data/raw/home_credit/")
    df = pd.read_csv(app)
    excluded = {"TARGET", "SK_ID_CURR"}
    static = [c for c in df.columns if c not in excluded]
    temporal = []
    inst = root / "installments_payments.csv"
    if inst.exists():
        pay = pd.read_csv(inst, usecols=["SK_ID_CURR", "NUM_INSTALMENT_NUMBER", "AMT_INSTALMENT", "AMT_PAYMENT", "DAYS_ENTRY_PAYMENT"])
        pay["PAY_RATIO"] = pay.AMT_PAYMENT / pay.AMT_INSTALMENT.replace(0, np.nan)
        agg = pay.sort_values("NUM_INSTALMENT_NUMBER").groupby("SK_ID_CURR").tail(12)
        for t in range(12):
            part = agg.groupby("SK_ID_CURR").nth(t)[["PAY_RATIO", "DAYS_ENTRY_PAYMENT"]].add_suffix(f"_{t}")
            df = df.join(part, on="SK_ID_CURR"); temporal.append(list(part.columns))
    return df, static, temporal, "TARGET", "SK_ID_CURR"

def _lending(root):
    files = list(Path(root).glob("*.csv"))
    if not files: raise FileNotFoundError("Place the LendingClub loan CSV in data/raw/lendingclub/")
    df = pd.read_csv(files[0], low_memory=False)
    status = df["loan_status"].astype(str).str.lower()
    keep = status.isin(DEFAULT_STATUSES | NONDEFAULT_STATUSES)
    df = df.loc[keep].copy(); df["target"] = status[keep].isin(DEFAULT_STATUSES).astype(int)
    leakage = {"loan_status", "target", "recoveries", "collection_recovery_fee", "total_rec_prncp", "total_pymnt", "last_pymnt_amnt", "out_prncp", "out_prncp_inv"}
    static = [c for c in df if c not in leakage and df[c].nunique(dropna=True) < max(5000, len(df)//2)]
    temporal_candidates = [[c for c in df if c.startswith(prefix)] for prefix in ("delinq_", "inq_last_", "acc_now_")]
    temporal = [x for x in temporal_candidates if x]
    return df, static, temporal, "target", "id" if "id" in df else None

def load_dataset(name, root):
    return {"uci": _uci, "home_credit": _home, "lendingclub": _lending}[name](root)

def temporal_array(df, groups):
    if not groups: return np.zeros((len(df), 1, 1), dtype=np.float32), np.ones((len(df), 1), dtype=bool)
    width = max(map(len, groups)); out = np.full((len(df), len(groups), width), np.nan, dtype=np.float32)
    for t, cols in enumerate(groups): out[:, t, :len(cols)] = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    mask = ~np.isnan(out).all(axis=2)
    return out, mask
