import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FoldPreprocessor:
    def __init__(self):
        self.num = []; self.cat = []
    def fit(self, frame):
        self.num = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.cat = [c for c in frame if c not in self.num]
        self.medians = frame[self.num].median()
        filled = frame[self.num].fillna(self.medians)
        self.lower = filled.quantile(.005); self.upper = filled.quantile(.995)
        self.scaler = StandardScaler().fit(filled.clip(self.lower, self.upper))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2).fit(frame[self.cat].fillna("unknown").astype(str)) if self.cat else None
        return self
    def transform(self, frame):
        missing = frame[self.num].isna().to_numpy(np.float32)
        numeric = self.scaler.transform(frame[self.num].fillna(self.medians).clip(self.lower, self.upper)).astype(np.float32)
        categorical = self.encoder.transform(frame[self.cat].fillna("unknown").astype(str)).astype(np.float32) if self.encoder else np.empty((len(frame), 0), np.float32)
        return np.concatenate([numeric, missing, categorical], axis=1)

def fit_temporal(train, others):
    med = np.nanmedian(train, axis=(0, 1)); med = np.nan_to_num(med)
    train_f = np.where(np.isnan(train), med, train)
    mean = train_f.mean(axis=(0, 1)); std = train_f.std(axis=(0, 1)) + 1e-8
    return [((np.where(np.isnan(x), med, x) - mean) / std).astype(np.float32) for x in [train] + others]
