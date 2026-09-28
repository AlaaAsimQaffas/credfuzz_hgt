import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def build_graph(x, k=10):
    n = len(x); k = min(k, max(1, n-1))
    cols = min(32, x.shape[1]); view = x[:, :cols]
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(view)
    dist, idx = nn.kneighbors(view); dist, idx = dist[:,1:], idx[:,1:]
    src = idx.reshape(-1); dst = np.repeat(np.arange(n), k)
    scale = np.median(dist) + 1e-8; sim = np.exp(-(dist.reshape(-1)**2)/(2*scale**2))
    conf = np.ones_like(sim); rep = sim.copy(); rec = np.ones_like(sim)
    evidence = np.stack([sim, rep, conf, rec], 1).astype(np.float32)
    self_idx = np.arange(n); src = np.concatenate([src, self_idx]); dst = np.concatenate([dst, self_idx])
    evidence = np.concatenate([evidence, np.ones((n,4), np.float32)])
    return torch.tensor(np.stack([src,dst]), dtype=torch.long), torch.tensor(evidence)
