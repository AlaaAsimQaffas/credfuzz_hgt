import math
import torch
from torch import nn

def segment_softmax(scores, dst, n):
    maximum = torch.full((n, scores.size(1)), -torch.inf, device=scores.device)
    maximum.scatter_reduce_(0, dst[:, None].expand_as(scores), scores, reduce="amax", include_self=True)
    exp = (scores - maximum[dst]).exp()
    denom = torch.zeros_like(maximum).scatter_add_(0, dst[:, None].expand_as(exp), exp)
    return exp / (denom[dst] + 1e-8)

class FuzzyGraphLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__(); self.heads = heads; self.dh = dim // heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*2, dim)); self.drop = nn.Dropout(dropout)
    def forward(self, x, edge_index, membership):
        src, dst = edge_index; n = x.size(0)
        q = self.q(x).view(n, self.heads, self.dh); k = self.k(x).view(n, self.heads, self.dh); v = self.v(x).view(n, self.heads, self.dh)
        score = (q[dst] * k[src]).sum(-1) / math.sqrt(self.dh) + membership.clamp_min(1e-8).log()[:, None]
        alpha = segment_softmax(score, dst, n)
        msg = alpha[..., None] * membership[:, None, None] * v[src]
        agg = torch.zeros_like(v); agg.index_add_(0, dst, msg)
        h = self.norm1(x + self.drop(self.o(agg.flatten(1))))
        return self.norm2(h + self.drop(self.ff(h))), alpha

class CredFuzzHGT(nn.Module):
    def __init__(self, static_dim, temporal_dim, cfg, use_graph=True, use_temporal=True, use_fuzzy=True, use_uncertainty=True):
        super().__init__(); d = cfg["hidden_dim"]; self.flags = use_graph, use_temporal, use_fuzzy, use_uncertainty
        self.static = nn.Sequential(nn.Linear(static_dim, d), nn.LayerNorm(d), nn.GELU())
        self.graph = nn.ModuleList([FuzzyGraphLayer(d, cfg["graph_heads"], cfg["dropout"]) for _ in range(cfg["graph_layers"])])
        self.temporal_in = nn.Linear(temporal_dim, d); self.position = nn.Parameter(torch.randn(1, 32, d) * .02)
        layer = nn.TransformerEncoderLayer(d, cfg["temporal_heads"], cfg["feedforward_dim"], cfg["dropout"], batch_first=True, activation="gelu", norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, cfg["temporal_layers"]); self.pool = nn.Linear(d, 1)
        self.gate = nn.Linear(d*2+2, d); self.residual = nn.Linear(d*2, d); self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d//2, 1))
    def forward(self, static, temporal, mask, edge_index, evidence):
        use_graph, use_temporal, use_fuzzy, use_uncertainty = self.flags
        membership = evidence.mean(1).clamp(.001, .999) if use_fuzzy else torch.ones(evidence.size(0), device=evidence.device)
        s = self.static(static); attention = None
        if use_graph:
            for layer in self.graph: s, attention = layer(s, edge_index, membership)
        t = self.temporal_in(temporal) + self.position[:, :temporal.size(1)]
        if use_temporal: t = self.temporal(t, src_key_padding_mask=~mask)
        weights = self.pool(t).squeeze(-1).masked_fill(~mask, -1e9).softmax(1); t = (t * weights[..., None]).sum(1)
        src, dst = edge_index; entropy = -(membership*membership.log()+(1-membership)*(1-membership).log())/math.log(2)
        su = torch.zeros(static.size(0), device=static.device); count = torch.zeros_like(su); su.index_add_(0, dst, entropy); count.index_add_(0, dst, torch.ones_like(entropy)); su = su/(count+1e-8)
        tu = -(weights.clamp_min(1e-8)*weights.clamp_min(1e-8).log()).sum(1)/math.log(max(2, weights.size(1)))
        if not use_uncertainty: su = torch.zeros_like(su); tu = torch.zeros_like(tu)
        if not use_graph: s = torch.zeros_like(s)
        if not use_temporal: t = torch.zeros_like(t)
        gate = torch.sigmoid(self.gate(torch.cat([s,t,su[:,None],tu[:,None]],1)))
        z = self.norm(gate*s+(1-gate)*t+self.residual(torch.cat([s,t],1)))
        return self.head(z).squeeze(1), {"membership": membership, "graph_attention": attention, "temporal_attention": weights, "gate": gate, "structural_uncertainty": su, "temporal_uncertainty": tu}
