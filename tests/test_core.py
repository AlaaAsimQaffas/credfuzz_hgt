import numpy as np, torch
from src.graph import build_graph
from src.metrics import evaluate, select_threshold
from src.model import CredFuzzHGT

def test_graph_and_forward():
    x=np.random.default_rng(1).normal(size=(24,8)).astype("float32"); edge,evidence=build_graph(x,3)
    cfg={"hidden_dim":16,"graph_layers":1,"graph_heads":4,"temporal_layers":1,"temporal_heads":4,"feedforward_dim":32,"dropout":.1}
    model=CredFuzzHGT(8,3,cfg); logits,aux=model(torch.tensor(x),torch.randn(24,6,3),torch.ones(24,6,dtype=torch.bool),edge,evidence)
    assert logits.shape==(24,); assert torch.isfinite(logits).all(); assert ((aux["membership"]>=0)&(aux["membership"]<=1)).all()

def test_metrics():
    y=np.array([0,0,1,1]); p=np.array([.1,.4,.6,.9]); t=select_threshold(y,p); assert evaluate(y,p,t)["roc_auc"]==1.0
