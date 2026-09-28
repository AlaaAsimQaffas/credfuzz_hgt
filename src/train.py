import copy, time
import numpy as np
import torch
from torch import nn
from .graph import build_graph
from .metrics import evaluate, select_threshold
from .model import CredFuzzHGT

def tensors(x, temporal, mask, device):
    return torch.tensor(x, device=device), torch.tensor(temporal, device=device), torch.tensor(mask, device=device)

def train_fold(x_train, t_train, m_train, y_train, x_val, t_val, m_val, y_val, x_test, t_test, m_test, y_test, cfg, flags=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); flags = flags or {}
    model = CredFuzzHGT(x_train.shape[1], t_train.shape[2], cfg["model"], **flags).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=.5)
    pos_weight = torch.tensor([(y_train==0).sum()/max(1,(y_train==1).sum())], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight if flags.get("cost_sensitive", True) else None)
    sets = [tensors(*z, device) for z in [(x_train,t_train,m_train),(x_val,t_val,m_val),(x_test,t_test,m_test)]]
    graphs = [tuple(v.to(device) for v in build_graph(z[0], cfg["model"]["neighbors"])) for z in [(x_train,),(x_val,),(x_test,)]]
    best = (-np.inf, None, 0); started = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train(); opt.zero_grad(); logits, aux = model(*sets[0], *graphs[0])
        loss = loss_fn(logits, torch.tensor(y_train, device=device, dtype=torch.float32))
        loss = loss + cfg["training"]["fuzzy_regularization"]*((aux["membership"]-graphs[0][1].mean(1))**2).mean()
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"]); opt.step()
        model.eval()
        with torch.no_grad(): vp = model(*sets[1], *graphs[1])[0].sigmoid().cpu().numpy()
        score = evaluate(y_val, vp, .5)["roc_auc"]; scheduler.step(score)
        if score > best[0]: best = (score, copy.deepcopy(model.state_dict()), epoch)
        if epoch-best[2] >= cfg["training"]["patience"]: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        vp = model(*sets[1], *graphs[1])[0].sigmoid().cpu().numpy(); logits, aux = model(*sets[2], *graphs[2]); tp = logits.sigmoid().cpu().numpy()
    threshold = select_threshold(y_val, vp); metrics = evaluate(y_test, tp, threshold)
    metrics.update({"best_epoch": best[2]+1, "seconds": time.time()-started, "parameters": sum(p.numel() for p in model.parameters())})
    explanations = {k: v.detach().cpu().numpy() for k,v in aux.items() if v is not None}
    return metrics, tp, explanations
