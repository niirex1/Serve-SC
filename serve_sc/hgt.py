"""Heterogeneous graph transformer for Stage-1 detection (Section IV-A).

This is the model used for the paper's detection results. It requires PyTorch.
A two-layer HGT with type-specific key/query/value projections and multi-head
attention over typed edges, mean-pooled over function nodes, fused with the
12-dimensional dense cue vector, and read out by a linear multi-label head.

The package falls back to a lightweight numpy classifier (see ``detect.py``)
when torch is not installed, so the demo and tests run without it; that fallback
is a convenience smoke-test path, not the method.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False

from .config import (DENSE_CUE_DIM, EDGE_TYPES, HGT_DROPOUT, HGT_HEADS,
                     HGT_HIDDEN, HGT_LAYERS, K, NODE_FEAT_DIM)

if TORCH_AVAILABLE:

    class HGTLayer(nn.Module):
        """One heterogeneous graph-transformer layer with per-edge-type K/Q/V."""

        def __init__(self, dim: int, heads: int):
            super().__init__()
            assert dim % heads == 0
            self.dim, self.heads, self.dk = dim, heads, dim // heads
            self.k = nn.ModuleDict({e: nn.Linear(dim, dim) for e in EDGE_TYPES})
            self.q = nn.ModuleDict({e: nn.Linear(dim, dim) for e in EDGE_TYPES})
            self.v = nn.ModuleDict({e: nn.Linear(dim, dim) for e in EDGE_TYPES})
            self.out = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(HGT_DROPOUT)

        def forward(self, h: "torch.Tensor", edges: dict) -> "torch.Tensor":
            n = h.size(0)
            agg = torch.zeros_like(h)
            deg = torch.zeros(n, device=h.device)
            for et in EDGE_TYPES:
                ei = edges.get(et)
                if ei is None or ei.shape[1] == 0:
                    continue
                src, dst = ei[0], ei[1]
                ks = self.k[et](h[src]).view(-1, self.heads, self.dk)
                qd = self.q[et](h[dst]).view(-1, self.heads, self.dk)
                vs = self.v[et](h[src]).view(-1, self.heads, self.dk)
                score = (ks * qd).sum(-1) / (self.dk ** 0.5)      # (E, heads)
                score = torch.exp(score - score.max())
                msg = (vs * score.unsqueeze(-1)).reshape(-1, self.dim)
                agg = agg.index_add(0, dst, msg)
                w = score.mean(-1)
                deg = deg.index_add(0, dst, w)
            deg = deg.clamp(min=1e-6).unsqueeze(-1)
            agg = self.out(agg / deg)
            return self.norm(h + self.drop(F.gelu(agg)))

    class HGTDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(NODE_FEAT_DIM, HGT_HIDDEN)
            self.layers = nn.ModuleList(
                [HGTLayer(HGT_HIDDEN, HGT_HEADS) for _ in range(HGT_LAYERS)])
            self.head = nn.Linear(HGT_HIDDEN + DENSE_CUE_DIM, K)

        def forward(self, feats, edges, func_idx, dense):
            h = self.proj(feats)
            for layer in self.layers:
                h = layer(h, edges)
            pooled = h[func_idx].mean(0) if func_idx.numel() > 0 else h.mean(0)
            fused = torch.cat([pooled, dense], dim=-1)
            return self.head(fused)               # logits (K,)

    def _to_edges(graph):
        return {et: torch.as_tensor(ei, dtype=torch.long)
                for et, ei in graph.edges.items()}

    def train_hgt(graphs, labels, seed=0, max_epochs=None):
        """Train the HGT on a list of ContractGraph and (N, K) labels."""
        import torch
        from .config import LR, MAX_EPOCHS, WEIGHT_DECAY
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HGTDetector()
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        Y = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        pos = Y.mean(0).clamp(1e-3, 1 - 1e-3)
        pos_weight = (1 - pos) / pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        feats = [torch.as_tensor(g.node_feats) for g in graphs]
        edges = [_to_edges(g) for g in graphs]
        fidx = [torch.as_tensor(g.func_idx, dtype=torch.long) for g in graphs]
        dense = [torch.as_tensor(g.dense_cues) for g in graphs]
        model.train()
        for _ in range(max_epochs or MAX_EPOCHS):
            opt.zero_grad()
            logits = torch.stack([model(feats[i], edges[i], fidx[i], dense[i])
                                  for i in range(len(graphs))])
            loss = loss_fn(logits, Y)
            loss.backward()
            opt.step()
        model.eval()
        return model

    def predict_hgt(model, graphs) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = []
            for g in graphs:
                logits = model(torch.as_tensor(g.node_feats), _to_edges(g),
                               torch.as_tensor(g.func_idx, dtype=torch.long),
                               torch.as_tensor(g.dense_cues))
                out.append(torch.sigmoid(logits).numpy())
        return np.asarray(out)
