import torch
import torch.nn as nn
import torch.nn.functional as F

class GSELayer(nn.Module):

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.edge_update = nn.Sequential(nn.Linear(node_dim * 2 + edge_dim, edge_dim), nn.ReLU(inplace=True), nn.Linear(edge_dim, edge_dim))
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.attn_score = nn.Linear(edge_dim, 1)
        self.node_update = nn.Sequential(nn.Linear(node_dim + edge_dim, node_dim), nn.ReLU(inplace=True), nn.Linear(node_dim, node_dim))

    def forward(self, nodes, edge_feats):
        N = nodes.shape[0]
        center = nodes.unsqueeze(1).expand(N, N, -1)
        neighbor = nodes.unsqueeze(0).expand(N, N, -1)
        edge_input = torch.cat([center, neighbor - center, edge_feats], dim=-1)
        edge_feats_new = self.edge_norm(self.edge_update(edge_input) + edge_feats)
        attn = self.attn_score(edge_feats_new)
        attn = F.softmax(attn, dim=1)
        agg = (attn * edge_feats_new).sum(dim=1)
        nodes_new = self.node_update(torch.cat([nodes, agg], dim=-1))
        return (nodes_new, edge_feats_new)

class RadarGSE(nn.Module):

    def __init__(self, in_dim=3, hidden_dim=64, out_dim=64, n_layers=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True))
        self.edge_init = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(inplace=True))
        self.layers = nn.ModuleList([GSELayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, radar_points):
        N = radar_points.shape[0]
        if N == 0:
            return torch.zeros(0, self.output_proj[0].out_features, device=radar_points.device)
        x = self.input_proj(radar_points)
        center = x.unsqueeze(1).expand(N, N, -1)
        neighbor = x.unsqueeze(0).expand(N, N, -1)
        edge_feats = self.edge_init(torch.cat([center, neighbor - center], dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x, edge_feats = layer(x, edge_feats)
            x = norm(x + residual)
        return self.output_proj(x)

class BatchRadarGSE(nn.Module):

    def __init__(self, hidden_dim=64, out_dim=64, n_layers=3, max_points=512, max_depth=80.0):
        super().__init__()
        self.gse = RadarGSE(in_dim=3, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers)
        self.max_points = max_points
        self.max_depth = max_depth

    def forward(self, sparse_depth, H_full=None, W_full=None):
        B, H, W = sparse_depth.shape
        if H_full is None:
            H_full = H
        if W_full is None:
            W_full = W
        device = sparse_depth.device
        out_dim = self.gse.output_proj[0].out_features
        feat_map = torch.zeros(B, out_dim, H, W, device=device)
        for b in range(B):
            ys, xs = torch.where(sparse_depth[b] > 0)
            N = len(ys)
            if N == 0:
                continue
            if N > self.max_points:
                perm = torch.randperm(N, device=device)[:self.max_points]
                ys = ys[perm]
                xs = xs[perm]
                N = self.max_points
            depths = sparse_depth[b, ys, xs]
            pts_norm = torch.stack([ys.float() / H_full, xs.float() / W_full, depths / self.max_depth], dim=-1)
            point_features = self.gse(pts_norm)
            feat_map[b, :, ys, xs] = point_features.T.to(feat_map.dtype)
        return feat_map
