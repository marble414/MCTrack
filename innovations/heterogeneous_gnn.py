import torch
from torch import nn
import dgl
from dgl.nn import GraphConv, HeteroGraphConv

class EdgeWeightPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, graph, h):
        # TODO: compute edge weights for each edge type
        raise NotImplementedError

class NodeClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h):
        # TODO: classify nodes
        raise NotImplementedError

class HeterogeneousSpatioTemporalGNN(nn.Module):
    """Heterogeneous spatio-temporal graph neural network."""
    def __init__(self, hidden_dim=256, num_layers=3):
        super().__init__()
        self.node_types = ["lidar_det", "image_det", "tracklet", "virtual"]
        self.edge_types = [
            ("lidar_det", "spatial", "lidar_det"),
            ("image_det", "spatial", "image_det"),
            ("lidar_det", "cross_modal", "image_det"),
            ("tracklet", "temporal", "lidar_det"),
            ("tracklet", "temporal", "image_det"),
            ("virtual", "occlusion", "tracklet"),
        ]
        self.conv_layers = nn.ModuleList(
            [
                HeteroGraphConv(
                    {
                        etype: GraphConv(hidden_dim, hidden_dim)
                        for etype in self.edge_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.edge_weight_predictor = EdgeWeightPredictor(hidden_dim)
        self.node_classifier = NodeClassifier(hidden_dim, num_classes=2)

    def construct_heterogeneous_graph(self, detections, tracklets, t):
        # TODO: implement graph construction
        raise NotImplementedError

    def forward(self, het_graph, node_features):
        h = node_features
        for conv in self.conv_layers:
            edge_weights = self.edge_weight_predictor(het_graph, h)
            h = conv(het_graph, h, mod_kwargs={"edge_weight": edge_weights})
            h = {ntype: torch.relu(feat) for ntype, feat in h.items()}
        association_scores = self.node_classifier(h)
        return association_scores, h
