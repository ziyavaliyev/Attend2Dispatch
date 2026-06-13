import torch
import torch.nn as nn
import torch.nn.functional as F
from jsp_rl.utils import compute_pna_degree_histogram
from torch_geometric.nn import GCNConv, GINConv, PNAConv, EGConv, GATConv, GATv2Conv


def make_conv(
    gnn_type: str,
    in_channels: int,
    out_channels: int,
    aggregators=None,
    scalers=None,
    deg=None,
    towers: int = 1,
    num_bases: int = 4,
    gat_heads: int = 4,
    gat_dropout: float = 0.0,
):
    if gnn_type == "gcn":
        return GCNConv(in_channels, out_channels)

    elif gnn_type == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        return GINConv(mlp)

    elif gnn_type == "egc":
        return EGConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=towers,
            num_bases=num_bases,
        )

    elif gnn_type == "pna":
        if deg is None:
            raise ValueError("PNAConv requires `deg`.")
        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "attenuation"]

        return PNAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )

    elif gnn_type == "gat":
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=True,
        )
    
    elif gnn_type == "gatv2":
        return GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=gat_heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=True,
        )

    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}")

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        gnn_type: str = "gcn",
        deg=None,
    ):
        super().__init__()
        self.conv1 = make_conv(
            gnn_type=gnn_type,
            in_channels=in_channels,
            out_channels=hidden_channels,
            deg=deg,
        )
        self.conv2 = make_conv(
            gnn_type=gnn_type,
            in_channels=hidden_channels,
            out_channels=out_channels,
            deg=deg,
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class VariationalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        gnn_type: str = "gcn",
        deg=None,
    ):
        super().__init__()
        self.conv1 = make_conv(
            gnn_type=gnn_type,
            in_channels=in_channels,
            out_channels=hidden_channels,
            deg=deg,
        )
        self.conv_mu = make_conv(
            gnn_type=gnn_type,
            in_channels=hidden_channels,
            out_channels=out_channels,
            deg=deg,
        )
        self.conv_logstd = make_conv(
            gnn_type=gnn_type,
            in_channels=hidden_channels,
            out_channels=out_channels,
            deg=deg,
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)