import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GATv2Conv

class JSPGATActorCritic(nn.Module):
    def __init__(
        self,
        token_dim,
        graph_feat_dim,
        hidden_dim=128,
        gat_hidden_dim=128,
        gat_out_dim=128,
        gat_heads=4,
        gat_layers=2,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        n_tokens=100,
    ):
        super().__init__()

        self.n_tokens = n_tokens
        self.graph_feat_dim = graph_feat_dim

        self.gat_layers = nn.ModuleList()

        if gat_layers == 1:
            self.gat_layers.append(
                GATv2Conv(
                    graph_feat_dim,
                    gat_out_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                )
            )
        else:
            self.gat_layers.append(
                GATv2Conv(
                    graph_feat_dim,
                    gat_hidden_dim,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                )
            )

            for _ in range(gat_layers - 2):
                self.gat_layers.append(
                    GATv2Conv(
                        gat_hidden_dim * gat_heads,
                        gat_hidden_dim,
                        heads=gat_heads,
                        concat=True,
                        dropout=dropout,
                    )
                )

            self.gat_layers.append(
                GATv2Conv(
                    gat_hidden_dim * gat_heads,
                    gat_out_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                )
            )

        self.input_proj = nn.Linear(gat_out_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic_value = nn.Linear(hidden_dim, 1)

    def _edge_index_from_A(self, A):
        src, dst = torch.nonzero(A > 0, as_tuple=True)

        if src.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=A.device)

        return torch.stack([src, dst], dim=0)

    def graph_encode(self, tokens):
        B, T, D = tokens.shape

        A_batch = tokens[:, :, :self.n_tokens]
        X_batch = tokens[:, :, self.n_tokens:]

        zs = []

        for b in range(B):
            A = A_batch[b]
            x = X_batch[b]
            edge_index = self._edge_index_from_A(A)

            z = x
            for i, conv in enumerate(self.gat_layers):
                z = conv(z, edge_index)
                if i < len(self.gat_layers) - 1:
                    z = torch.relu(z)

            zs.append(z)

        return torch.stack(zs, dim=0)

    def encode(self, tokens):
        z = self.graph_encode(tokens)
        z = self.input_proj(z)
        z = z + self.pos_embedding[:, : z.size(1), :]
        return self.encoder(z)

    def get_logits_and_value(self, tokens, mask=None):
        z = self.encode(tokens)

        logits = self.actor_head(z).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)

        scores = self.critic_pool(z).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = (z * weights.unsqueeze(-1)).sum(dim=1)
        value = self.critic_value(pooled)

        return logits, value

    def get_value(self, tokens):
        _, value = self.get_logits_and_value(tokens, mask=None)
        return value

    def get_action_and_value(self, tokens, mask, action=None):
        logits, value = self.get_logits_and_value(tokens, mask)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value