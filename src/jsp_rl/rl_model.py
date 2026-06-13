import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class JSPActorCritic(nn.Module):
    def __init__(
        self,
        token_dim=16,
        hidden_dim=128,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        n_tokens=100,
    ):
        super().__init__()

        self.input_proj = nn.Linear(token_dim, hidden_dim)

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

    def encode(self, tokens):
        z = self.input_proj(tokens)
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

    def load_bc_actor(self, path, strict=False):
        state = torch.load(path, map_location="cpu")

        mapping = {
            "input_proj.": "input_proj.",
            "encoder.": "encoder.",
            "actor_head.": "actor_head.",
            "pos_embedding": "pos_embedding",
        }

        own = self.state_dict()
        loaded = {}

        for old_k, v in state.items():
            for src, dst in mapping.items():
                if old_k.startswith(src):
                    new_k = old_k.replace(src, dst, 1)

                    if new_k in own and own[new_k].shape == v.shape:
                        loaded[new_k] = v

        own.update(loaded)
        self.load_state_dict(own, strict=False)

        print(f"Loaded {len(loaded)} tensors from BC checkpoint.")
        if len(loaded) == 0:
            print("BC keys example:", list(state.keys())[:10])
            print("RL keys example:", list(own.keys())[:10])