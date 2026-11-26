import torch
import torch.nn as nn

from x_transformers import ContinuousTransformerWrapper, Encoder

from .training import sinusoidal_embedding
from ..data import TIME_ID

class Encoder_Transformer(nn.Module):
    def __init__(self, n_embd=128, n_layers=6, n_heads=4, *, variables=None, variable_ids=None, use_adaptive_rmsnorm=True,
                 feature_embedding="shared"):
        '''Pass either variables (list[Variable]) or variable_ids (list[int]), in column order.
        feature_embedding: "shared" (one value layer + variable-id embedding) or "per_feature" (own linear layer per variable id).'''
        super().__init__()
        if (variables is None) == (variable_ids is None):
            raise ValueError("pass either variables or variable_ids")
        if feature_embedding not in ("shared", "per_feature"):
            raise ValueError(f"feature_embedding must be 'shared' or 'per_feature', not {feature_embedding!r}")
        ids = [v.id for v in variables] if variables is not None else list(variable_ids)
        self.n_features = len(ids)
        self.n_embd = n_embd
        self.use_adaptive_rmsnorm = use_adaptive_rmsnorm
        self.feature_embedding = feature_embedding
        self.register_buffer("token_ids", torch.tensor(ids + [TIME_ID]))  # features, then time last
        if feature_embedding == "shared":
            self.variable_embedding = nn.Embedding(max(ids) + 1, n_embd)  # looked up by id
            self.value_embedding = nn.Linear(1, n_embd, bias=False)  # features' embedding
        else:  # per_feature: own Linear(1, n_embd) with bias per variable id, as two id-indexed tables
            self.value_weight = nn.Embedding(max(ids) + 1, n_embd)  # slope per variable id
            self.value_bias = nn.Embedding(max(ids) + 1, n_embd)    # offset per variable id (replaces the variable embedding)
        self.known_embedding = nn.Embedding(2, n_embd)
        self.transformer = ContinuousTransformerWrapper(
            dim_in = None,                 # no input layer: the wrapper gets ready token embeddings
            dim_out = 1,
            max_seq_len = self.n_features + 1, # plus time token
            use_abs_pos_emb = False,       # token identity comes from var_emb instead
            attn_layers = Encoder(
                dim = n_embd,
                depth = n_layers,
                heads = n_heads,
                use_adaptive_rmsnorm = use_adaptive_rmsnorm,  # the time embedding scales every norm
                dim_condition = n_embd,
            )
)

    def forward(self, idx, known=None):
        '''
        idx:   (B, n_features + 1) = [features..., time]
        known: (n_features,) or (B, n_features) bool; True = conditioned on
        '''
        B, _ = idx.shape
        if known is None:
            known = torch.zeros((B, self.n_features), dtype=torch.bool, device=idx.device)
        else:
            known = torch.as_tensor(known, dtype=torch.bool, device=idx.device)
            if known.ndim == 1:
                known = known.expand(B, -1)
        known_full = torch.cat([known, known.new_ones((B, 1))], dim=1)  # (B,T)
        kno_emb = self.known_embedding(known_full.long())  # (B,T,C)

        x = idx[:, :-1].unsqueeze(-1)                                      # (B, n_features, 1)
        if self.feature_embedding == "shared":
            feature_tokens = self.value_embedding(x)                       # (B, n_features, C)
            var_emb = self.variable_embedding(self.token_ids)
            sum_embeds = kno_emb + var_emb        # known + variable-id embedding, added inside
        else:
            feature_ids = self.token_ids[:-1]                              # ids of the feature columns
            feature_tokens = x * self.value_weight(feature_ids) + self.value_bias(feature_ids)  # (B, n_features, C)
            sum_embeds = kno_emb                  # identity comes from value_bias
        t_emb = sinusoidal_embedding(idx[:, -1], self.n_embd)              # (B, C)
        tokens = torch.cat([feature_tokens, t_emb[:, None, :]], dim=1)     # (B, T, C), time token last
        extra = {"condition": t_emb} if self.use_adaptive_rmsnorm else {}  # time also conditions the norms

        velocity = self.transformer(tokens, sum_embeds=sum_embeds, **extra)
        velocity = velocity.squeeze(-1)[:, :-1]  # (B, n_features)

        return velocity.masked_fill(known, 0.0)
