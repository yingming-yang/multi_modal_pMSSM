import torch
import torch.nn as nn

from x_transformers import ContinuousTransformerWrapper, Encoder

class Encoder_Transformer(nn.Module):
    def __init__(self, n_embd=128, n_layers=6, n_heads=4, n_features=13):
        super().__init__()
        self.n_features = n_features
        self.known_embedding = nn.Embedding(2, n_embd)
        self.transformer = ContinuousTransformerWrapper(
            dim_in = 1,
            dim_out = 1,
            max_seq_len = n_features + 1, # plus time dimension
            attn_layers = Encoder(
                dim = n_embd,
                depth = n_layers,
                heads = n_heads
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

        velocity = self.transformer(idx.unsqueeze(-1), sum_embeds=kno_emb)
        velocity = velocity.squeeze(-1)[:, :-1]  # (B,2)

        return velocity.masked_fill(known, 0.0)