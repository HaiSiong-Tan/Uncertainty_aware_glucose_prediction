import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional encoding added to input embeddings for Transformer models."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1)]


class e_Transformers(nn.Module):
    """
    Transformer-based time-series model producing evidential outputs
    (alpha, beta, nu, gamma) for uncertainty estimation.
    """
    def __init__(
        self,
        input_dim=4,
        d_model=64,
        n_heads=4,
        num_layers=2,
        ff_dim=128,
        output_dim=6,
        max_len=100
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.softplus = nn.Softplus()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Evidential outputs
        self.output_alpha = nn.Linear(d_model, output_dim)
        self.output_beta = nn.Linear(d_model, output_dim)
        self.output_nu = nn.Linear(d_model, output_dim)
        self.output_gamma = nn.Linear(d_model, output_dim)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=causal_mask)

        h_last = h[:, -1, :]
        alpha = 1.1 + self.softplus(self.output_alpha(h_last))
        beta  = self.softplus(self.output_beta(h_last))
        nu    = 0.1 + self.softplus(self.output_nu(h_last))
        gamma = self.output_gamma(h_last)

        return alpha, beta, nu, gamma