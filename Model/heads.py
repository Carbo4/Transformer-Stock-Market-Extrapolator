import math
import torch
import torch.nn as nn


class DriftHead(nn.Module):
    """Autoregressive transformer decoder predicting gaussian drift on CLOSE.
    Expects `memory` shaped (B, T_ctx, d_model) and `tgt` shaped (B, L_future)
    where `tgt` is the standardized previous-target sequence for teacher forcing
    (shifted right): the last context value followed by previous targets.
    Returns mean and logvar tensors shaped (B, L, 1).
    """
    def __init__(self, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.tgt_embed = nn.Linear(1, d_model)
        self.out = nn.Linear(d_model, 3)  # mean, logvar, skew

    def _causal_mask(self, L, device):
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, memory: torch.Tensor, t: torch.Tensor, tgt: torch.Tensor):
        # memory: (B, T_ctx, E), tgt: (B, L)
        B, L = tgt.shape
        device = memory.device
        # embed tgt -> (L, B, E)
        tgt_e = self.tgt_embed(tgt.unsqueeze(-1)).permute(1, 0, 2)
        # memory -> (S, N, E)
        mem = memory.permute(1, 0, 2)
        mask = self._causal_mask(L, device)
        out = self.decoder(tgt_e, mem, tgt_mask=mask)
        out = out.permute(1, 0, 2)
        stats = self.out(out)
        mean, logvar, skew = stats[..., 0:1], stats[..., 1:2], stats[..., 2:3]
        return mean, logvar, skew


class JumpHead(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.tgt_embed = nn.Linear(1, d_model)
        self.mask_out = nn.Linear(d_model, 1)
        self.mag_out = nn.Linear(d_model * 2 + 1, 1)

    def forward(self, memory: torch.Tensor, information_proxy: torch.Tensor):
        # memory: (B, T_ctx, E), information_proxy: (B, L)
        B, L = information_proxy.shape
        mem = memory.permute(1, 0, 2)  # (S, N, E)
        tgt = self.tgt_embed(information_proxy.unsqueeze(-1)).permute(1, 0, 2)
        # no causal mask (non-autoregressive)
        dec = self.decoder(tgt, mem)  # (L, B, E)
        dec = dec.permute(1, 0, 2)  # (B, L, E)

        mask_logits = self.mask_out(dec).squeeze(-1)

        # magnitude prediction: combine decoder output with a summary of memory
        mem_summary = memory[:, -1:, :].expand(-1, L, -1)
        mag_in = torch.cat([dec, mem_summary, information_proxy.unsqueeze(-1)], dim=-1)
        mag = self.mag_out(mag_in).squeeze(-1)
        return mask_logits, mag
