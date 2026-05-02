import math
import torch
import torch.nn as nn

def causal_mask(L, device) -> torch.Tensor:
    return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)

class DriftHead(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dropout=0.1, input_dim=1) -> None:
        super().__init__()
        dec_layer    = nn.TransformerDecoderLayer(
            d_model                     ,
            nhead                       ,
            dim_feedforward = 4*d_model ,
            dropout         = dropout   ,
            batch_first     = True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.embed   = nn.Linear(input_dim, d_model)
        self.out     = nn.Linear(d_model, 3)  # mean, logvar, skew    

    def forward(
            self, 
            memory           : torch.Tensor , 
            decoder_sequence : torch.Tensor
        ) -> tuple[torch.Tensor , torch.Tensor , torch.Tensor]:

        device = memory.device
        B, L, D = decoder_sequence.shape

        # memory: (B, T_ctx, E), decoder_sequence: (B, L, D)
        decoder_se  = self.embed(decoder_sequence)  # (B, L, d_model)
        mask        = causal_mask(L, device)
        out         = self.decoder(decoder_se, memory, tgt_mask=mask)
        stats       = self.out(out)
        return stats[..., 0:1], stats[..., 1:2], stats[..., 2:3]  # mean, logvar, skew


class JumpHead(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dropout=0.1, input_dim=1) -> None:
        super().__init__()
        dec_layer    = nn.TransformerDecoderLayer(
            d_model                     ,
            nhead                       ,
            dim_feedforward = 4*d_model ,
            dropout         = dropout   ,
            batch_first     = True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.embed   = nn.Linear(input_dim, d_model)
        self.out     = nn.Linear(d_model, 3)  # mean, logvar, skew    

    def forward(
            self, 
            memory           : torch.Tensor , 
            decoder_sequence : torch.Tensor ,
        ) -> tuple[torch.Tensor , torch.Tensor , torch.Tensor]:
        device = memory.device
        B, L, D = decoder_sequence.shape

        # memory: (B, T_ctx, E), decoder_sequence: (B, L, D)
        decoder_se  = self.embed(decoder_sequence)  # (B, L, d_model)
        mask   = causal_mask(L, device)
        out    = self.decoder(decoder_se, memory, tgt_mask=mask)
        stats  = self.out(out)
        return stats[..., 0:1], stats[..., 1:2], stats[..., 2:3] # mean, logvar, skew
