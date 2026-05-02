import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, 
        encoder_layer        : nn.TransformerEncoderLayer , 
        num_layers           : int                        ,  
        norm                 : nn.Module | None = None    , 
        enable_nested_tensor : bool             = True    , 
        mask_check           : bool             = True
    ) -> None:
        super().__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer        , 
            num_layers           , 
            norm                 , 
            enable_nested_tensor , 
            mask_check
        )
        self.n_heads = encoder_layer.self_attn.num_heads
        self.register_buffer(
            "slopes",
            -torch.exp2(
                torch.linspace(0, -8, self.n_heads)
            )[None, :, None, None],
        )  # [1, n_heads]


    def forward(self, src: torch.Tensor, ) -> torch.Tensor:
        
        B, T, _ = src.shape
        i = torch.arange(T).reshape(-1, 1)
        j = torch.arange(T).reshape(1, -1)
        bias_mask = -torch.abs(i - j).to(src.dtype)
        return self.encoder(src, mask= bias_mask)