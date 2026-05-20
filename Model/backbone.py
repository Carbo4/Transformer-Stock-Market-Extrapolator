import torch
import torch.nn as nn


class Backbone(nn.TransformerEncoder):
    
    def __init__(self, encoder_layer, num_layers, norm = None, enable_nested_tensor = True, 
                 mask_check = True) -> None:
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor, mask_check)
        
        self.n_heads = encoder_layer.self_attn.num_heads
        slopes = -torch.exp2(torch.linspace(0, -8, self.n_heads))[None, :, None, None] # [1, n_heads]
        self.register_buffer('slopes', slopes)
        
    
    def forward(self, src: torch.Tensor, t: torch.Tensor, src_key_padding_mask=None, is_causal=None):
        B, T, _ = src.shape

        dt = torch.log1p(torch.abs(t[:, :, None] - t[:, None, :]))
        bias_mask = (dt.unsqueeze(1) * self.slopes).reshape(B * self.n_heads, T, T).to(src.dtype)
        # print(src.shape, bias_mask.shape)
        return super().forward(src, mask=bias_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)