
import torch
import torch.nn as nn

# Mock Mamba class
class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, inference_params=None):
        return self.proj(x)
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None

# Mock RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = None

    def forward(self, x):
        return x * self.weight

def layer_norm_fn(x, weight, bias, eps=1e-5, residual=None, prenorm=False, residual_in_fp32=False):
    return x, residual

def rms_norm_fn(x, weight, bias, eps=1e-5, residual=None, prenorm=False, residual_in_fp32=False):
    return x, residual
