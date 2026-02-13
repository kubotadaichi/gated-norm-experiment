
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        # Numerical stability handling
        x_norm = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_norm

class PreAffineRMSNorm(nn.Module):
    """
    PreAffineRMSNorm: RMSNorm(lambda_1 * x)
    where lambda_1 is a learnable scaling vector.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # Standard RMSNorm weight
        # Initialize lambda_1 to ones so that it starts as identity
        self.lambda_1 = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Element-wise scaling before normalization
        x_scaled = x * self.lambda_1
        
        # Standard RMSNorm on the scaled input
        norm = x_scaled.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_scaled * torch.rsqrt(norm + self.eps)
        return self.weight * x_norm

class GatedNorm(nn.Module):
    """
    GatedNorm: Element-wise low-rank self-gating mechanism.
    
    The paper describes:
    y = RMSNorm(x)  (or PreAffineRMSNorm output)
    yg = sigmoid(W_up(swish(W_down(y))))
    y' = yg * y
    """
    def __init__(self, d_model: int, rank: int = 16):
        super().__init__()
        # Low-rank projection
        self.w_down = nn.Linear(d_model, rank, bias=False)
        self.w_up = nn.Linear(rank, d_model, bias=False)
        
    def forward(self, x):
        # x is assumed to be the output of a normalization layer
        
        # Compute gating signal
        gate = self.w_down(x)
        gate = F.silu(gate)  # swish is effectively silu
        gate = self.w_up(gate)
        gate = torch.sigmoid(gate)
        
        # Apply gate
        return x * gate

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, norm_type="rmsnorm", use_gated_norm=False):
        super().__init__()
        self.norm_type = norm_type.lower()
        self.use_gated_norm = use_gated_norm
        
        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        
        # Normalization 1 (Pre-Attention)
        self.norm1 = self._get_norm(d_model, self.norm_type)
        if use_gated_norm:
            self.gated_norm1 = GatedNorm(d_model)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Normalization 2 (Pre-FFN)
        self.norm2 = self._get_norm(d_model, self.norm_type)
        if use_gated_norm:
            self.gated_norm2 = GatedNorm(d_model)

    def _get_norm(self, d_model, norm_type):
        if norm_type == "rmsnorm":
            return RMSNorm(d_model)
        elif norm_type == "preaffine":
            return PreAffineRMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(self, x):
        # Pre-Norm architecture
        residual = x
        
        # 1. Norm
        x_norm = self.norm1(x)
        
        # 2. Optional Gating (GatedNorm applies to the output of Norm)
        if self.use_gated_norm:
            x_norm = self.gated_norm1(x_norm)
            
        # 3. Attention
        # nn.MultiheadAttention expects (batch, seq, feature) if batch_first=True
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        
        # 4. Residual Connection
        x = residual + attn_out
        
        residual = x
        
        # 5. Norm
        x_norm = self.norm2(x)
        
        # 6. Optional Gating
        if self.use_gated_norm:
            x_norm = self.gated_norm2(x_norm)
        
        # 7. FFN
        ffn_out = self.ffn(x_norm)
        
        # 8. Residual Connection
        x = residual + ffn_out
        
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, norm_type="rmsnorm", use_gated_norm=False, max_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, norm_type, use_gated_norm)
            for _ in range(n_layers)
        ])
        
        # Final Norm
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        elif norm_type == "preaffine":
            self.norm = PreAffineRMSNorm(d_model)
        else:
             raise ValueError(f"Unknown norm type: {norm_type}")

        self.use_gated_norm = use_gated_norm
        if use_gated_norm:
            self.final_gated_norm = GatedNorm(d_model)
            
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        # x: (batch, seq_len)
        b, t = x.size()
        positions = torch.arange(t, device=x.device).unsqueeze(0)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        if self.use_gated_norm:
            x = self.final_gated_norm(x)
            
        logits = self.head(x)
        return logits
