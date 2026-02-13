
import torch
import torch.nn as nn
from model import RMSNorm, PreAffineRMSNorm, GatedNorm, TransformerBlock

def verify_rms_norm():
    print("Verifying RMSNorm...")
    d_model = 64
    norm = RMSNorm(d_model)
    x = torch.randn(10, 20, d_model)
    exclude_dim = 10
    
    # Manually create an outlier
    x[:, :, exclude_dim] = 1000.0
    
    y = norm(x)
    
    # Check output shape
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    
    # Check if norm is roughly 1 (ignoring epsilon effect)
    y_norm = y.pow(2).mean(dim=-1).sqrt()
    # It won't be exactly 1 due to the learned weight initialized to 1, but close.
    # Actually weight is all 1s, so it should be close to 1.
    assert torch.allclose(y_norm, torch.ones_like(y_norm), atol=1e-3), "RMSNorm output norm is not close to 1"
    
    print("RMSNorm verified.")

def verify_preaffine_rmsnorm():
    print("Verifying PreAffineRMSNorm...")
    d_model = 64
    norm = PreAffineRMSNorm(d_model)
    x = torch.randn(10, 20, d_model)
    
    y = norm(x)
    
    # Check output shape
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    
    # Check parameters
    assert hasattr(norm, 'lambda_1'), "PreAffineRMSNorm missing lambda_1"
    assert norm.lambda_1.shape == (d_model,), "lambda_1 shape mismatch"
    
    print("PreAffineRMSNorm verified.")

def verify_gated_norm():
    print("Verifying GatedNorm...")
    d_model = 64
    # GatedNorm takes the output of a norm layer
    # Its input x is already normalized
    x = torch.randn(10, 20, d_model)
    
    gated_norm = GatedNorm(d_model)
    y = gated_norm(x)
    
    # Check output shape
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"
    
    print("GatedNorm verified.")

def verify_transformer_block():
    print("Verifying TransformerBlock...")
    d_model = 64
    n_head = 4
    bs = 2
    seq_len = 16
    
    x = torch.randn(bs, seq_len, d_model)
    
    # 1. Standard RMSNorm
    print("  - Configuration: RMSNorm")
    block = TransformerBlock(d_model, n_head, norm_type="rmsnorm", use_gated_norm=False)
    y = block(x)
    assert y.shape == x.shape
    
    # 2. PreAffineRMSNorm
    print("  - Configuration: PreAffineRMSNorm")
    block = TransformerBlock(d_model, n_head, norm_type="preaffine", use_gated_norm=False)
    y = block(x)
    assert y.shape == x.shape

    # 3. GatedNorm + RMSNorm
    print("  - Configuration: RMSNorm + GatedNorm")
    block = TransformerBlock(d_model, n_head, norm_type="rmsnorm", use_gated_norm=True)
    y = block(x)
    assert y.shape == x.shape

    # 4. GatedNorm + PreAffineRMSNorm
    print("  - Configuration: PreAffineRMSNorm + GatedNorm")
    block = TransformerBlock(d_model, n_head, norm_type="preaffine", use_gated_norm=True)
    y = block(x)
    assert y.shape == x.shape
    
    # Gradient check
    print("  - Gradient Check")
    loss = y.sum()
    loss.backward()
    
    # Check if gradients exist
    for name, param in block.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        
    print("TransformerBlock verified.")

if __name__ == "__main__":
    verify_rms_norm()
    verify_preaffine_rmsnorm()
    verify_gated_norm()
    verify_transformer_block()
    print("\nAll verifications passed successfully!")
