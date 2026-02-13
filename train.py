
import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerBlock

def train():
    # Configuration
    d_model = 128
    n_head = 8
    n_layers = 4
    batch_size = 32
    seq_len = 64
    lr = 1e-3
    n_steps = 100
    
    # Select architecture: 'rmsnorm' (baseline), 'preaffine', 'rmsnorm_gated', 'preaffine_gated'
    # Here we demo the most advanced one: PreAffineRMSNorm + GatedNorm
    norm_type = "preaffine"
    use_gated_norm = True
    
    print(f"Training with norm_type={norm_type}, use_gated_norm={use_gated_norm}")
    
    # Model
    model = nn.Sequential(*[
        TransformerBlock(d_model, n_head, norm_type=norm_type, use_gated_norm=use_gated_norm)
        for _ in range(n_layers)
    ])
    
    # Dummy data
    # (batch, seq, d_model) -> we are just learning identity mapping or noise for demo
    x = torch.randn(batch_size, seq_len, d_model)
    target = torch.randn(batch_size, seq_len, d_model)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            
    print("Training finished.")

if __name__ == "__main__":
    train()
