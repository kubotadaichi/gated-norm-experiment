
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from model import MiniGPT, RMSNorm, PreAffineRMSNorm
from experiment import get_data, evaluate, DEFAULT_CONFIG as CONFIG
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_model_from_hub(repo_id, filename, config_name, device):
    print(f"Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

    print(f"Loading {config_name}...")
    # Determine config
    if config_name == "Baseline":
        norm_type = "rmsnorm"
        use_gated_norm = False
    elif config_name == "GatedNorm":
        norm_type = "rmsnorm"
        use_gated_norm = True
    elif config_name == "PreAffine":
        norm_type = "preaffine"
        use_gated_norm = False
    else:
        raise ValueError(f"Unknown config: {config_name}")

    model = MiniGPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        n_layers=CONFIG["n_layers"],
        norm_type=norm_type,
        use_gated_norm=use_gated_norm,
        max_len=CONFIG["max_len"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    return model

def analyze_weights(model, config_name):
    # Collect weights
    rms_weights = []
    preaffine_lambdas = []
    
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm) or isinstance(module, PreAffineRMSNorm):
            if hasattr(module, "weight"): # The scale parameters gamma
                rms_weights.append(module.weight.detach().cpu().numpy())
            
        if isinstance(module, PreAffineRMSNorm):
            if hasattr(module, "lambda_1"):
                preaffine_lambdas.append(module.lambda_1.detach().cpu().numpy())
                
    if not rms_weights:
        print("No RMSNorm weights found.")
        return

    os.makedirs("results", exist_ok=True)

    # Plot RMSNorm Weights
    plt.figure(figsize=(10, 6))
    for i, w in enumerate(rms_weights):
        # Sort values to see distribution
        sorted_w = np.sort(np.abs(w))
        plt.plot(sorted_w, label=f"Layer/Norm {i}")
        
    plt.title(f"{config_name}: RMSNorm Weight Distribution (Sorted)")
    plt.xlabel("Dimension Index (Sorted)")
    plt.ylabel("Weight Magnitude")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"results/weights_{config_name.lower()}_rms.png")
    plt.close()
    print(f"Saved RMSNorm weight plot to results/weights_{config_name.lower()}_rms.png")
    
    # Plot PreAffine Lambdas if exist
    if preaffine_lambdas:
        plt.figure(figsize=(10, 6))
        for i, w in enumerate(preaffine_lambdas):
            sorted_w = np.sort(np.abs(w))
            plt.plot(sorted_w, label=f"Layer/Norm {i}")
            
        plt.title(f"{config_name}: PreAffine Lambda Distribution (Sorted)")
        plt.xlabel("Dimension Index (Sorted)")
        plt.ylabel("Lambda Magnitude")
        plt.yscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(f"results/weights_{config_name.lower()}_lambda.png")
        plt.close()
        print(f"Saved PreAffine lambda plot to results/weights_{config_name.lower()}_lambda.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="daichi202/gated-norm-test", help="HF Repo ID")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data for evaluation
    _, val_loader = get_data(CONFIG)

    configs = [
        ("Baseline", "model_baseline.pt"),
        ("GatedNorm", "model_gatednorm.pt"),
        ("PreAffine", "model_preaffine.pt")
    ]

    results = []

    for config_name, filename in configs:
        model = load_model_from_hub(args.repo_id, filename, config_name, device)
        if model:
            # Evaluate
            print(f"Evaluating {config_name}...")
            val_loss = evaluate(model, val_loader, device)
            print(f"{config_name} Val Loss: {val_loss:.4f}")
            results.append({"config": config_name, "val_loss": val_loss})
            
            # Analyze Weights
            analyze_weights(model, config_name)
        else:
            print(f"Skipping {config_name} due to download failure.")

    print("\n=== Final Evaluation Results ===")
    for res in results:
        print(f"{res['config']}: {res['val_loss']:.4f}")
