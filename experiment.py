
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from model import MiniGPT, RMSNorm, PreAffineRMSNorm
import pandas as pd
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

import argparse

# Default Configuration
DEFAULT_CONFIG = {
    "d_model": 256,
    "n_head": 4,
    "n_layers": 4,
    "vocab_size": 50257, # GPT-2 vocab size
    "max_len": 128,
    "batch_size": 32,
    "lr": 5e-4,
    "epochs": 1, # Keep it short for demo
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_interval": 10,
    "eval_interval": 100,
}

def get_data(config=None):
    if config is None:
        config = DEFAULT_CONFIG
        
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config["max_len"], padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format("torch")
    
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def train_model(config_name, norm_type, use_gated_norm):
    print(f"Starting training for {config_name}: norm_type={norm_type}, gated={use_gated_norm}")
    
    train_loader, val_loader = get_data(CONFIG)
    
    model = MiniGPT(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        n_head=CONFIG["n_head"],
        n_layers=CONFIG["n_layers"],
        norm_type=norm_type,
        use_gated_norm=use_gated_norm,
        max_len=CONFIG["max_len"]
    ).to(CONFIG["device"])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    logs = []
    step = 0
    start_time = time.time()
    
    model.train()
    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(train_loader, desc=f"{config_name} Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(CONFIG["device"])
            
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % CONFIG["log_interval"] == 0:
                logs.append({
                    "config": config_name,
                    "step": step,
                    "loss": loss.item(),
                    "time": time.time() - start_time
                })
                progress_bar.set_postfix({"loss": loss.item()})
                
            if step % CONFIG["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, CONFIG["device"])
                logs.append({
                    "config": config_name,
                    "step": step,
                    "val_loss": val_loss,
                    "time": time.time() - start_time
                })
                model.train()
                print(f"Step {step}: Val Loss = {val_loss:.4f}")
        
        # Save Model at end of epoch
        save_path = os.path.join("results", f"model_{config_name.lower()}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path} (Epoch {epoch+1})")
        
        # Upload to HF Hub if requested (at end of each epoch)
        if CONFIG.get("push_to_hub") and CONFIG.get("hf_repo_id"):
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not token:
                print("Warning: HF_TOKEN not found in environment variables. Skipping upload.")
            else:
                print(f"Uploading {config_name} model to Hugging Face Hub: {CONFIG['hf_repo_id']}")
                api = HfApi(token=token)
                try:
                    api.upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo=f"model_{config_name.lower()}.pt",
                        repo_id=CONFIG["hf_repo_id"],
                        repo_type="model"
                    )
                    print("Upload successful.")
                except Exception as e:
                    print(f"Upload failed: {e}")

    print(f"Finished {config_name}")
    
    # Save Model
    return logs
    
def analyze_weights(config_name):
    print(f"Analyzing weights for {config_name}...")
    model_path = os.path.join("results", f"model_{config_name.lower()}.pt")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    # Re-instantiate model structure to load weights
    # We need to know the config used.
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
        return

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
        return

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT with different norms.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"], help="Learning rate")
    parser.add_argument("--d_model", type=int, default=DEFAULT_CONFIG["d_model"], help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_CONFIG["n_layers"], help="Number of layers")
    parser.add_argument("--n_head", type=int, default=DEFAULT_CONFIG["n_head"], help="Number of heads")
    parser.add_argument("--config_name", type=str, default="all", help="Config to run (Baseline, GatedNorm, PreAffine, or all)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model weights to Hugging Face Hub")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Hugging Face Repo ID (e.g., username/repo_name)")
    
    args = parser.parse_args()
    
    # Update CONFIG
    global CONFIG
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG.update({
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_head": args.n_head,
        "push_to_hub": args.push_to_hub,
        "hf_repo_id": args.hf_repo_id
    })
    
    print(f"Configuration: {CONFIG}")

    os.makedirs("results", exist_ok=True)
    
    configs_to_run = []
    if args.config_name == "all":
        configs_to_run = ["Baseline", "GatedNorm", "PreAffine"]
    else:
        configs_to_run = [args.config_name]
        
    for cfg in configs_to_run:
        if cfg == "Baseline":
            logs = train_model("Baseline", norm_type="rmsnorm", use_gated_norm=False)
            pd.DataFrame(logs).to_csv("results/baseline.csv", index=False)
            analyze_weights("Baseline")
        elif cfg == "GatedNorm":
            logs = train_model("GatedNorm", norm_type="rmsnorm", use_gated_norm=True)
            pd.DataFrame(logs).to_csv("results/gatednorm.csv", index=False)
            analyze_weights("GatedNorm")
        elif cfg == "PreAffine":
            logs = train_model("PreAffine", norm_type="preaffine", use_gated_norm=False)
            pd.DataFrame(logs).to_csv("results/preaffine.csv", index=False)
            analyze_weights("PreAffine")

    print("Experiments finished.")
