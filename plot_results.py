
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_results():
    results_dir = "results"
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    if not csv_files:
        print("No results found.")
        return

    plt.figure(figsize=(12, 5))
    
    # 1. Training Loss
    plt.subplot(1, 2, 1)
    for csv_file in csv_files:
        name = os.path.basename(csv_file).replace(".csv", "")
        df = pd.read_csv(csv_file)
        # Filter out validation rows
        train_df = df[df["val_loss"].isna()]
        plt.plot(train_df["step"], train_df["loss"], label=name, alpha=0.7)
    
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Validation Loss
    plt.subplot(1, 2, 2)
    for csv_file in csv_files:
        name = os.path.basename(csv_file).replace(".csv", "")
        df = pd.read_csv(csv_file)
        # Filter for validation rows
        val_df = df[df["val_loss"].notna()]
        if not val_df.empty:
            plt.plot(val_df["step"], val_df["val_loss"], label=name, marker="o")
    
    plt.title("Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiment_results.png")
    print("Plot saved to experiment_results.png")

if __name__ == "__main__":
    plot_results()
