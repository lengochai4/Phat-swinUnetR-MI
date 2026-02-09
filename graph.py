import os
import pandas as pd
import matplotlib.pyplot as plt
from config import Config  # Import Config từ file của bạn

def plot_training_results(log_filename='training_log.csv'):
    """
    Hàm vẽ đồ thị dựa trên file log được lưu trong Config.LOG_PATH
    """
    # ===============================
    # Cấu hình Style
    # ===============================
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0
    })

    # Đường dẫn file log từ Config
    log_path = os.path.join(Config.LOG_PATH, log_filename)
    
    if not os.path.exists(log_path):
        print(f"Error: Không tìm thấy file log tại {log_path}")
        return

    df = pd.read_csv(log_path)
    epochs = df["epoch"]

    # ===============================
    # Đồ thị 1: Loss & Load Balancing Loss (Đặc thù MoE)
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], label="Total Train Loss", color='blue')
    plt.plot(epochs, df["val_loss"], label="Total Val Loss", color='red', linestyle='--')
    
    # Nếu bạn có lưu lại Load Balancing Loss của MoE
    if "aux_loss" in df.columns:
        plt.plot(epochs, df["aux_loss"], label="MoE Aux Loss", color='green', alpha=0.7)
        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Swin-MoE Training Progress: Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_PATH, "loss_plot.png"))
    plt.show()

    # ===============================
    # Đồ thị 2: BraTS Dice Scores (WT, TC, ET)
    # ===============================
    plt.figure(figsize=(8, 5))
    # Vẽ các chỉ số Dice trung bình từ MetricMonitor
    if "val_dice_wt" in df.columns:
        plt.plot(epochs, df["val_dice_wt"], label="Dice WT (Whole Tumor)", marker='o', markersize=4)
        plt.plot(epochs, df["val_dice_tc"], label="Dice TC (Tumor Core)", marker='s', markersize=4)
        plt.plot(epochs, df["val_dice_et"], label="Dice ET (Enhancing Tumor)", marker='^', markersize=4)
    
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Metrics per Region (BraTS)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.LOG_PATH, "dice_metrics_plot.png"))
    plt.show()

    # ===============================
    # Đồ thị 3: Hausdorff Distance (HD95)
    # ===============================
    if "val_hd95" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, df["val_hd95"], label="Avg HD95", color='purple')
        plt.xlabel("Epoch")
        plt.ylabel("Distance (mm)")
        plt.title("Average Hausdorff Distance (HD95)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.LOG_PATH, "hd95_plot.png"))
        plt.show()

# Gọi hàm để chạy
if __name__ == "__main__":
    plot_training_results()