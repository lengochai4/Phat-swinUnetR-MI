import os

class Config:
    # 1. Path Settings
    # Đường dẫn đến thư mục chứa dữ liệu BraTS 2020 (chứa các folder Patient_ID)
    DATA_PATH = './data/MICCAI_BraTS2020_TrainingData'
    # File txt chứa danh sách ID bệnh nhân để train/val
    TRAIN_LIST = './datasets/train.txt'
    VAL_LIST = './datasets/val.txt'
    
    # Nơi lưu checkpoint và logs
    OUTPUT_DIR = './output'
    SAVE_CKPT_PATH = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_PATH = os.path.join(OUTPUT_DIR, 'logs')

    # 2. Model Hyperparameters (Dựa trên Swin-UNet)
    IMG_SIZE = 224
    IN_CHANNELS = 12       # 4 modals * 3 slices (2.5D logic)
    NUM_CLASSES = 3        # WT, TC, ET (Theo logic file metrics/losses đã viết)
    
    # MoE Settings
    NUM_EXPERTS = 4        # Số lượng chuyên gia trong mỗi block MoE
    MOE_MLP_RATIO = 4      # Tỉ lệ mở rộng hidden dim trong MLP
    MOE_ALPHA = 0.01       # Trọng số cho Load Balancing Loss

    # Swin Backbone Settings (Tiny version là phổ biến nhất)
    PATCH_SIZE = 4
    WINDOW_SIZE = 7
    EMBED_DIM = 96
    DEPTHS = [2, 2, 2, 2]
    NUM_HEADS = [3, 6, 12, 24]

    # 3. Training Settings
    BATCH_SIZE = 24        # Điều chỉnh tùy theo VRAM (24-32 cho 2.5D thường ổn trên 24GB VRAM)
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # 4. Hardware Settings
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 8        # Số luồng CPU để load data
    PIN_MEMORY = True

    @staticmethod
    def display():
        """Hiển thị các thông số cấu hình"""
        print("\n--- Experiment Configuration ---")
        for key, value in Config.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("--------------------------------\n")

# Tạo các thư mục cần thiết nếu chưa có
os.makedirs(Config.SAVE_CKPT_PATH, exist_ok=True)
os.makedirs(Config.LOG_PATH, exist_ok=True)