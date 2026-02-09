import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import các thành phần đã viết
from config import Config
from dataset_brats import BraTS2020_25D
from fat_vision_transformer import SwinUnet
from losses import BraTSMoELoss
from metrics import MetricMonitor, test_single_volume

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    # Sử dụng tqdm để theo dõi tiến độ
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: lấy cả logits và gating_weights từ MoE
        # Lưu ý: Cần đảm bảo model trả về tuple (output, gating_weights)
        outputs, gating_weights = model(images)
        
        # Tính toán loss (Segmentation + Load Balancing)
        loss = criterion(outputs, masks, gating_weights)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    monitor = MetricMonitor()
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs, _ = model(images)
            
            # Chuyển sang sigmoid để tính metric
            preds = torch.sigmoid(outputs)
            
            # Tính Dice và HD95 cho từng batch (đơn giản hóa cho 2D slices)
            # Trong thực tế, bạn có thể loop qua từng ảnh trong batch
            for i in range(images.size(0)):
                res = test_single_volume(preds[i], masks[i])
                monitor.update(res)
                
    return monitor.get_avg()

def main():
    # 1. Khởi tạo cấu hình
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    
    # 2. Setup Dataloader
    train_ds = BraTS2020_25D(cfg.DATA_PATH, split='train', img_size=cfg.IMG_SIZE)
    val_ds = BraTS2020_25D(cfg.DATA_PATH, split='val', img_size=cfg.IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                            num_workers=cfg.NUM_WORKERS)

    # 3. Khởi tạo Model (Swin-Unet MoE)
    # config ở đây là object config của Swin, ta truyền cfg của mình vào
    model = SwinUnet(cfg, img_size=cfg.IMG_SIZE, num_classes=cfg.NUM_CLASSES).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # 5. Loss Function
    criterion = BraTSMoELoss(alpha=cfg.MOE_ALPHA, num_experts=cfg.NUM_EXPERTS)

    # 6. Vòng lặp huấn luyện
    best_dice = 0.0
    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        
        scheduler.step()

        # Log kết quả
        print(f"Loss: {train_loss:.4f} | Mean Dice: {val_metrics['Avg_Dice_Mean']:.4f}")
        print(f"Dice WT: {val_metrics['Avg_Dice_WT']:.4f} | TC: {val_metrics['Avg_Dice_TC']:.4f} | ET: {val_metrics['Avg_Dice_ET']:.4f}")

        # Lưu checkpoint tốt nhất
        if val_metrics['Avg_Dice_Mean'] > best_dice:
            best_dice = val_metrics['Avg_Dice_Mean']
            save_path = os.path.join(cfg.SAVE_CKPT_PATH, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()