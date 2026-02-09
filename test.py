import os
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2

from config import Config
from fat_vision_transformer import SwinUnet
from metrics import test_single_volume, MetricMonitor

def test():
    # 1. Cấu hình & Thiết bị
    device = torch.device(Config.DEVICE)
    
    # Giả lập object config phù hợp với yêu cầu của SwinUnet class
    class SwinConfig:
        def __init__(self):
            self.DATA = type('obj', (), {'IMG_SIZE': Config.IMG_SIZE})
            self.MODEL = type('obj', (), {
                'SWIN': type('obj', (), {
                    'PATCH_SIZE': Config.PATCH_SIZE,
                    'IN_CHANS': Config.IN_CHANNELS,
                    'EMBED_DIM': Config.EMBED_DIM,
                    'DEPTHS': Config.DEPTHS,
                    'NUM_HEADS': Config.NUM_HEADS,
                    'WINDOW_SIZE': Config.WINDOW_SIZE,
                    'MLP_RATIO': Config.MOE_MLP_RATIO,
                    'QKV_BIAS': True,
                    'QK_SCALE': None,
                    'APE': False,
                    'PATCH_NORM': True
                }),
                'DROP_RATE': 0.0,
                'DROP_PATH_RATE': 0.1
            })
            self.TRAIN = type('obj', (), {'USE_CHECKPOINT': False})

    # 2. Khởi tạo Model & Load Checkpoint
    model_config = SwinConfig()
    model = SwinUnet(model_config, img_size=Config.IMG_SIZE, num_classes=Config.NUM_CLASSES).to(device)
    
    checkpoint_path = os.path.join(Config.SAVE_CKPT_PATH, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"--- Loaded checkpoint from {checkpoint_path} ---")
    else:
        print("--- No checkpoint found! Testing with random weights ---")

    model.eval()
    monitor = MetricMonitor()

    # 3. Lấy danh sách bệnh nhân từ file validation list
    with open(Config.VAL_LIST, 'r') as f:
        val_patients = [line.strip() for line in f.readlines()]

    print(f"Starting Evaluation on {len(val_patients)} cases...")

    with torch.no_grad():
        for patient_id in tqdm(val_patients):
            patient_path = os.path.join(Config.DATA_PATH, patient_id)
            
            # Load Label gốc để so sánh (Ground Truth)
            label_nii = nib.load(os.path.join(patient_path, f"{patient_id}_seg.nii.gz"))
            label_vol = label_nii.get_fdata() # [H, W, D] (thường là 240, 240, 155)
            
            # Tạo mảng trống để chứa dự đoán [3, H, W, D]
            full_pred = np.zeros((3, *label_vol.shape))
            
            # Load các modal
            modals = ['flair', 't1', 't1ce', 't2']
            modal_data = {}
            for m in modals:
                modal_data[m] = nib.load(os.path.join(patient_path, f"{patient_id}_{m}.nii.gz")).get_fdata()

            # Sliding Window theo trục Z (lát cắt)
            # Bỏ qua lát đầu/cuối do logic 2.5D (cần i-1 và i+1)
            for i in range(1, label_vol.shape[2] - 1):
                input_slices = []
                for m in modals:
                    for offset in [-1, 0, 1]:
                        slc = modal_data[m][:, :, i + offset]
                        slc = cv2.resize(slc, (Config.IMG_SIZE, Config.IMG_SIZE))
                        # Normalize (Z-score)
                        slc = (slc - slc.mean()) / (slc.std() + 1e-8)
                        input_slices.append(slc)
                
                # Biến thành tensor [1, 12, 224, 224]
                input_tensor = torch.from_numpy(np.array(input_slices)).float().unsqueeze(0).to(device)
                
                # Predict
                outputs, _ = model(input_tensor) # outputs: [1, 3, 224, 224]
                outputs = torch.sigmoid(outputs).squeeze(0).cpu().numpy()
                
                # Resize ngược lại kích thước gốc và đưa vào full_pred
                for cls_idx in range(3):
                    pred_resized = cv2.resize(outputs[cls_idx], (label_vol.shape[1], label_vol.shape[0]))
                    full_pred[cls_idx, :, :, i] = pred_resized

            # 4. Hậu xử lý Label (Convert GT sang 3 vùng WT, TC, ET)
            # Theo chuẩn BraTS: WT = labels 1,2,4; TC = 1,4; ET = 4
            gt_wt = (label_vol > 0).astype(np.float32)
            gt_tc = np.logical_or(label_vol == 1, label_vol == 4).astype(np.float32)
            gt_et = (label_vol == 4).astype(np.float32)
            gt_combined = np.stack([gt_wt, gt_tc, gt_et], axis=0)

            # Tính metric cho từng case (trung bình các lát cắt)
            # Do test_single_volume trong file metrics.py nhận [3, H, W], ta thực hiện loop qua các slice
            case_metrics = {'Dice_WT': [], 'Dice_TC': [], 'Dice_ET': [], 'HD95_WT': [], 'HD95_TC': [], 'HD95_ET': []}
            
            # Chỉ tính trên các slice có chứa u để tránh loãng kết quả (hoặc tính toàn bộ tùy nhu cầu)
            for i in range(full_pred.shape[3]):
                if gt_combined[:,:,:,i].sum() > 0:
                    res = test_single_volume(torch.from_numpy(full_pred[:,:,:,i]), torch.from_numpy(gt_combined[:,:,:,i]))
                    for k, v in res.items():
                        case_metrics[k].append(v)
            
            # Update vào monitor tổng
            avg_case = {k: np.mean(v) if len(v)>0 else 0 for k, v in case_metrics.items()}
            monitor.update(avg_case)

    # 5. In kết quả cuối cùng
    final_results = monitor.get_avg()
    print("\n" + "="*30)
    print("FINAL VALIDATION RESULTS")
    print(f"Dice Whole Tumor (WT): {final_results['Avg_Dice_WT']:.4f}")
    print(f"Dice Tumor Core (TC):  {final_results['Avg_Dice_TC']:.4f}")
    print(f"Dice Enhancing Tumor (ET): {final_results['Avg_Dice_ET']:.4f}")
    print(f"Mean Dice Score: {final_results['Avg_Dice_Mean']:.4f}")
    print(f"Mean HD95: {final_results['Avg_HD95_Mean']:.4f}")
    print("="*30)

if __name__ == '__main__':
    test()