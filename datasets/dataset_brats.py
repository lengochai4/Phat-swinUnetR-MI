import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage

class BraTS2020_25D(data.Dataset):
    def __init__(self, base_dir, split='train', img_size=224, num_slices=3):
        """
        Args:
            base_dir: Đường dẫn đến thư mục BraTS2020 (chứa MICCAI_BraTS_2020_Data_Training/Validation)
            split: 'train' hoặc 'val'
            img_size: Kích thước resize (224, 224)
            num_slices: Số lát cắt cho mỗi modal (3 cho 2.5D)
        """
        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.num_slices = num_slices
        self.offset = num_slices // 2
        
        # Lấy danh sách ID bệnh nhân
        self.patient_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # Danh sách các lát cắt hợp lệ (bỏ qua các lát cắt đầu và cuối khối volume để tránh lỗi index 2.5D)
        self.all_slices = []
        for patient_id in self.patient_list:
            # BraTS volume thường có 155 lát cắt
            # Ta bỏ qua các lát cắt quá trống ở hai đầu (ví dụ chỉ lấy từ lát 20 đến 130)
            for slice_idx in range(20, 135):
                self.all_slices.append((patient_id, slice_idx))

    def __len__(self):
        return len(self.all_slices)

    def normalize(self, data):
        """Z-score normalization"""
        mean = data.mean()
        std = data.std()
        return (data - mean) / (std + 1e-8)

    def process_label(self, label):
        """
        BraTS labels: 1: NCR/NET, 2: ED, 4: ET, 0: Background
        Chuyển đổi thành 4 kênh (bao gồm background) cho Segmentation task
        """
        h, w = label.shape
        target = np.zeros((4, h, w), dtype=np.float32)
        target[0][label == 0] = 1 # Background
        target[1][label == 1] = 1 # NCR/NET
        target[2][label == 2] = 1 # ED
        target[3][label == 4] = 1 # ET
        return target

    def __getitem__(self, idx):
        patient_id, slice_idx = self.all_slices[idx]
        patient_path = os.path.join(self.base_dir, patient_id)
        
        modals = ['flair', 't1', 't1ce', 't2']
        combined_25d_input = []

        for modal in modals:
            # Load file .nii.gz
            img_path = os.path.join(patient_path, f"{patient_id}_{modal}.nii.gz")
            img_vol = nib.load(img_path).get_fdata()
            
            # Lấy 3 lát cắt liên tiếp (i-1, i, i+1)
            for i in range(slice_idx - self.offset, slice_idx + self.offset + 1):
                slice_data = img_vol[:, :, i]
                # Resize
                slice_data = cv2.resize(slice_data, (self.img_size, self.img_size))
                # Normalize
                slice_data = self.normalize(slice_data)
                combined_25d_input.append(slice_data)

        # Chuyển list thành numpy array [12, 224, 224]
        image_tensor = np.array(combined_25d_input, dtype=np.float32)
        
        # Load Label (chỉ lấy lát cắt trung tâm i)
        label_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")
        label_vol = nib.load(label_path).get_fdata()
        label_slice = label_vol[:, :, slice_idx]
        label_slice = cv2.resize(label_slice, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        target_tensor = self.process_label(label_slice)

        return torch.from_numpy(image_tensor), torch.from_numpy(target_tensor)