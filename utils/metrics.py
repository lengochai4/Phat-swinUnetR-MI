import numpy as np
import torch
from medpy import metric

def calculate_metric_per_case(pred, gt):
    """
    Tính Dice và Hausdorff Distance cho một case (cặp pred/gt)
    pred, gt: numpy array [H, W]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        # Dự đoán có vùng u nhưng thực tế không có
        return 0, 0 
    else:
        # Cả hai đều không có vùng u (thường hiếm gặp trong vùng u não đã crop)
        return 1, 0

def test_single_volume(prediction, target):
    """
    Tính toán metric cho cả 3 vùng của BraTS
    prediction, target: [3, H, W] (Kênh 0: WT, 1: TC, 2: ET)
    """
    prediction = prediction.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    
    # Chuyển đổi sang nhị phân (threshold 0.5)
    prediction = (prediction > 0.5).astype(np.float32)
    
    results = {
        'Dice_WT': 0, 'Dice_TC': 0, 'Dice_ET': 0,
        'HD95_WT': 0, 'HD95_TC': 0, 'HD95_ET': 0
    }
    
    # Tính cho từng vùng
    results['Dice_WT'], results['HD95_WT'] = calculate_metric_per_case(prediction[0], target[0])
    results['Dice_TC'], results['HD95_TC'] = calculate_metric_per_case(prediction[1], target[1])
    results['Dice_ET'], results['HD95_ET'] = calculate_metric_per_case(prediction[2], target[2])
    
    return results

class MetricMonitor:
    """Class hỗ trợ lưu trữ và tính trung bình metric trong quá trình huấn luyện"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_wt = []
        self.dice_tc = []
        self.dice_et = []
        self.hd_wt = []
        self.hd_tc = []
        self.hd_et = []

    def update(self, results):
        self.dice_wt.append(results['Dice_WT'])
        self.dice_tc.append(results['Dice_TC'])
        self.dice_et.append(results['Dice_ET'])
        self.hd_wt.append(results['HD95_WT'])
        self.hd_tc.append(results['HD95_TC'])
        self.hd_et.append(results['HD95_ET'])

    def get_avg(self):
        return {
            'Avg_Dice_WT': np.mean(self.dice_wt),
            'Avg_Dice_TC': np.mean(self.dice_tc),
            'Avg_Dice_ET': np.mean(self.dice_et),
            'Avg_Dice_Mean': (np.mean(self.dice_wt) + np.mean(self.dice_tc) + np.mean(self.dice_et)) / 3,
            'Avg_HD95_Mean': (np.mean(self.hd_wt) + np.mean(self.hd_tc) + np.mean(self.hd_et)) / 3
        }