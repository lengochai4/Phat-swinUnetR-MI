import torch
import torch.nn as nn
import torch.nn.functional as F

class BraTSMoELoss(nn.Module):
    def __init__(self, alpha=0.01, num_experts=4):
        """
        Args:
            alpha: Hệ số điều tiết cho Balancing Loss (thường từ 0.01 đến 0.1).
            num_experts: Số lượng expert đã cấu hình trong mô hình.
        """
        super(BraTSMoELoss, self).__init__()
        self.alpha = alpha
        self.num_experts = num_experts
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, score, target):
        """Tính toán Soft Dice Loss cho đa nhãn (WT, TC, ET)"""
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def load_balancing_loss(self, gating_weights):
        """
        Tính Load Balancing Loss dựa trên gating_weights thu thập được.
        gating_weights: List các Tensor shape [B, L, num_experts] từ các block.
        """
        if not gating_weights:
            return 0
        
        # Gộp tất cả trọng số từ tất cả các tầng và các token lại
        # all_weights shape: [N_layers * Batch * Tokens, num_experts]
        all_weights = torch.cat([w.view(-1, self.num_experts) for w in gating_weights], dim=0)
        
        # p_i: Xác suất trung bình mỗi expert được chọn trên toàn bộ tập dữ liệu hiện tại
        p = all_weights.mean(0)
        
        # Sử dụng công thức tính hệ số biến thiên (Coefficient of Variation) 
        # hoặc tổng bình phương để ép p tiến tới phân phối đều (1/num_experts)
        loss = self.num_experts * torch.sum(p**2)
        return loss

    def forward(self, inputs, target, gating_weights):
        """
        Args:
            inputs: Logits từ mô hình [B, 3, H, W] (WT, TC, ET)
            target: Ground truth [B, 3, H, W]
            gating_weights: List các trọng số từ Gating network của MoE
        """
        # 1. Segmentation Loss (BCE + Dice)
        # Chuyển logits sang xác suất qua sigmoid cho Dice
        probs = torch.sigmoid(inputs)
        
        seg_bce = self.bce(inputs, target)
        seg_dice = self.dice_loss(probs, target)
        
        loss_seg = seg_bce + seg_dice
        
        # 2. MoE Balancing Loss
        loss_balance = self.load_balancing_loss(gating_weights)
        
        # 3. Total Loss: Loss_total = Loss_seg + alpha * Loss_balancing
        total_loss = loss_seg + self.alpha * loss_balance
        
        return total_loss, loss_seg, loss_balance

# Ví dụ sử dụng trong vòng lặp training:
# criterion = BraTSMoELoss(alpha=0.01, num_experts=4)
# logits, weights = model(images)
# loss, seg_l, bal_l = criterion(logits, masks, weights)