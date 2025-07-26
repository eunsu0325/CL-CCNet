# framework/losses.py - 디버그 코드 추가 버전
"""
CoCoNut Loss Functions with Debug

DESIGN PHILOSOPHY:
- SupConLoss 직접 구현
- W2ML 의존성 제거
- 안정적인 contrastive learning
- 🔍 NaN 문제 디버깅 추가
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss with Debug"""
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: [batch_size, n_views, feature_dim]
            labels: [batch_size]
            mask: contrastive mask of shape [batch_size, batch_size]
        Returns:
            loss: scalar
        """
        # 🔍 DEBUG: 입력 데이터 확인
        print(f"🔍 DEBUG: features shape={features.shape}")
        print(f"🔍 DEBUG: features nan={torch.isnan(features).any()}")
        print(f"🔍 DEBUG: features inf={torch.isinf(features).any()}")
        print(f"🔍 DEBUG: features min={features.min():.6f}, max={features.max():.6f}")
        print(f"🔍 DEBUG: temperature={self.temperature}")
        
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 🔍 DEBUG: 라벨과 마스크 확인
        print(f"🔍 DEBUG: labels={labels.flatten().tolist() if labels is not None else None}")
        print(f"🔍 DEBUG: unique labels={torch.unique(labels).tolist() if labels is not None else None}")
        print(f"🔍 DEBUG: mask shape={mask.shape}")
        print(f"🔍 DEBUG: mask sum per row={mask.sum(1).tolist()}")

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # 🔍 DEBUG: dot product 확인
        print(f"🔍 DEBUG: anchor_dot_contrast shape={anchor_dot_contrast.shape}")
        print(f"🔍 DEBUG: anchor_dot_contrast min={anchor_dot_contrast.min():.6f}, max={anchor_dot_contrast.max():.6f}")
        print(f"🔍 DEBUG: anchor_dot_contrast nan={torch.isnan(anchor_dot_contrast).any()}")
        print(f"🔍 DEBUG: anchor_dot_contrast inf={torch.isinf(anchor_dot_contrast).any()}")
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 🔍 DEBUG: numerical stability 후 확인
        print(f"🔍 DEBUG: logits_max={logits_max.flatten()[:5].tolist()}...")
        print(f"🔍 DEBUG: logits min={logits.min():.6f}, max={logits.max():.6f}")

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 🔍 DEBUG: 최종 마스크 확인
        print(f"🔍 DEBUG: final mask shape={mask.shape}")
        print(f"🔍 DEBUG: mask sum per row min={mask.sum(1).min():.6f}")
        print(f"🔍 DEBUG: mask sum per row max={mask.sum(1).max():.6f}")
        print(f"🔍 DEBUG: rows with zero mask={(mask.sum(1) == 0).sum().item()}")

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # 🔍 DEBUG: exp_logits 확인
        print(f"🔍 DEBUG: exp_logits min={exp_logits.min():.6f}, max={exp_logits.max():.6f}")
        print(f"🔍 DEBUG: exp_logits sum per row min={exp_logits.sum(1).min():.6f}")
        print(f"🔍 DEBUG: exp_logits sum per row max={exp_logits.sum(1).max():.6f}")
        print(f"🔍 DEBUG: exp_logits nan={torch.isnan(exp_logits).any()}")
        print(f"🔍 DEBUG: exp_logits inf={torch.isinf(exp_logits).any()}")
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 🔍 DEBUG: log_prob 확인
        print(f"🔍 DEBUG: log_prob min={log_prob.min():.6f}, max={log_prob.max():.6f}")
        print(f"🔍 DEBUG: log_prob nan={torch.isnan(log_prob).any()}")

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        
        # 🔍 DEBUG: 0으로 나누기 방지 확인
        zero_mask_rows = (mask_sum == 0)
        if zero_mask_rows.any():
            print(f"🚨 WARNING: {zero_mask_rows.sum().item()} rows have zero mask sum!")
            print(f"🚨 Zero mask row indices: {torch.where(zero_mask_rows)[0].tolist()}")
        
        # 0으로 나누기 방지
        mask_sum_safe = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum_safe
        
        # 🔍 DEBUG: 최종 계산 확인
        print(f"🔍 DEBUG: mean_log_prob_pos min={mean_log_prob_pos.min():.6f}, max={mean_log_prob_pos.max():.6f}")
        print(f"🔍 DEBUG: mean_log_prob_pos nan={torch.isnan(mean_log_prob_pos).any()}")

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        # 🔍 DEBUG: 최종 손실 확인
        print(f"🔍 DEBUG: final loss={loss.item():.6f}")
        print(f"🔍 DEBUG: final loss nan={torch.isnan(loss).any()}")
        print("="*60)

        return loss


def create_coconut_loss(temperature=0.07):
    """CoCoNut용 손실함수 생성"""
    return SupConLoss(temperature=temperature)


def get_coconut_loss_config():
    """CoCoNut 권장 손실함수 설정"""
    return {
        "continual_loss": {
            "type": "SupConLoss",
            "temperature": 0.07
        },
        "pretrain_loss": {
            "arcface_weight": 0.8,
            "supcon_weight": 0.2,
            "temperature": 0.07
        }
    }