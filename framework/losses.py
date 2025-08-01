import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss
    
    Adapted from: https://github.com/HobbitLong/SupContrast
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, dim] or [bsz, dim]
            labels: ground truth of shape [bsz]
            mask: contrastive mask of shape [bsz, bsz], optional
        Returns:
            A scalar loss
        """
        device = features.device
        
        # Handle both [bsz, dim] and [bsz, n_views, dim] formats
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_labels = torch.cat([labels for _ in range(contrast_count)], dim=0)
        
        # Compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    단순화된 Combined Loss - SupConLoss만 사용
    (마할라노비스 관련 기능 제거)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # SupCon Loss
        self.temperature = config.get('temp', 0.07)
        self.supcon = SupConLoss(temperature=self.temperature)
        
        # 가중치 (이제 supcon만 사용하므로 실제로는 무시됨)
        self.supcon_weight = config.get('online_learning', {}).get('supcon_weight', 1.0)
        
        print(f"[Loss] SupConLoss initialized (temp={self.temperature})")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                epoch: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Ground truth labels [batch_size]
            epoch: Current epoch (unused now)
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary of individual losses
        """
        # L2 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # SupCon Loss만 계산
        supcon_loss = self.supcon(embeddings, labels)
        
        # 전체 손실 (이제는 SupCon만)
        total_loss = supcon_loss * self.supcon_weight
        
        # 손실 딕셔너리
        loss_dict = {
            'total': total_loss.item(),
            'supcon': supcon_loss.item()
        }
        
        return total_loss, loss_dict
    
    def get_loss_weights(self) -> dict:
        """현재 손실 가중치 반환"""
        return {
            'supcon': self.supcon_weight
        }


def get_loss(config: dict) -> nn.Module:
    """설정에 따른 손실 함수 반환"""
    loss_type = config.get('type', 'SupConLoss')
    
    if loss_type == 'SupConLoss':
        return SupConLoss(temperature=config.get('temp', 0.07))
    elif loss_type == 'CombinedLoss':
        return CombinedLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")