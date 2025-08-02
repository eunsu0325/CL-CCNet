# framework/losses.py - CCNet 스타일 수정 버전
"""
Loss Functions for CoCoNut

DESIGN PHILOSOPHY:
- SupConLoss for CCNet-style multi-view learning
- NaN prevention for stable training
- Clean and simple implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    Modified for CCNet-style training with NaN prevention
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.eps = 1e-8  # NaN 방지용 epsilon

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model with NaN prevention
        
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 입력 검증
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # NaN 체크
        if torch.isnan(features).any():
            print("[Loss] ⚠️ NaN detected in input features!")
            return torch.tensor(0.0, device=device)

        batch_size = features.shape[0]
        
        # 최소 2개 샘플 필요
        if batch_size < 2:
            print("[Loss] ⚠️ Batch size < 2, returning zero loss")
            return torch.tensor(0.0, device=device)
        
        # 마스크 생성
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

        contrast_count = features.shape[1]
        
        # view가 1개만 있는 경우 경고
        if contrast_count < 2:
            print(f"[Loss] ⚠️ Only {contrast_count} view(s) provided, "
                  "SupCon works best with 2+ views")
        
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 특징 정규화 (코사인 유사도를 위해)
        anchor_feature = F.normalize(anchor_feature, dim=1)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        # NaN 방지: 분모가 0이 되는 것 방지
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        # compute mean of log-likelihood over positive
        # NaN 방지: mask.sum(1)이 0이 되는 것 방지
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=self.eps)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 추가 안전장치: 유효한 positive pair가 없는 경우
        valid_mask = mask_sum > self.eps
        if valid_mask.sum() == 0:
            print("[Loss] ⚠️ No valid positive pairs found!")
            return torch.tensor(0.0, device=device)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)
        
        # 유효한 샘플만 평균
        if valid_mask.view(anchor_count, batch_size).sum() > 0:
            loss = loss[valid_mask.view(anchor_count, batch_size)].mean()
        else:
            loss = loss.mean()

        # 최종 NaN 체크
        if torch.isnan(loss):
            print("[Loss] ⚠️ NaN in final loss, returning zero")
            return torch.tensor(0.0, device=device)

        return loss


class SimplifiedContrastiveLoss(nn.Module):
    """
    단순화된 대조 학습 손실 (백업용)
    view 차원 없이 직접 처리
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8
        
    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # L2 정규화
        features = F.normalize(features, dim=1)
        
        # 유사도 행렬
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 라벨 마스크
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)
        
        # 자기 자신 제외 마스크
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # exp 계산
        exp_sim = torch.exp(similarity)
        
        # 분모: 모든 negative + 자기 자신 제외
        denominator = (exp_sim * negative_mask).sum(dim=1, keepdim=True) + self.eps
        
        # 분자: positive pairs
        numerator = exp_sim * positive_mask
        
        # 각 positive pair에 대한 loss
        loss_per_positive = -torch.log(numerator / (denominator + numerator) + self.eps)
        
        # positive pair가 있는 샘플만 평균
        num_positives = positive_mask.sum(dim=1)
        mask = num_positives > 0
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        loss = (loss_per_positive * positive_mask).sum(dim=1)
        loss = loss[mask] / num_positives[mask]
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    CCNet 스타일 대조 학습 손실
    """
    
    def __init__(self, config: Dict):
        super(ContrastiveLoss, self).__init__()
        
        # 설정
        self.temperature = config.get('temp', 0.07)
        self.use_simplified = config.get('use_simplified', False)
        
        # 손실 함수
        if self.use_simplified:
            self.loss_fn = SimplifiedContrastiveLoss(temperature=self.temperature)
            print(f"[Loss] Using Simplified Contrastive Loss")
        else:
            self.loss_fn = SupConLoss(temperature=self.temperature)
            print(f"[Loss] Using SupCon Loss (CCNet style)")
        
        print(f"[Loss] Temperature: {self.temperature}")
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, n_views, feature_dim] or [batch_size, feature_dim]
            labels: [batch_size]
        Returns:
            dict with 'total' and component losses
        """
        # 입력 검증
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"Batch size mismatch: features {features.shape[0]} vs labels {labels.shape[0]}")
        
        # NaN 체크
        if torch.isnan(features).any() or torch.isnan(labels.float()).any():
            print("[Loss] ⚠️ NaN in input, returning zero loss")
            return {
                'total': torch.tensor(0.0, device=features.device),
                'supcon': torch.tensor(0.0, device=features.device)
            }
        
        # 클래스 수 체크
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            print(f"[Loss] ⚠️ Only {len(unique_labels)} class(es) in batch, "
                  "contrastive learning needs 2+")
        
        # SimplifiedLoss는 2D 입력 필요
        if self.use_simplified and len(features.shape) == 3:
            # Multi-view를 flatten
            batch_size, n_views, feature_dim = features.shape
            features = features.view(batch_size * n_views, feature_dim)
            labels = labels.repeat_interleave(n_views)
        
        # SupConLoss는 3D 입력 필요
        elif not self.use_simplified and len(features.shape) == 2:
            features = features.unsqueeze(1)  # [batch, 1, dim]
        
        # 손실 계산
        loss = self.loss_fn(features, labels)
        
        # 결과 반환
        if isinstance(loss, dict):
            return loss
        else:
            return {
                'total': loss,
                'supcon': loss
            }
    
    def debug_batch(self, features: torch.Tensor, labels: torch.Tensor):
        """배치 디버깅 정보 출력"""
        print("\n[Loss Debug]")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")
        
        if len(features.shape) == 3:
            print(f"  Views per sample: {features.shape[1]}")
            print(f"  Feature dim: {features.shape[2]}")
        else:
            print(f"  Feature dim: {features.shape[1]}")
        
        # 각 라벨별 샘플 수
        label_counts = {}
        for label in labels.tolist():
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"  Label distribution: {label_counts}")
        
        # Positive pairs 수 계산
        total_positive_pairs = 0
        for count in label_counts.values():
            if count > 1:
                total_positive_pairs += count * (count - 1) // 2
        print(f"  Total positive pairs: {total_positive_pairs}")


def create_coconut_loss(config: Dict) -> ContrastiveLoss:
    """CoCoNut용 손실함수 생성 (CCNet 스타일)"""
    return ContrastiveLoss(config)

