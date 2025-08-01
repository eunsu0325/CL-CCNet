# framework/losses.py - 마할라노비스 제거 버전
"""
Simplified Loss Functions for CoCoNut

DESIGN PHILOSOPHY:
- SupConLoss만 사용
- 마할라노비스 손실 제거
- 단순하고 효과적인 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SimplifiedContrastiveLoss(nn.Module):
    """
    간소화된 대조 학습 손실 (SupCon만 사용)
    
    마할라노비스 제거됨
    """
    
    def __init__(self, config: Dict):
        super(SimplifiedContrastiveLoss, self).__init__()
        
        # SupCon 설정
        self.temperature = config.get('temp', 0.07)
        self.supcon_loss = SupConLoss(temperature=self.temperature)
        
        print(f"[Loss] Simplified Loss initialized:")
        print(f"   SupCon temperature: {self.temperature}")
        print(f"   Mahalanobis: REMOVED")
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, feature_dim] or [batch_size, n_views, feature_dim]
            labels: [batch_size]
        Returns:
            dict with 'total' loss
        """
        # SupCon은 [batch, n_views, dim] 형태 필요
        if len(features.shape) == 2:
            features_supcon = features.unsqueeze(1)
        else:
            features_supcon = features
            
        # SupCon 손실만 계산
        loss_supcon = self.supcon_loss(features_supcon, labels)
        
        return {
            'total': loss_supcon,
            'supcon': loss_supcon
        }


def create_coconut_loss(config: Dict) -> SimplifiedContrastiveLoss:
    """CoCoNut용 간소화된 손실함수 생성"""
    return SimplifiedContrastiveLoss(config)


def test_simplified_loss():
    """간소화된 손실 함수 테스트"""
    print("\n=== Testing Simplified Loss ===")
    
    # 테스트 데이터
    batch_size = 20
    feature_dim = 128
    num_classes = 4
    
    # 각 클래스별로 클러스터 생성
    embeddings = []
    labels = []
    
    for i in range(num_classes):
        # 각 클래스는 다른 중심을 가짐
        center = torch.randn(feature_dim) * 5
        class_samples = center + torch.randn(batch_size // num_classes, feature_dim) * 0.5
        embeddings.append(class_samples)
        labels.extend([i] * (batch_size // num_classes))
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    
    # Loss 계산
    config = {'temp': 0.07}
    loss_fn = create_coconut_loss(config)
    result = loss_fn(embeddings, labels)
    
    print(f"Total Loss: {result['total'].item():.4f}")
    print(f"SupCon Loss: {result['supcon'].item():.4f}")
    
    print("=== Test Complete ===\n")


if __name__ == "__main__":
    test_simplified_loss()