# framework/losses.py - Mahalanobis Loss 추가 버전
"""
CoCoNut Loss Functions with Mahalanobis

DESIGN PHILOSOPHY:
- SupConLoss 직접 구현
- Mahalanobis Loss for tighter clusters
- 교대 학습 지원
- ON/OFF 가능한 모듈형 설계
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    
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
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        mask_sum_safe = torch.clamp(mask_sum, min=1e-6)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum_safe

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MahalanobisLoss(nn.Module):
    """
    Mahalanobis 거리 기반 손실 함수
    
    각 클래스의 샘플들이 tight한 cluster를 형성하도록 유도
    - Full covariance: 학습 시 정교한 분포 모델링
    - Diagonal covariance: 빠른 계산 옵션
    """
    
    def __init__(self, use_full_covariance=True, regularization=1e-6):
        super(MahalanobisLoss, self).__init__()
        self.use_full_covariance = use_full_covariance
        self.regularization = regularization
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                return_stats: bool = False) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, feature_dim]
            labels: [batch_size]
            return_stats: 통계 정보도 반환할지 여부
        Returns:
            loss: scalar (또는 dict if return_stats=True)
        """
        device = embeddings.device
        unique_labels = torch.unique(labels)
        
        total_loss = 0.0
        class_losses = []
        class_stats = {}
        
        for label in unique_labels:
            # 같은 클래스의 샘플들만 선택
            mask = labels == label
            class_embeddings = embeddings[mask]
            
            if class_embeddings.size(0) < 2:
                # 샘플이 1개면 스킵
                continue
                
            # 평균과 중심화
            mean = class_embeddings.mean(dim=0, keepdim=True)
            centered = class_embeddings - mean
            
            if self.use_full_covariance:
                # Full covariance matrix
                cov = (centered.T @ centered) / (centered.size(0) - 1)
                # 정규화 (numerical stability)
                cov += torch.eye(cov.size(0), device=device) * self.regularization
                
                try:
                    # Cholesky 분해를 통한 안정적인 역행렬 계산
                    L = torch.linalg.cholesky(cov)
                    # Mahalanobis 거리: (x-μ)ᵀ Σ⁻¹ (x-μ)
                    # Cholesky를 사용하면: ||L⁻¹(x-μ)||²
                    z = torch.triangular_solve(centered.T, L, upper=False)[0].T
                    mahal_dists = torch.sum(z * z, dim=1)
                    
                except RuntimeError:
                    # Cholesky 실패 시 diagonal fallback
                    var = centered.var(dim=0, unbiased=True) + self.regularization
                    mahal_dists = torch.sum(centered**2 / var, dim=1)
                    
            else:
                # Diagonal covariance (빠른 계산)
                var = centered.var(dim=0, unbiased=True) + self.regularization
                mahal_dists = torch.sum(centered**2 / var, dim=1)
            
            # 손실: 클래스 내 Mahalanobis 거리의 평균
            # sqrt를 취해서 거리로 변환
            class_loss = torch.sqrt(mahal_dists + 1e-6).mean()
            class_losses.append(class_loss)
            total_loss += class_loss
            
            # 통계 정보 저장
            if return_stats:
                class_stats[label.item()] = {
                    'mean_distance': class_loss.item(),
                    'sample_count': class_embeddings.size(0),
                    'mean_norm': mean.norm().item()
                }
        
        # 평균 손실
        if len(class_losses) > 0:
            loss = total_loss / len(class_losses)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if return_stats:
            return {
                'loss': loss,
                'class_stats': class_stats,
                'num_classes': len(class_losses)
            }
        
        return loss


class CombinedContrastiveLoss(nn.Module):
    """
    SupCon + Mahalanobis 결합 손실
    
    교대 학습 및 가중치 조절 지원
    """
    
    def __init__(self, config: Dict):
        super(CombinedContrastiveLoss, self).__init__()
        
        # SupCon 설정
        self.temperature = config.get('temp', 0.07)
        self.supcon_loss = SupConLoss(temperature=self.temperature)
        
        # Mahalanobis 설정
        online_config = config.get('online_learning', {})
        self.enable_mahalanobis = online_config.get('enable_mahalanobis', True)
        self.supcon_weight = online_config.get('supcon_weight', 1.0)
        self.mahal_weight = online_config.get('mahal_weight', 0.2)
        self.alternate_training = online_config.get('alternate_training', True)
        
        if self.enable_mahalanobis:
            self.mahal_loss = MahalanobisLoss(use_full_covariance=True)
        else:
            self.mahal_loss = None
            
        print(f"[Loss] Combined Loss initialized:")
        print(f"   SupCon weight: {self.supcon_weight}")
        print(f"   Mahalanobis: {'ON' if self.enable_mahalanobis else 'OFF'}")
        if self.enable_mahalanobis:
            print(f"   Mahal weight: {self.mahal_weight}")
            print(f"   Alternate training: {self.alternate_training}")
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, 
                phase: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch_size, feature_dim] or [batch_size, n_views, feature_dim]
            labels: [batch_size]
            phase: 'supcon', 'mahal', or None (both)
        Returns:
            dict with 'total', 'supcon', 'mahal' losses
        """
        # SupCon은 [batch, n_views, dim] 형태 필요
        if len(features.shape) == 2:
            features_supcon = features.unsqueeze(1)
        else:
            features_supcon = features
            
        # 교대 학습 모드
        if self.alternate_training and phase is not None:
            if phase == 'supcon':
                loss_supcon = self.supcon_loss(features_supcon, labels)
                return {
                    'total': loss_supcon * self.supcon_weight,
                    'supcon': loss_supcon,
                    'mahal': torch.tensor(0.0, device=features.device)
                }
            elif phase == 'mahal' and self.enable_mahalanobis:
                features_flat = features_supcon.squeeze(1) if len(features_supcon.shape) == 3 else features
                loss_mahal = self.mahal_loss(features_flat, labels)
                return {
                    'total': loss_mahal * self.mahal_weight,
                    'supcon': torch.tensor(0.0, device=features.device),
                    'mahal': loss_mahal
                }
        
        # 동시 학습 모드 (phase=None)
        loss_supcon = self.supcon_loss(features_supcon, labels)
        
        if self.enable_mahalanobis:
            features_flat = features_supcon.squeeze(1) if len(features_supcon.shape) == 3 else features
            loss_mahal = self.mahal_loss(features_flat, labels)
            total_loss = self.supcon_weight * loss_supcon + self.mahal_weight * loss_mahal
        else:
            loss_mahal = torch.tensor(0.0, device=features.device)
            total_loss = self.supcon_weight * loss_supcon
        
        return {
            'total': total_loss,
            'supcon': loss_supcon,
            'mahal': loss_mahal
        }
    
    def set_phase(self, phase: str):
        """학습 단계 설정 (교대 학습용)"""
        self.current_phase = phase
        
    def disable_mahalanobis(self):
        """Mahalanobis 손실 비활성화 (ablation study용)"""
        self.enable_mahalanobis = False
        print("[Loss] Mahalanobis loss DISABLED")
        
    def enable_mahalanobis(self):
        """Mahalanobis 손실 활성화"""
        if self.mahal_loss is not None:
            self.enable_mahalanobis = True
            print("[Loss] Mahalanobis loss ENABLED")


def create_coconut_loss(config: Dict) -> CombinedContrastiveLoss:
    """CoCoNut용 손실함수 생성 헬퍼"""
    return CombinedContrastiveLoss(config)


def test_mahalanobis_loss():
    """Mahalanobis Loss 테스트"""
    print("\n=== Testing Mahalanobis Loss ===")
    
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
    loss_fn = MahalanobisLoss(use_full_covariance=True)
    result = loss_fn(embeddings, labels, return_stats=True)
    
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Number of classes: {result['num_classes']}")
    print("\nPer-class statistics:")
    for class_id, stats in result['class_stats'].items():
        print(f"  Class {class_id}: mean_dist={stats['mean_distance']:.4f}, "
              f"samples={stats['sample_count']}")
    
    # Diagonal vs Full 비교
    loss_diag = MahalanobisLoss(use_full_covariance=False)
    loss_diag_val = loss_diag(embeddings, labels)
    print(f"\nDiagonal loss: {loss_diag_val.item():.4f}")
    
    print("=== Test Complete ===\n")


if __name__ == "__main__":
    test_mahalanobis_loss()