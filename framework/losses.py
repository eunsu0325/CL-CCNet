# framework/losses.py - 완전 단순화된 버전
"""
CoCoNut Simplified Loss Functions

DESIGN PHILOSOPHY:
- Focus on Replay Buffer innovation only
- Use proven SupCon loss for stable continual learning
- Remove all W2ML complexity for clear paper contribution

STATUS: W2ML removed, basic SupCon only
"""

# 기존 CCNet의 검증된 SupConLoss를 그대로 사용
from loss import SupConLoss

# 기존 복잡한 W2ML 관련 클래스들 모두 제거:
# - CompleteW2MLSupConLoss (삭제)
# - DifficultyWeightedSupConLoss (삭제)
# - create_w2ml_loss() (삭제)
# - benchmark_faiss_w2ml_performance() (삭제)


def create_coconut_loss(temperature=0.07):
    """
    CoCoNut용 단순한 손실함수 생성
    
    Args:
        temperature: SupCon 온도 파라미터
        
    Returns:
        기본 SupConLoss 인스턴스
    """
    return SupConLoss(temperature=temperature)


def get_coconut_loss_config():
    """
    CoCoNut 권장 손실함수 설정
    
    Returns:
        단순화된 설정 딕셔너리
    """
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


if __name__ == "__main__":
    # 단순한 테스트
    print("🥥 CoCoNut Simplified Loss Functions")
    print("✅ Using basic SupConLoss for continual learning")
    
    # 기본 손실함수 테스트
    loss_fn = create_coconut_loss()
    print(f"✅ Created SupConLoss with temperature: {loss_fn.temperature}")
    
    # 테스트 데이터로 검증
    import torch
    batch_size = 8
    features = torch.randn(batch_size, 2, 512)  # [batch, views, feature_dim]
    labels = torch.randint(0, 4, (batch_size,))
    
    loss = loss_fn(features, labels)
    print(f"✅ Test loss computed: {loss.item():.6f}")
    print("🚀 Simplified loss functions ready!")