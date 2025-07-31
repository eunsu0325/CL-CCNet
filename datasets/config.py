# datasets/config.py - 배치 지원 추가
"""
COCONUT Dataset Configuration

DESIGN PHILOSOPHY:
- Batch processing support
- Dataset-specific configuration
"""

import dataclasses
from pathlib import Path
from typing import Optional

# framework/config.py에 추가할 내용

@dataclasses.dataclass
class UserNodeConfig:
    """사용자 노드 시스템 설정"""
    config_file: Path
    enable_user_nodes: bool = True  # 사용자 노드 시스템 ON/OFF
    node_save_path: str = "./results/user_nodes/"
    collision_threshold: float = 0.5
    use_faiss_index: bool = True
    max_samples_per_user: int = 10
    
    # PQ 압축 설정
    enable_compression: bool = False
    pq_nbits: int = 8
    pq_nsegments: int = 32
    
    def __post_init__(self):
        print(f"[Config] 🎯 User Node System:")
        print(f"   Enabled: {self.enable_user_nodes}")
        if self.enable_user_nodes:
            print(f"   Collision threshold: {self.collision_threshold}")
            print(f"   Max samples per user: {self.max_samples_per_user}")
            print(f"   Compression: {'ON' if self.enable_compression else 'OFF'}")

@dataclasses.dataclass
class LoopClosureConfig:
    """루프 클로저 설정"""
    config_file: Path
    enabled: bool = True  # 루프 클로저 ON/OFF
    retraining_epochs: int = 5
    priority_weight: float = 2.0
    
    def __post_init__(self):
        print(f"[Config] 🔄 Loop Closure:")
        print(f"   Enabled: {self.enabled}")
        if self.enabled:
            print(f"   Retraining epochs: {self.retraining_epochs}")
            print(f"   Priority weight: {self.priority_weight}")

# LossConfig 수정 - 온라인 학습 설정 추가
@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정 (수정됨)"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"
    
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치
    
    # 🔥 온라인 학습 설정 (새로 추가)
    online_learning: Optional[Dict] = None
    
    def __post_init__(self):
        if self.online_learning is None:
            self.online_learning = {
                'enable_mahalanobis': True,
                'supcon_weight': 1.0,
                'mahal_weight': 0.2,
                'alternate_training': True
            }
        
        print(f"[Config] 📊 Loss Configuration:")
        print(f"   Base: {self.type} (temp={self.temp})")
        if self.online_learning:
            print(f"   Online Learning:")
            print(f"     Mahalanobis: {'ON' if self.online_learning['enable_mahalanobis'] else 'OFF'}")
            print(f"     Weights: SupCon={self.online_learning['supcon_weight']}, "
                  f"Mahal={self.online_learning['mahal_weight']}")
            print(f"     Alternate: {self.online_learning['alternate_training']}")