# framework/config.py - 배치 구성 제어 완전 버전

"""
CoCoNut Framework Configuration Classes

DESIGN PHILOSOPHY:
- Controlled batch composition for optimal continual learning
- Separate batch sizes for pretraining vs continual learning
- Precise positive/hard sample ratios
- Extensible sampling strategies
"""

import dataclasses
from pathlib import Path
from typing import Optional, List

# === 연속학습 실험용 설정들 ===
@dataclasses.dataclass
class ContinualLearnerConfig:
    """연속 학습기 설정 (🔥 배치 구성 제어 추가)"""
    config_file: Path
    adaptation: bool
    adaptation_epochs: int
    sync_frequency: int
    replay_weight: float
    new_data_weight: float
    
    # 기본 학습 설정
    intermediate_save_frequency: Optional[int] = 100
    learning_rate: Optional[float] = 0.001
    
    # 🔥 NEW: 연속학습 전용 배치 사이즈 (PalmRecognizer와 분리)
    continual_batch_size: Optional[int] = 10     # 연속학습 배치 사이즈
    batch_size: Optional[int] = None             # 호환성 유지 (deprecated)
    
    # 🔥 NEW: 배치 구성 비율 제어
    target_positive_ratio: Optional[float] = 0.3      # 30% positive pairs
    hard_mining_ratio: Optional[float] = 0.3          # 30% hard samples  
    enable_hard_mining: Optional[bool] = True         # Hard mining 활성화
    
    # 호환성 유지
    hard_mining_ratio_legacy: Optional[float] = None  # 기존 이름 지원
    
    def __post_init__(self):
        """설정 검증 및 호환성 처리"""
        # 1. 호환성 처리: batch_size → continual_batch_size
        if self.batch_size is not None and self.continual_batch_size == 10:
            self.continual_batch_size = self.batch_size
            print(f"[Config] Using legacy batch_size: {self.continual_batch_size}")
        
        # 2. 호환성 처리: hard_mining_ratio_legacy
        if self.hard_mining_ratio_legacy is not None:
            self.hard_mining_ratio = self.hard_mining_ratio_legacy
            print(f"[Config] Using legacy hard_mining_ratio: {self.hard_mining_ratio}")
        
        # 3. 비율 검증
        if not (0.0 <= self.target_positive_ratio <= 1.0):
            raise ValueError(f"target_positive_ratio must be 0.0-1.0, got {self.target_positive_ratio}")
        
        if not (0.0 <= self.hard_mining_ratio <= 1.0):
            raise ValueError(f"hard_mining_ratio must be 0.0-1.0, got {self.hard_mining_ratio}")
        
        # 4. 배치 구성 검증
        # Positive pairs는 2개씩이므로, 실제 positive sample 수는 ratio * 2
        max_positive_samples = self.target_positive_ratio * self.continual_batch_size
        max_hard_samples = self.hard_mining_ratio * self.continual_batch_size
        total_reserved = max_positive_samples + max_hard_samples
        
        if total_reserved > self.continual_batch_size:
            print(f"[Config] Warning: positive + hard samples ({total_reserved:.1f}) > batch_size ({self.continual_batch_size})")
            print(f"[Config] Some overlap may occur, adjusting ratios...")
            
            # 자동 조정
            adjustment_factor = self.continual_batch_size / total_reserved * 0.9
            self.target_positive_ratio *= adjustment_factor
            self.hard_mining_ratio *= adjustment_factor
            
            print(f"[Config] Adjusted ratios: positive={self.target_positive_ratio:.2f}, hard={self.hard_mining_ratio:.2f}")
        
        # 5. 최종 계획 출력
        planned_positive = int(self.continual_batch_size * self.target_positive_ratio)
        planned_hard = int(self.continual_batch_size * self.hard_mining_ratio)
        planned_regular = self.continual_batch_size - planned_positive - planned_hard
        
        print(f"[Config] 🎯 Continual Learning Batch Plan (size: {self.continual_batch_size}):")
        print(f"   Positive samples: {planned_positive} ({self.target_positive_ratio:.1%})")
        print(f"   Hard samples: {planned_hard} ({self.hard_mining_ratio:.1%})")
        print(f"   Regular samples: {planned_regular} ({planned_regular/self.continual_batch_size:.1%})")

@dataclasses.dataclass
class ReplayBufferConfig:
    """지능형 리플레이 버퍼 설정 (🔥 샘플링 전략 제어 추가)"""
    config_file: Path
    maximize_diversity: bool
    max_buffer_size: int
    similarity_threshold: float
    storage_path: str
    feature_extraction_for_diversity: bool
    
    # 기존 설정
    enable_smart_sampling: Optional[bool] = True
    diversity_update_frequency: Optional[int] = 10
    model_save_path: Optional[str] = "./results/models/"
    
    # 🔥 NEW: 샘플링 전략 제어
    sampling_strategy: Optional[str] = "controlled"    # "controlled", "balanced", "original"
    force_positive_pairs: Optional[bool] = True        # 항상 positive pairs 보장
    min_positive_pairs: Optional[int] = 1              # 최소 positive pair 수  
    max_positive_ratio: Optional[float] = 0.5          # 최대 positive ratio 제한
    
    def __post_init__(self):
        """샘플링 전략 검증"""
        valid_strategies = ["controlled", "balanced", "original"]
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}, got {self.sampling_strategy}")
        
        if not (0.0 <= self.max_positive_ratio <= 1.0):
            raise ValueError(f"max_positive_ratio must be 0.0-1.0, got {self.max_positive_ratio}")
        
        print(f"[Config] 🎯 Replay Buffer Sampling:")
        print(f"   Strategy: {self.sampling_strategy}")
        print(f"   Force positive pairs: {self.force_positive_pairs}")
        print(f"   Min positive pairs: {self.min_positive_pairs}")
        print(f"   Max positive ratio: {self.max_positive_ratio:.1%}")

@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정 (기존과 동일)"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치

@dataclasses.dataclass
class ModelSavingConfig:
    """모델 저장 설정 (기존과 동일)"""
    config_file: Path
    final_save_path: str = "/content/drive/MyDrive/CoCoNut_STAR"
    intermediate_save_frequency: int = 100
    enable_intermediate_save: bool = True
    include_timestamp: bool = True
    auto_generate_readme: bool = True

@dataclasses.dataclass
class DataAugmentationConfig:
    """데이터 증강 설정 (기존과 동일)"""
    config_file: Path
    enable_augmentation: bool = True
    augmentation_probability: float = 0.4
    enable_geometric: bool = True
    geometric_probability: float = 0.3
    max_rotation_degrees: int = 3
    max_translation_ratio: float = 0.05
    enable_resolution_adaptation: bool = True
    resolution_probability: float = 0.3
    intermediate_resolutions: Optional[List] = None
    resize_methods: Optional[List] = None
    enable_noise: bool = True
    noise_probability: float = 0.3
    noise_std_range: Optional[List] = None
    
    def __post_init__(self):
        """기본값 설정"""
        if self.intermediate_resolutions is None:
            self.intermediate_resolutions = [[64, 64], [96, 96], [160, 160]]
        if self.resize_methods is None:
            self.resize_methods = ["nearest", "bilinear", "bicubic", "lanczos"]
        if self.noise_std_range is None:
            self.noise_std_range = [0.01, 0.03]

# === 사전 훈련용 설정들 (기존과 동일) ===
@dataclasses.dataclass
class TrainingConfig:
    """사전 훈련용 Training 설정"""
    config_file: Path
    batch_size: int
    epoch_num: int
    lr: float
    redstep: int
    gpu_id: int

@dataclasses.dataclass  
class PathsConfig:
    """사전 훈련용 경로 설정"""
    config_file: Path
    checkpoint_path: str
    results_path: str
    save_interval: int
    test_interval: int