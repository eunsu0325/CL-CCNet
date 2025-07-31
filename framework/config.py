# framework/config.py - 배치 기반 재설계 버전

"""
CoCoNut Framework Configuration Classes (Batch-based Redesign)

DESIGN PHILOSOPHY:
- Batch processing for efficiency
- Simple hard negative mining
- No forced positive pair logic (automatic with batch)
"""

import dataclasses
from pathlib import Path
from typing import Optional, List

@dataclasses.dataclass
class ContinualLearnerConfig:
    """연속 학습기 설정 (배치 기반 단순화)"""
    config_file: Path
    adaptation: bool
    adaptation_epochs: int
    sync_frequency: int
    replay_weight: float
    new_data_weight: float
    
    # 기본 학습 설정
    intermediate_save_frequency: Optional[int] = 100
    learning_rate: Optional[float] = 0.001
    
    # 🔥 NEW: 단순화된 배치 설정
    training_batch_size: Optional[int] = 32      # 전체 학습 배치 크기
    hard_negative_ratio: Optional[float] = 0.3   # 하드 네거티브 비율만
    
    def __post_init__(self):
        """설정 검증"""
        if not (0.0 <= self.hard_negative_ratio <= 1.0):
            raise ValueError(f"hard_negative_ratio must be 0.0-1.0, got {self.hard_negative_ratio}")
        
        print(f"[Config] 🎯 Batch-based Continual Learning:")
        print(f"   Training batch size: {self.training_batch_size}")
        print(f"   Hard negative ratio: {self.hard_negative_ratio:.1%}")

@dataclasses.dataclass
class ReplayBufferConfig:
    """리플레이 버퍼 설정 (단순화)"""
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
    
    # 🔥 NEW: 단순화된 설정
    samples_per_user_limit: Optional[int] = 3    # 사용자당 최대 저장 수
    
    def __post_init__(self):
        print(f"[Config] 🎯 Simplified Replay Buffer:")
        print(f"   Max buffer size: {self.max_buffer_size}")
        print(f"   Samples per user limit: {self.samples_per_user_limit}")
        print(f"   Diversity threshold: {self.similarity_threshold}")

@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정 (변경 없음)"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치

@dataclasses.dataclass
class ModelSavingConfig:
    """모델 저장 설정 (변경 없음)"""
    config_file: Path
    final_save_path: str = "/content/drive/MyDrive/CoCoNut_STAR"
    intermediate_save_frequency: int = 100
    enable_intermediate_save: bool = True
    include_timestamp: bool = True
    auto_generate_readme: bool = True

@dataclasses.dataclass
class DataAugmentationConfig:
    """데이터 증강 설정 (변경 없음)"""
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
        if self.intermediate_resolutions is None:
            self.intermediate_resolutions = [[64, 64], [96, 96], [160, 160]]
        if self.resize_methods is None:
            self.resize_methods = ["nearest", "bilinear", "bicubic", "lanczos"]
        if self.noise_std_range is None:
            self.noise_std_range = [0.01, 0.03]

# === 사전 훈련용 설정들 (변경 없음) ===
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