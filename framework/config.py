"""
CoCoNut Framework Configuration Classes

DESIGN PHILOSOPHY:
- Focus on Replay Buffer configuration
- Remove all W2ML complexity
- Keep essential continual learning settings
- Maintain checkpoint and buffer configurations
"""

import dataclasses
from pathlib import Path
from typing import Optional, List

# === 연속학습 실험용 설정들 ===
@dataclasses.dataclass
class ReplayBufferConfig:
    """지능형 리플레이 버퍼 설정 (핵심 기여)"""
    config_file: Path
    maximize_diversity: bool
    max_buffer_size: int
    similarity_threshold: float
    storage_path: str
    feature_extraction_for_diversity: bool
    # 버퍼 관리 설정
    enable_smart_sampling: Optional[bool] = True
    diversity_update_frequency: Optional[int] = 10
    # 모델 저장 경로
    model_save_path: Optional[str] = "./results/models/"

@dataclasses.dataclass
class ContinualLearnerConfig:
    """연속 학습기 설정 (Hard Mining 지원)"""
    config_file: Path
    adaptation: bool
    adaptation_epochs: int
    sync_frequency: int
    replay_weight: float
    new_data_weight: float
    # 중간 저장 빈도 설정
    intermediate_save_frequency: Optional[int] = 100
    # 기본 학습 설정
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 10
    # 🔥 Hard Mining 설정
    enable_hard_mining: Optional[bool] = True
    hard_mining_ratio: Optional[float] = 0.3

@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정 (단순화됨)"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치

@dataclasses.dataclass
class ModelSavingConfig:
    """모델 저장 설정"""
    config_file: Path
    final_save_path: str = "/content/drive/MyDrive/CoCoNut_STAR"
    intermediate_save_frequency: int = 100
    enable_intermediate_save: bool = True
    include_timestamp: bool = True
    auto_generate_readme: bool = True

@dataclasses.dataclass
class DataAugmentationConfig:
    """데이터 증강 설정"""
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

# === 사전 훈련용 설정들 ===
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
