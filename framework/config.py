# framework/config.py - W2ML 설정 제거된 단순 버전
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
from typing import Optional

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
    """연속 학습기 설정 (단순화됨)"""
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

@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정 (단순화됨)"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치

# W2MLExperimentConfig 클래스 완전 제거

@dataclasses.dataclass
class ModelSavingConfig:
    """모델 저장 설정"""
    config_file: Path
    final_save_path: str = "/content/drive/MyDrive/CoCoNut_STAR"
    intermediate_save_frequency: int = 100
    enable_intermediate_save: bool = True
    include_timestamp: bool = True
    auto_generate_readme: bool = True

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

@dataclasses.dataclass
class PalmRecognizerConfig:
    """손금 인식기 (CCNet) 설정 - Headless 지원 추가"""
    config_file: Path
    architecture: str
    num_classes: int
    com_weight: float
    feature_dimension: int
    
    # 기본 학습 설정
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 1024
    
    # 모델 로딩
    load_weights_folder: Optional[str] = None
    
    # 🔥 NEW: Headless Configuration
    headless_mode: Optional[bool] = False  # true: 헤드 제거, false: 헤드 유지
    verification_method: Optional[str] = "classification"  # "classification" or "metric"
    metric_type: Optional[str] = "cosine"  # "cosine" or "l2"
    similarity_threshold: Optional[float] = 0.5  # 메트릭 기반 인증 임계값
    
    def __post_init__(self):
        """설정 검증 및 자동 조정"""
        # Headless 모드에서는 metric verification 강제
        if self.headless_mode and self.verification_method == "classification":
            print(f"[Config] Warning: Headless mode requires metric verification. "
                  f"Changing from '{self.verification_method}' to 'metric'")
            self.verification_method = "metric"
        
        # 설정 정보 출력
        print(f"[Config] 🔧 Model Configuration:")
        print(f"   Architecture: {self.architecture}")
        print(f"   Headless Mode: {self.headless_mode}")
        print(f"   Verification: {self.verification_method}")
        if self.verification_method == "metric":
            print(f"   Metric Type: {self.metric_type}")
            print(f"   Threshold: {self.similarity_threshold}")
    