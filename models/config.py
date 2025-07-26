# models/config.py - 모델 설정 (파일명 수정)
"""
COCONUT Model Configuration

DESIGN PHILOSOPHY:
- Unified model configuration for both stages
- CCNet architecture parameters
- Feature dimension specification
"""

import dataclasses
from pathlib import Path
from typing import Optional

@dataclasses.dataclass
class PalmRecognizerConfig:
    """손금 인식기 (CCNet) 설정"""
    config_file: Path
    architecture: str
    num_classes: int
    com_weight: float
    feature_dimension: int
    # 사전 훈련용
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 1024
    # 온라인 적응용  
    load_weights_folder: Optional[str] = None
    #1
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

print("✅ PalmRecognizerConfig 수정 완료!")