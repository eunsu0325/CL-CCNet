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