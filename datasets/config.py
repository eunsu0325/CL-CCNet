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

@dataclasses.dataclass
class DatasetConfig:
    """데이터셋 설정"""
    config_file: Path
    type: str
    height: int
    width: int
    use_angle_normalization: bool
    
    # 🔥 NEW: 배치 처리 설정
    samples_per_label: Optional[int] = 5  # 라벨당 샘플 수
    
    # 사전 훈련용
    train_set_file: Optional[str] = None
    test_set_file: Optional[str] = None
    # 온라인 적응용
    dataset_path: Optional[Path] = None
    
    def __post_init__(self):
        print(f"[Dataset] 🎯 Batch configuration:")
        print(f"   Samples per label: {self.samples_per_label}")
        print(f"   Dataset type: {self.type}")