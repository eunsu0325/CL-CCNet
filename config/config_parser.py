# config/config_parser.py - 설정 파서 (수정된 버전)
"""
COCONUT Configuration Parser

DESIGN PHILOSOPHY:
- Unified configuration management for both stages
- Automatic type validation and conversion
- Clear separation between pretrain and adaptation configs
- 🔥 ModelSaving configuration support
"""

import dataclasses
import sys
from os import PathLike
from pathlib import Path
from typing import List, Union, get_args, get_origin

import yaml

from datasets.config import DatasetConfig
from framework.config import (
    ContinualLearnerConfig, ReplayBufferConfig, LossConfig, 
    W2MLExperimentConfig, TrainingConfig, PathsConfig, ModelSavingConfig
)
from models.config import PalmRecognizerConfig


class ConfigParser():
    """
    COCONUT 설정 파서
    
    FEATURES:
    - Supports both pretrain and adaptation configurations
    - Automatic type validation and conversion
    - Extensible design for new configuration types
    - 🔥 ModelSaving configuration support
    """
    
    def __init__(self, config_file: Union[str, PathLike, Path]) -> None:
        self.filename = Path(config_file)
        self.config_dict = {}

        # 설정 속성 초기화
        self.dataset = None
        self.palm_recognizer = None
        self.continual_learner = None
        self.replay_buffer = None
        self.loss = None
        self.w2ml_experiment = None
        self.model_saving = None  # 🔥 새로운 모델 저장 설정
        
        # 사전 훈련 전용 설정
        self.training = None
        self.paths = None

        self.parse()

    def parse(self):
        """설정 파일을 파싱하고 객체를 생성합니다."""
        
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.config_dict = yaml.safe_load(file)

        # YAML 리스트를 튜플로 변환
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if isinstance(value, List):
                    config_type[key] = tuple(value)

        # 데이터 타입 검증 및 자동 변환
        for config_type_key, config_type in self.config_dict.items():
            # 🔥 특별 처리: Design_Documentation은 무시
            if config_type_key == 'Design_Documentation':
                print(f"[CONFIG] Skipping Design_Documentation (metadata only)")
                continue
                
            config_class_name = f'{config_type_key}Config' 
            
            # 클래스 찾기 시도
            config_type_class = None
            try:
                config_type_class = getattr(sys.modules[__name__], config_class_name)
            except AttributeError:
                # 현재 모듈에 없으면 import된 모듈들에서 찾기
                for module_name in ['datasets.config', 'framework.config', 'models.config']:
                    try:
                        module = sys.modules[module_name]
                        config_type_class = getattr(module, config_class_name)
                        break
                    except (KeyError, AttributeError):
                        continue
            
            if config_type_class is None:
                print(f"[CONFIG] Warning: {config_class_name} not found, skipping...")
                continue

            for field in dataclasses.fields(config_type_class):
                if field.name == 'config_file':
                    continue
                
                if field.name not in config_type:
                    continue

                value = config_type[field.name]
                if value is not None:
                    expected_type = field.type
                    if get_origin(expected_type) is Union:
                        expected_type = [tp for tp in get_args(expected_type) if tp is not type(None)]
                    else:
                        expected_type = [expected_type]

                    if not any(isinstance(value, tp) for tp in expected_type):
                        if len(expected_type) == 1:
                            print(f'[CONFIG] Converting {field.name} from {type(value).__name__} '
                                  f'to {expected_type[0].__name__}.')
                            config_type[field.name] = expected_type[0](value)
                        else:
                            assert False, f'Ambiguous type for {field.name}, cannot auto-convert!'

        # 모든 설정 객체에 원본 설정 파일의 경로 추가
        for config_type_key, config_type in self.config_dict.items():
            if config_type_key != 'Design_Documentation':  # 메타데이터 제외
                config_type['config_file'] = self.filename

        # 경로 문자열을 Path 객체로 변환
        for config_type_key, config_type in self.config_dict.items():
            if config_type_key != 'Design_Documentation':  # 메타데이터 제외
                for key, value in config_type.items():
                    if isinstance(value, str) and ('path' in key.lower() or 'folder' in key.lower()):
                        config_type[key] = Path(value).absolute()

        # 최종 설정 객체 생성
        if 'Dataset' in self.config_dict:
            self.dataset = DatasetConfig(**self.config_dict['Dataset'])
        if 'PalmRecognizer' in self.config_dict:
            self.palm_recognizer = PalmRecognizerConfig(**self.config_dict['PalmRecognizer'])
        if 'ContinualLearner' in self.config_dict:
            self.continual_learner = ContinualLearnerConfig(**self.config_dict['ContinualLearner'])
        if 'ReplayBuffer' in self.config_dict:
            self.replay_buffer = ReplayBufferConfig(**self.config_dict['ReplayBuffer'])
        if 'Loss' in self.config_dict:
            self.loss = LossConfig(**self.config_dict['Loss'])
        if 'W2ML_Experiment' in self.config_dict:
            self.w2ml_experiment = W2MLExperimentConfig(**self.config_dict['W2ML_Experiment'])
        # 🔥 새로운 모델 저장 설정
        if 'ModelSaving' in self.config_dict:
            self.model_saving = ModelSavingConfig(**self.config_dict['ModelSaving'])
        else:
            # 기본값으로 설정 (호환성 유지)
            self.model_saving = None
            
        # 사전 훈련 전용 설정
        if 'Training' in self.config_dict:
            self.training = TrainingConfig(**self.config_dict['Training'])
        if 'Paths' in self.config_dict:
            self.paths = PathsConfig(**self.config_dict['Paths'])

    def __str__(self):
        string = ''
        if self.dataset:
            string += f'----- Dataset --- START -----\n{self.dataset}\n----- Dataset --- END -------\n'
        if self.palm_recognizer:
            string += f'----- PalmRecognizer --- START -----\n{self.palm_recognizer}\n----- PalmRecognizer --- END -------\n'
        if self.continual_learner:
            string += f'----- ContinualLearner --- START -----\n{self.continual_learner}\n----- ContinualLearner --- END -------\n'
        if self.replay_buffer:
            string += f'----- ReplayBuffer --- START -----\n{self.replay_buffer}\n----- ReplayBuffer --- END -------\n'
        if self.loss:
            string += f'----- Loss --- START -----\n{self.loss}\n----- Loss --- END -------\n'
        if self.w2ml_experiment:
            string += f'----- W2ML_Experiment --- START -----\n{self.w2ml_experiment}\n----- W2ML_Experiment --- END -------\n'
        if self.model_saving:
            string += f'----- ModelSaving --- START -----\n{self.model_saving}\n----- ModelSaving --- END -------\n'
        if self.training:
            string += f'----- Training --- START -----\n{self.training}\n----- Training --- END -------\n'
        if self.paths:
            string += f'----- Paths --- START -----\n{self.paths}\n----- Paths --- END -------\n'
        return string