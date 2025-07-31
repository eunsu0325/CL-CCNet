# config/config_parser.py - User Node 지원 추가
"""
COCONUT Configuration Parser

DESIGN PHILOSOPHY:
- Unified configuration management for both stages
- Automatic type validation and conversion
- Clear separation between pretrain and adaptation configs
- 🔥 UserNode and LoopClosure configuration support
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
    TrainingConfig, PathsConfig, ModelSavingConfig, DataAugmentationConfig,
    UserNodeConfig, LoopClosureConfig  # 🔥 NEW
)
from models.config import PalmRecognizerConfig


class ConfigParser():
    """
    COCONUT 설정 파서
    
    FEATURES:
    - Supports both pretrain and adaptation configurations
    - Automatic type validation and conversion
    - Extensible design for new configuration types
    - 🔥 UserNode and LoopClosure configuration support
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
        self.model_saving = None
        self.data_augmentation = None
        self.user_node = None  # 🔥 NEW
        self.loop_closure = None  # 🔥 NEW
        
        # 사전 훈련 전용 설정
        self.training = None
        self.paths = None

        self.parse()

    def _convert_type(self, value, expected_type):
        """안전한 타입 변환"""
        if expected_type == List or get_origin(expected_type) == list:
            if isinstance(value, (list, tuple)):
                return list(value)
            else:
                return [value]
        elif expected_type == tuple:
            if isinstance(value, (list, tuple)):
                return tuple(value)
            else:
                return (value,)
        else:
            return expected_type(value)

    def parse(self):
        """설정 파일을 파싱하고 객체를 생성합니다."""
        
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.config_dict = yaml.safe_load(file)

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

            if not isinstance(config_type, dict):
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
                            try:
                                # 🔥 안전한 타입 변환 사용
                                config_type[field.name] = self._convert_type(value, expected_type[0])
                            except Exception as e:
                                print(f'[CONFIG] Warning: Failed to convert {field.name}: {e}')
                                # 변환 실패시 원본 값 유지
                                pass
                        else:
                            print(f'[CONFIG] Warning: Ambiguous type for {field.name}, keeping original value')

        # 모든 설정 객체에 원본 설정 파일의 경로 추가
        for config_type_key, config_type in self.config_dict.items():
            if config_type_key != 'Design_Documentation' and isinstance(config_type, dict):
                config_type['config_file'] = self.filename

        # 경로 문자열을 Path 객체로 변환
        for config_type_key, config_type in self.config_dict.items():
            if config_type_key != 'Design_Documentation' and isinstance(config_type, dict):
                for key, value in config_type.items():
                    if isinstance(value, str) and ('path' in key.lower() or 'folder' in key.lower()):
                        config_type[key] = Path(value).absolute()

        # 최종 설정 객체 생성
        try:
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
            if 'ModelSaving' in self.config_dict:
                self.model_saving = ModelSavingConfig(**self.config_dict['ModelSaving'])
            else:
                self.model_saving = None
            
            if 'DataAugmentation' in self.config_dict:
                self.data_augmentation = DataAugmentationConfig(**self.config_dict['DataAugmentation'])
            else:
                self.data_augmentation = None
            
            # 🔥 NEW: User Node and Loop Closure
            if 'UserNode' in self.config_dict:
                self.user_node = UserNodeConfig(**self.config_dict['UserNode'])
            else:
                self.user_node = None
                
            if 'LoopClosure' in self.config_dict:
                self.loop_closure = LoopClosureConfig(**self.config_dict['LoopClosure'])
            else:
                self.loop_closure = None
                
            # 사전 훈련 전용 설정
            if 'Training' in self.config_dict:
                self.training = TrainingConfig(**self.config_dict['Training'])
            if 'Paths' in self.config_dict:
                self.paths = PathsConfig(**self.config_dict['Paths'])
                
        except Exception as e:
            print(f"[CONFIG] Error creating config objects: {e}")
            raise

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
        if self.model_saving:
            string += f'----- ModelSaving --- START -----\n{self.model_saving}\n----- ModelSaving --- END -------\n'
        if self.data_augmentation:
            string += f'----- DataAugmentation --- START -----\n{self.data_augmentation}\n----- DataAugmentation --- END -------\n'
        if self.user_node:  # 🔥 NEW
            string += f'----- UserNode --- START -----\n{self.user_node}\n----- UserNode --- END -------\n'
        if self.loop_closure:  # 🔥 NEW
            string += f'----- LoopClosure --- START -----\n{self.loop_closure}\n----- LoopClosure --- END -------\n'
        if self.training:
            string += f'----- Training --- START -----\n{self.training}\n----- Training --- END -------\n'
        if self.paths:
            string += f'----- Paths --- START -----\n{self.paths}\n----- Paths --- END -------\n'
        return string

print("✅ config_parser.py User Node 지원 추가 완료!")