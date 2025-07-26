# framework/coconut.py - import 부분 수정
"""
=== COCONUT STAGE 2: CONTINUAL LEARNING WITH HEADLESS SUPPORT ===
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import json
import pickle
import time
from pathlib import Path
from tqdm import tqdm
import datetime

# Faiss import with fallback
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
    print("[System] 🚀 Faiss available - Buffer optimization enabled")
except ImportError:
    FAISS_AVAILABLE = False
    print("[System] ⚠️ Faiss not found - using PyTorch fallback")

from models.ccnet_model import ccnet, HeadlessVerifier
from framework.replay_buffer import CoconutReplayBuffer
from .losses import SupConLoss  # 🔥 수정된 import
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class CoconutSystem:
    def __init__(self, config):
        """
        Continual Learning CoCoNut System with Headless Support
        
        NEW FEATURES:
        - Headless mode configuration
        - Metric-based verification
        - Runtime head removal
        """
        print("="*80)
        print("🥥 COCONUT STAGE 2: HEADLESS CONTINUAL LEARNING")
        print("="*80)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🔥 Headless Configuration
        self.headless_mode = getattr(config.palm_recognizer, 'headless_mode', False)
        self.verification_method = getattr(config.palm_recognizer, 'verification_method', 'classification')
        self.metric_type = getattr(config.palm_recognizer, 'metric_type', 'cosine')
        self.similarity_threshold = getattr(config.palm_recognizer, 'similarity_threshold', 0.5)
        
        print(f"🔧 HEADLESS CONFIGURATION:")
        print(f"   Headless Mode: {self.headless_mode}")
        print(f"   Verification: {self.verification_method}")
        if self.verification_method == 'metric':
            print(f"   Metric Type: {self.metric_type}")
            print(f"   Threshold: {self.similarity_threshold}")
        print("="*80)
        
        # 체크포인트 경로 설정
        self.checkpoint_dir = Path('/content/drive/MyDrive/CoCoNut_STAR/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 시스템 구성 요소 초기화
        self._initialize_models_with_headless()
        self._initialize_replay_buffer()
        self._initialize_verification_system()
        self._initialize_basic_learning()
        
        # 학습 상태 초기화
        self.learner_step_count = 0
        self.global_dataset_index = 0
        self._initialize_simple_stats()
        
        # 이전 체크포인트에서 복원
        self._resume_from_latest_checkpoint()
        
        print(f"[System] 🥥 CoCoNut Headless ready!")
        print(f"[System] Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"[System] Starting from step: {self.learner_step_count}")

    def _initialize_models_with_headless(self):
        """Headless 지원으로 모델 초기화"""
        print(f"[System] Initializing CCNet models (headless: {self.headless_mode})...")
        cfg_model = self.config.palm_recognizer
        
        # 🔥 Headless 모드로 모델 생성
        self.predictor_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode
        ).to(self.device)
        
        self.learner_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode
        ).to(self.device)
        
        # 사전 훈련된 가중치 로드
        weights_path = cfg_model.load_weights_folder
        print(f"[System] Loading pretrained weights from: {weights_path}")
        try:
            # 전체 모델 가중치 로드 (head 포함)
            full_state_dict = torch.load(weights_path, map_location=self.device)
            
            if self.headless_mode:
                # Headless 모드: classification head 제거
                print("[System] 🔪 Removing classification head from pretrained weights...")
                filtered_state_dict = {k: v for k, v in full_state_dict.items() 
                                     if not k.startswith('arclayer_')}
                print(f"   Removed {len(full_state_dict) - len(filtered_state_dict)} head parameters")
                
                self.predictor_net.load_state_dict(filtered_state_dict, strict=False)
                self.learner_net.load_state_dict(filtered_state_dict, strict=False)
                print("[System] ✅ Headless models loaded (head removed)")
            else:
                # Normal 모드: 전체 가중치 로드
                self.predictor_net.load_state_dict(full_state_dict)
                self.learner_net.load_state_dict(full_state_dict)
                print("[System] ✅ Full models loaded (head included)")
                
        except FileNotFoundError:
            print(f"[System] ⚠️ Pretrained weights not found. Starting with random weights.")
        except Exception as e:
            print(f"[System] ❌ Failed to load weights: {e}")
            
        self.predictor_net.eval()
        self.learner_net.train()
        
        # 모델 정보 출력
        pred_info = self.predictor_net.get_model_info()
        learn_info = self.learner_net.get_model_info()
        print(f"[System] Predictor: {pred_info}")
        print(f"[System] Learner: {learn_info}")

    def _initialize_verification_system(self):
        """검증 시스템 초기화 (Headless vs Classification)"""
        if self.verification_method == 'metric':
            # 메트릭 기반 검증기 초기화
            self.verifier = HeadlessVerifier(
                metric_type=self.metric_type,
                threshold=self.similarity_threshold
            )
            print(f"[System] ✅ Metric-based verifier initialized")
        else:
            # Classification 기반 검증
            self.verifier = None
            print(f"[System] ✅ Classification-based verification")

    def _initialize_replay_buffer(self):
        """리플레이 버퍼 초기화 (Hard Mining + 데이터 증강 설정 추가)"""
        print("[System] Initializing Intelligent Replay Buffer...")
        cfg_buffer = self.config.replay_buffer
        cfg_model = self.config.palm_recognizer

        buffer_storage_path = Path(cfg_buffer.storage_path)
        
        self.replay_buffer = CoconutReplayBuffer(
            config=cfg_buffer,
            storage_dir=buffer_storage_path,
            feature_dimension=cfg_model.feature_dimension 
        )
        
        # 리플레이 버퍼에 특징 추출기 설정
        self.replay_buffer.set_feature_extractor(self.learner_net)
        
        # 🔥 Hard Mining 설정 전달
        cfg_learner = self.config.continual_learner
        if cfg_learner:
            self.replay_buffer.update_hard_mining_config(
                getattr(cfg_learner, 'enable_hard_mining', False),
                getattr(cfg_learner, 'hard_mining_ratio', 0.3)
            )
        
        # 🔥 데이터 증강 설정 전달
        cfg_augmentation = self.config.data_augmentation
        if cfg_augmentation:
            self.replay_buffer.update_augmentation_config(
                getattr(cfg_augmentation, 'enable_augmentation', False),
                cfg_augmentation
            )

    def _initialize_basic_learning(self):
        """기본 연속학습 시스템 초기화"""
        print("[System] 🎯 Initializing continual learning...")
        
        cfg_model = self.config.palm_recognizer
        cfg_loss = self.config.loss
        
        # Adam 옵티마이저
        self.optimizer = optim.Adam(
            self.learner_net.parameters(), 
            lr=cfg_model.learning_rate
        )
        
        # SupCon 손실 함수 (headless/normal 공통)
        self.contrastive_loss = SupConLoss(
            temperature=getattr(cfg_loss, 'temp', 0.07)
        )
        
        print(f"[System] ✅ Learning system initialized")
        print(f"[System] Optimizer: Adam (lr={cfg_model.learning_rate})")
        print(f"[System] Loss: SupConLoss (temp={getattr(cfg_loss, 'temp', 0.07)})")

    def _initialize_simple_stats(self):
        """통계 초기화"""
        self.simple_stats = {
            'total_learning_steps': 0,
            'buffer_additions': 0,
            'buffer_skips': 0,
            'losses': [],
            'processing_times': [],
            'batch_sizes': [],
            'buffer_diversity_scores': [],
            'verification_accuracies': []  # 새로운 메트릭
        }

    def process_single_frame(self, image: torch.Tensor, user_id: int):
        """
        단일 프레임 처리 (Headless 지원)
        """
        image = image.to(self.device)

        # 1. 예측기를 통한 실시간 인증
        self.predictor_net.eval()
        with torch.no_grad():
            if self.headless_mode:
                # Headless: 특징만 추출
                _, predictor_features = self.predictor_net(image.unsqueeze(0))
                embedding_from_predictor = predictor_features.squeeze(0)
            else:
                # Normal: 분류 + 특징
                logits, features = self.predictor_net(image.unsqueeze(0))
                embedding_from_predictor = features.squeeze(0)
        
        # 2. 학습기를 통한 최신 특징 추출
        self.learner_net.eval()
        with torch.no_grad():
            if self.headless_mode:
                _, learner_features = self.learner_net(image.unsqueeze(0))
                latest_embedding = learner_features.squeeze(0)
            else:
                _, features = self.learner_net(image.unsqueeze(0))
                latest_embedding = features.squeeze(0)
        self.learner_net.train()
        
        # 3. 리플레이 버퍼에 추가
        buffer_size_before = len(self.replay_buffer.image_storage)
        self.replay_buffer.add(image, user_id)
        buffer_size_after = len(self.replay_buffer.image_storage)
        
        # 통계 업데이트
        if buffer_size_after > buffer_size_before:
            self.simple_stats['buffer_additions'] += 1
        else:
            self.simple_stats['buffer_skips'] += 1
        
        # 4. 연속학습 조건 확인
        buffer_size = len(self.replay_buffer.image_storage)
        unique_users = len(set([item['user_id'] for item in self.replay_buffer.image_storage]))
        
        if unique_users < 2:
            print(f"[Learning] 📊 Waiting for diversity (Dataset pos: {self.global_dataset_index}):")
            print(f"   Buffer size: {buffer_size}")
            print(f"   Unique users: {unique_users}/2 minimum")
            return
        
        # 5. 연속학습 실행
        self._basic_continual_learning_with_headless(image, user_id)

    def _basic_continual_learning_with_headless(self, new_image, new_user_id):
        """Headless 지원 기본 연속학습"""
        self.learner_step_count += 1
        
        print(f"[Learning] {'='*50}")
        print(f"[Learning] {'HEADLESS' if self.headless_mode else 'CLASSIFICATION'} CONTINUAL STEP {self.learner_step_count}")
        print(f"[Learning] {'='*50}")
        
        cfg_learner = self.config.continual_learner
        cfg_model = self.config.palm_recognizer
        target_batch_size = cfg_model.batch_size

        # 배치 구성
        replay_count = target_batch_size - 1
        replay_images, replay_labels = self.replay_buffer.sample_with_replacement(replay_count)
        
        all_images = [new_image] + replay_images
        all_labels = [new_user_id] + replay_labels
        
        actual_batch_size = len(all_images)
        
        print(f"[Learning] Batch Analysis:")
        print(f"   Target batch size: {target_batch_size}")
        print(f"   Actual batch size: {actual_batch_size}")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
        
        # 연속학습 실행
        total_loss = 0.0
        processing_start = time.time()
        
        for epoch in range(cfg_learner.adaptation_epochs):
            print(f"[Learning] 🔄 Adaptation epoch {epoch+1}/{cfg_learner.adaptation_epochs}")
            
            if self.headless_mode:
                epoch_loss = self._run_headless_learning_step(all_images, all_labels)
            else:
                epoch_loss = self._run_classification_learning_step(all_images, all_labels)
            
            total_loss += epoch_loss
        
        processing_time = time.time() - processing_start
        average_loss = total_loss / cfg_learner.adaptation_epochs
        
        # 통계 업데이트
        self.simple_stats['total_learning_steps'] += 1
        self.simple_stats['losses'].append(average_loss)
        self.simple_stats['processing_times'].append(processing_time)
        self.simple_stats['batch_sizes'].append(actual_batch_size)
        
        # 버퍼 다양성 통계
        diversity_stats = self.replay_buffer.get_diversity_stats()
        self.simple_stats['buffer_diversity_scores'].append(diversity_stats['diversity_score'])
        
        print(f"[Learning] 📊 Step {self.learner_step_count} Results:")
        print(f"   Average loss: {average_loss:.6f}")
        print(f"   Processing time: {processing_time*1000:.2f}ms")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
        
        # 모델 동기화 체크
        if self.learner_step_count % cfg_learner.sync_frequency == 0:
            self._sync_weights()

    def _run_headless_learning_step(self, images: list, labels: list):
        """Headless 모드 학습 스텝"""
        print(f"[Learning] 🧠 Headless learning with {len(images)} samples")
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # 임베딩 추출
        embeddings = []
        for i, img in enumerate(images):
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # Headless forward: logits=None, features만 사용
            _, embedding = self.learner_net(img)
            embeddings.append(embedding)
        
        # 배치 텐서 구성
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # SupCon 손실 계산
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        print("[Learning] 🎯 Computing SupCon loss (headless mode)...")
        loss = self.contrastive_loss(embeddings_for_loss, labels_tensor)
        
        # 역전파
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            print("[Learning] ✅ Headless gradient update completed")
        else:
            print("[Learning] ⚠️ No gradient - loss computation issue")
        
        print(f"[Learning] ✅ Headless Loss: {loss.item():.6f}")
        return loss.item()

    def _run_classification_learning_step(self, images: list, labels: list):
        """Classification 모드 학습 스텝"""
        print(f"[Learning] 🧠 Classification learning with {len(images)} samples")
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # 임베딩 추출
        embeddings = []
        for i, img in enumerate(images):
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # Classification forward: 분류와 특징 모두 사용
            _, embedding = self.learner_net(img)
            embeddings.append(embedding)
        
        # 배치 텐서 구성
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # SupCon 손실 계산 (classification head와 독립적)
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)
        
        print("[Learning] 🎯 Computing SupCon loss (classification mode)...")
        loss = self.contrastive_loss(embeddings_for_loss, labels_tensor)
        
        # 역전파
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            print("[Learning] ✅ Classification gradient update completed")
        else:
            print("[Learning] ⚠️ No gradient - loss computation issue")
        
        print(f"[Learning] ✅ Classification Loss: {loss.item():.6f}")
        return loss.item()

    def _sync_weights(self):
        """가중치 동기화 (Headless 지원)"""
        self.predictor_net.load_state_dict(self.learner_net.state_dict())
        self.predictor_net.eval()
        
        print(f"\n[Sync] 🔄 MODEL SYNCHRONIZATION ({'Headless' if self.headless_mode else 'Classification'})")
        print(f"[Sync] {'='*50}")
        print(f"[Sync] ✅ Predictor updated!")
        print(f"[Sync] {'='*50}\n")

    # 나머지 메서드들 (run_experiment, _save_complete_checkpoint 등)은 기존과 동일하게 유지...
    def run_experiment(self):
        """연속학습 실험 실행"""
        print(f"[System] Starting {'headless' if self.headless_mode else 'classification'} continual learning from step {self.learner_step_count}...")

        # 타겟 데이터셋 준비
        cfg_dataset = self.config.dataset
        target_dataset = MyDataset(txt=str(cfg_dataset.dataset_path), train=False)
        target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False)
        
        # 이미 처리한 데이터들은 건너뛰기
        dataset_list = list(target_dataloader)
        total_steps = len(dataset_list)
        
        if self.global_dataset_index >= total_steps:
            print(f"[System] All data already processed! ({self.global_dataset_index}/{total_steps})")
            return
        
        print(f"[System] Resuming from dataset position {self.global_dataset_index}/{total_steps}")
        print(f"[System] Remaining data: {total_steps - self.global_dataset_index}")

        # 이어서 학습할 데이터만 추출
        remaining_data = dataset_list[self.global_dataset_index:]
        
        for data_offset, (datas, user_id) in enumerate(tqdm(remaining_data, desc="Continual Learning")):
            
            # 전체 데이터셋에서의 현재 위치 업데이트
            self.global_dataset_index = self.global_dataset_index + data_offset
            
            primary_image = datas[0].squeeze(0)
            user_id = user_id.item()

            # 한 프레임 처리
            self.process_single_frame(primary_image, user_id)

            # 설정된 빈도에 따라 체크포인트 저장
            save_frequency = getattr(self.config.continual_learner, 'intermediate_save_frequency', 50)
            if save_frequency > 0 and self.learner_step_count > 0 and self.learner_step_count % save_frequency == 0:
                self._save_complete_checkpoint()

        # 마지막 데이터 처리 후 인덱스 업데이트
        self.global_dataset_index = total_steps

        # 실험 종료 후 최종 체크포인트 저장
        print(f"\n[System] {'Headless' if self.headless_mode else 'Classification'} continual learning experiment finished.")
        self._save_complete_checkpoint()
        self.save_system_state()

    # 기존의 다른 메서드들 (_save_complete_checkpoint, save_system_state 등)은 
    # 동일하게 유지하되 headless 정보만 추가로 저장
    def _save_complete_checkpoint(self):
        """완전한 체크포인트 저장 (headless 정보 포함)"""
        step = self.learner_step_count
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 체크포인트 데이터 준비
        checkpoint = {
            'step_count': step,
            'global_dataset_index': self.global_dataset_index,
            'timestamp': timestamp,
            'learner_state_dict': self.learner_net.state_dict(),
            'predictor_state_dict': self.predictor_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'simple_stats': self.simple_stats,
            # 🔥 Headless 정보 추가
            'headless_mode': self.headless_mode,
            'verification_method': self.verification_method,
            'config_info': {
                'batch_size': self.config.palm_recognizer.batch_size,
                'learning_rate': self.config.palm_recognizer.learning_rate,
                'loss_temperature': getattr(self.config.loss, 'temp', 0.07),
                'headless_mode': self.headless_mode,
                'verification_method': self.verification_method,
            }
        }
        
        # 메인 체크포인트 저장
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        print(f"[Checkpoint] 💾 Complete checkpoint saved:")
        print(f"   📁 Model: checkpoint_step_{step}.pth")
        print(f"   🔧 Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"   📍 Dataset position: {self.global_dataset_index}")

    def save_system_state(self):
        """시스템 상태 저장 (headless 정보 포함)"""
        custom_save_path = Path('/content/drive/MyDrive/CoCoNut_STAR')
        custom_save_path.mkdir(parents=True, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "headless" if self.headless_mode else "classification"
        
        # 모드별로 다른 파일명 사용
        custom_learner_path = custom_save_path / f'coconut_{mode_suffix}_model_{timestamp}.pth'
        custom_predictor_path = custom_save_path / f'coconut_{mode_suffix}_predictor_{timestamp}.pth'
        
        torch.save(self.learner_net.state_dict(), custom_learner_path)
        torch.save(self.predictor_net.state_dict(), custom_predictor_path)
        
        print(f"[System] ✅ CoCoNut {mode_suffix.title()} 모델 저장 완료:")
        print(f"  🎯 사용자 지정 경로: {custom_save_path}")
        print(f"  📁 Learner 모델: {custom_learner_path.name}")
        print(f"  📁 Predictor 모델: {custom_predictor_path.name}")
        print(f"  🔧 Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"  🕐 타임스탬프: {timestamp}")

    def _resume_from_latest_checkpoint(self):
  
      checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_step_*.pth'))
      
      if not checkpoint_files:
          print("[Resume] 📂 No checkpoints found - starting fresh")
          return
      
      # 가장 최신 체크포인트 찾기
      latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
      step_num = int(latest_checkpoint.stem.split('_')[-1])
      
      print(f"[Resume] 🔄 Found checkpoint: {latest_checkpoint.name}")
      print(f"[Resume] 📍 Resuming from step: {step_num}")
      
      try:
          # 체크포인트 로드
          checkpoint = torch.load(latest_checkpoint, map_location=self.device)
          
          # 🔥 Headless 모드용 state_dict 필터링
          learner_state_dict = checkpoint['learner_state_dict']
          predictor_state_dict = checkpoint['predictor_state_dict']
          
          if self.headless_mode:
              print("[Resume] 🔪 Filtering out classification head from checkpoint...")
              # arclayer_ 로 시작하는 키들 제거
              learner_filtered = {k: v for k, v in learner_state_dict.items() 
                                if not k.startswith('arclayer_')}
              predictor_filtered = {k: v for k, v in predictor_state_dict.items() 
                                  if not k.startswith('arclayer_')}
              
              removed_count = len(learner_state_dict) - len(learner_filtered)
              print(f"   Removed {removed_count} classification head parameters")
              
              # 필터링된 state_dict 로드
              self.learner_net.load_state_dict(learner_filtered, strict=False)
              self.predictor_net.load_state_dict(predictor_filtered, strict=False)
          else:
              # Normal 모드: 전체 로드
              self.learner_net.load_state_dict(learner_state_dict)
              self.predictor_net.load_state_dict(predictor_state_dict)
          
          # 옵티마이저 상태 복원
          self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
          # 학습 상태 복원
          self.learner_step_count = checkpoint['step_count']
          self.global_dataset_index = checkpoint.get('global_dataset_index', 0)
          self.simple_stats = checkpoint.get('simple_stats', self.simple_stats)
          
          print(f"[Resume] ✅ Successfully resumed from step {self.learner_step_count}")
          print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
          print(f"   Dataset position: {self.global_dataset_index}")
      
      except Exception as e:
          print(f"[Resume] ❌ Failed to resume: {e}")
          print(f"[Resume] 🔄 Starting fresh instead")
          self.learner_step_count = 0
          self.global_dataset_index = 0

# 실제 파일에 적용
print("🔧 체크포인트 로딩 로직 수정 중...")