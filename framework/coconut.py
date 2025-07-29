# framework/coconut.py - 완전 제어된 배치 구성 시스템 (수정 버전)

"""
=== COCONUT STAGE 2: CONTROLLED BATCH CONTINUAL LEARNING ===

🔥 MODIFIED FEATURES:
- 🎯 학습 먼저, 저장 나중 순서로 변경
- 💪 새로운 샘플 즉시 학습 활용
- 📊 스마트 버퍼 저장 로직
- 🔧 긍정쌍 확보를 위한 강제 저장
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
from .losses import SupConLoss
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class CoconutSystem:
    def __init__(self, config):
        """
        Continual Learning CoCoNut System with Controlled Batch Composition
        
        🔥 NEW FEATURES:
        - 학습 먼저, 저장 나중 순서
        - 새로운 샘플 즉시 활용
        - 스마트 버퍼 관리
        """
        print("="*80)
        print("🥥 COCONUT STAGE 2: IMPROVED CONTINUAL LEARNING")
        print("="*80)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Headless Configuration
        self.headless_mode = getattr(config.palm_recognizer, 'headless_mode', False)
        self.verification_method = getattr(config.palm_recognizer, 'verification_method', 'classification')
        self.metric_type = getattr(config.palm_recognizer, 'metric_type', 'cosine')
        self.similarity_threshold = getattr(config.palm_recognizer, 'similarity_threshold', 0.5)
        
        # 🔥 NEW: Controlled Batch Composition Configuration
        cfg_learner = self.config.continual_learner
        
        self.continual_batch_size = getattr(cfg_learner, 'continual_batch_size', 10)
        self.target_positive_ratio = getattr(cfg_learner, 'target_positive_ratio', 0.3)
        self.hard_mining_ratio = getattr(cfg_learner, 'hard_mining_ratio', 0.3)
        self.enable_hard_mining = getattr(cfg_learner, 'enable_hard_mining', True)
        
        print(f"🔧 IMPROVED BATCH COMPOSITION:")
        print(f"   Continual Batch Size: {self.continual_batch_size}")
        print(f"   Target Positive Ratio: {self.target_positive_ratio:.1%}")
        print(f"   Hard Mining Ratio: {self.hard_mining_ratio:.1%}")
        print(f"   Hard Mining Enabled: {self.enable_hard_mining}")
        print(f"🔧 HEADLESS CONFIGURATION:")
        print(f"   Headless Mode: {self.headless_mode}")
        print(f"   Verification: {self.verification_method}")
        print("="*80)
        
        # 체크포인트 경로 설정
        self.checkpoint_dir = Path('/content/drive/MyDrive/CoCoNut_STAR/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 시스템 구성 요소 초기화
        self._initialize_models_with_headless()
        self._initialize_controlled_replay_buffer()
        self._initialize_verification_system()
        self._initialize_basic_learning()
        
        # 학습 상태 초기화
        self.learner_step_count = 0
        self.global_dataset_index = 0
        self._initialize_enhanced_stats()
        
        # 이전 체크포인트에서 복원
        self._resume_from_latest_checkpoint()
        
        print(f"[System] 🥥 CoCoNut Improved System ready!")
        print(f"[System] Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"[System] Continual batch size: {self.continual_batch_size}")
        print(f"[System] Starting from step: {self.learner_step_count}")

    def _initialize_models_with_headless(self):
        """Headless 지원으로 모델 초기화 - 128차원 압축 지원"""
        print(f"[System] Initializing CCNet models (headless: {self.headless_mode})...")
        cfg_model = self.config.palm_recognizer
        
        compression_dim = getattr(cfg_model, 'compression_dim', 128)
        
        self.predictor_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode,
            compression_dim=compression_dim
        ).to(self.device)
        
        self.learner_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode,
            compression_dim=compression_dim
        ).to(self.device)
        
        self.feature_dimension = compression_dim if self.headless_mode else 2048
        
        # 사전 훈련된 가중치 로드
        weights_path = cfg_model.load_weights_folder
        print(f"[System] Loading pretrained weights from: {weights_path}")
        try:
            full_state_dict = torch.load(weights_path, map_location=self.device)
            
            if self.headless_mode:
                print("[System] 🔪 Removing classification head from pretrained weights...")
                filtered_state_dict = {k: v for k, v in full_state_dict.items() 
                                    if not k.startswith('arclayer_')}
                print(f"   Removed {len(full_state_dict) - len(filtered_state_dict)} head parameters")
                
                self.predictor_net.load_state_dict(filtered_state_dict, strict=False)
                self.learner_net.load_state_dict(filtered_state_dict, strict=False)
                print("[System] ✅ Headless models loaded (head removed)")
            else:
                self.predictor_net.load_state_dict(full_state_dict)
                self.learner_net.load_state_dict(full_state_dict)
                print("[System] ✅ Full models loaded (head included)")
                
        except FileNotFoundError:
            print(f"[System] ⚠️ Pretrained weights not found. Starting with random weights.")
        except Exception as e:
            print(f"[System] ❌ Failed to load weights: {e}")
            
        self.predictor_net.eval()
        self.learner_net.train()
        
        print(f"[System] 🎯 Feature dimension: {self.feature_dimension}D")

    def _initialize_controlled_replay_buffer(self):
        """🔥 NEW: 제어된 배치 구성 리플레이 버퍼 초기화"""
        print("[System] Initializing Improved Replay Buffer...")
        cfg_buffer = self.config.replay_buffer

        buffer_storage_path = Path(cfg_buffer.storage_path)
        
        self.replay_buffer = CoconutReplayBuffer(
            config=cfg_buffer,
            storage_dir=buffer_storage_path,
            feature_dimension=self.feature_dimension
        )
        
        # 특징 추출기 설정
        self.replay_buffer.set_feature_extractor(self.learner_net)
        
        # 🔥 NEW: Controlled Batch Composition 설정 전달
        self.replay_buffer.update_batch_composition_config(
            self.target_positive_ratio, 
            self.hard_mining_ratio
        )
        
        # Hard Mining 설정 전달
        self.replay_buffer.update_hard_mining_config(
            self.enable_hard_mining,
            self.hard_mining_ratio
        )
        
        # 데이터 증강 설정 전달
        cfg_augmentation = self.config.data_augmentation
        if cfg_augmentation:
            self.replay_buffer.update_augmentation_config(
                getattr(cfg_augmentation, 'enable_augmentation', False),
                cfg_augmentation
            )

    def _initialize_verification_system(self):
        """검증 시스템 초기화"""
        if self.verification_method == 'metric':
            self.verifier = HeadlessVerifier(
                metric_type=self.metric_type,
                threshold=self.similarity_threshold
            )
            print(f"[System] ✅ Metric-based verifier initialized")
        else:
            self.verifier = None
            print(f"[System] ✅ Classification-based verification")

    def _initialize_basic_learning(self):
        """기본 연속학습 시스템 초기화"""
        print("[System] 🎯 Initializing continual learning...")
        
        cfg_model = self.config.palm_recognizer
        cfg_loss = self.config.loss
        
        self.optimizer = optim.Adam(
            self.learner_net.parameters(), 
            lr=cfg_model.learning_rate
        )
        
        self.contrastive_loss = SupConLoss(
            temperature=getattr(cfg_loss, 'temp', 0.07)
        )
        
        print(f"[System] ✅ Learning system initialized")

    def _initialize_enhanced_stats(self):
        """확장된 통계 초기화"""
        self.simple_stats = {
            'total_learning_steps': 0,
            'buffer_additions': 0,
            'buffer_skips': 0,
            'losses': [],
            'processing_times': [],
            'batch_sizes': [],
            'buffer_diversity_scores': [],
            'verification_accuracies': [],
            'positive_ratios_achieved': [],
            'hard_ratios_achieved': [],
            'regular_ratios_achieved': [],
            'positive_pairs_counts': [],
            'hard_samples_counts': [],
            'regular_samples_counts': [],
            'zero_mask_incidents': [],
            'batch_compositions': []
        }

    def process_single_frame(self, image: torch.Tensor, user_id: int):
        """
        🔥 MODIFIED: 개선된 단일 프레임 처리 - 학습 먼저, 저장 나중
        """
        image = image.to(self.device)
        print(f"\\n[Process] 🎯 Processing new sample for User {user_id}")

        # 1. 예측기를 통한 실시간 인증 (기존과 동일)
        self.predictor_net.eval()
        with torch.no_grad():
            if self.headless_mode:
                _, predictor_features = self.predictor_net(image.unsqueeze(0))
                embedding_from_predictor = predictor_features.squeeze(0)
            else:
                logits, features = self.predictor_net(image.unsqueeze(0))
                embedding_from_predictor = features.squeeze(0)
        
        # 2. 학습기를 통한 최신 특징 추출 (기존과 동일)
        self.learner_net.eval()
        with torch.no_grad():
            if self.headless_mode:
                _, learner_features = self.learner_net(image.unsqueeze(0))
                latest_embedding = learner_features.squeeze(0)
            else:
                _, features = self.learner_net(image.unsqueeze(0))
                latest_embedding = features.squeeze(0)
        self.learner_net.train()
        
        # 3. 🔥 NEW: 학습 조건 확인 (버퍼 저장 전에!)
        buffer_size = len(self.replay_buffer.image_storage)
        unique_users = len(set([item['user_id'] for item in self.replay_buffer.image_storage]))
        
        print(f"[Process] 📊 Current buffer state:")
        print(f"   Buffer size: {buffer_size}")
        print(f"   Unique users: {unique_users}")
        
        # 4. 🔥 NEW: 조건 만족시 즉시 학습 (새로운 샘플 + 기존 버퍼)
        should_learn = self._should_start_learning(user_id)
        learned = False
        
        if should_learn:
            print(f"[Process] 🎯 Starting continual learning...")
            try:
                self._improved_continual_learning(image, user_id, latest_embedding)
                learned = True
            except Exception as e:
                print(f"[Process] ❌ Learning failed: {e}")
                learned = False
        else:
            print(f"[Process] ⏳ Learning conditions not met, skipping...")

        # 5. 🔥 NEW: 학습 완료 후 선별적 버퍼 저장
        buffer_size_before = len(self.replay_buffer.image_storage)
        
        try:
            storage_decision = self.replay_buffer.smart_add(image, user_id, learned=learned)
        except AttributeError:
            # smart_add가 없으면 기존 add 사용
            print("[Process] ⚠️ smart_add not available, using original add")
            self.replay_buffer.add(image, user_id)
            storage_decision = "original_add_used"
        
        buffer_size_after = len(self.replay_buffer.image_storage)

        # 6. 통계 업데이트
        if buffer_size_after > buffer_size_before:
            self.simple_stats['buffer_additions'] += 1
            print(f"[Process] ✅ Sample stored in buffer (reason: {storage_decision})")
        else:
            self.simple_stats['buffer_skips'] += 1
            print(f"[Process] ⚠️ Sample not stored (reason: {storage_decision})")

    def _should_start_learning(self, user_id: int) -> bool:
        """🔥 NEW: 학습 시작 조건 판단"""
        buffer_size = len(self.replay_buffer.image_storage)

        if buffer_size == 0:
            print(f"[Learning] 📭 Empty buffer - no learning possible")
            return False

        # 새로운 사용자인지 확인
        existing_user_samples = [item for item in self.replay_buffer.image_storage 
                               if item['user_id'] == user_id]

        if len(existing_user_samples) > 0:
            print(f"[Learning] 🎯 User {user_id} has {len(existing_user_samples)} samples in buffer - can create positive pairs!")
            return True

        # 새로운 사용자면 다른 사용자가 있는지 확인
        unique_users = len(set([item['user_id'] for item in self.replay_buffer.image_storage]))

        if unique_users >= 1:
            print(f"[Learning] 🆕 New user {user_id}, but {unique_users} other users available - can learn with negatives")
            return True
        else:
            print(f"[Learning] 🚫 New user {user_id}, no other users - cannot learn")
            return False

    def _improved_continual_learning(self, new_image, new_user_id, new_embedding):
        """🔥 NEW: 개선된 연속학습 - 새로운 샘플을 포함한 배치로 학습"""
        self.learner_step_count += 1

        print(f"[Learning] {'='*70}")
        print(f"[Learning] IMPROVED CONTINUAL STEP {self.learner_step_count}")
        print(f"[Learning] New sample: User {new_user_id}")
        print(f"[Learning] {'='*70}")

        # 배치 구성: 새로운 샘플 1개 + 리플레이 샘플들
        target_batch_size = self.continual_batch_size
        replay_count = max(0, target_batch_size - 1)  # 새로운 샘플 제외

        print(f"[Learning] 🎯 Creating batch with new sample + {replay_count} replay samples")

        # 리플레이 샘플링 (기존 방식 활용)
        if replay_count > 0:
            replay_images, replay_labels = self.replay_buffer.sample_with_replacement(
                replay_count, 
                new_embedding=new_embedding,
                current_user_id=new_user_id
            )
        else:
            replay_images, replay_labels = [], []

        # 최종 배치 구성
        all_images = [new_image] + replay_images
        all_labels = [new_user_id] + replay_labels
        actual_batch_size = len(all_images)

        print(f"[Learning] 📊 Final batch composition:")
        print(f"   Target batch size: {target_batch_size}")
        print(f"   Actual batch size: {actual_batch_size}")
        print(f"   New sample: User {new_user_id}")
        print(f"   Replay samples: {len(replay_labels)}")

        # 배치 구성 분석
        self._analyze_final_batch_composition(all_labels, actual_batch_size)

        # 연속학습 실행
        cfg_learner = self.config.continual_learner
        total_loss = 0.0

        for epoch in range(cfg_learner.adaptation_epochs):
            print(f"[Learning] 🔄 Adaptation epoch {epoch+1}/{cfg_learner.adaptation_epochs}")

            if self.headless_mode:
                epoch_loss = self._run_headless_learning_step(all_images, all_labels)
            else:
                epoch_loss = self._run_classification_learning_step(all_images, all_labels)

            total_loss += epoch_loss

        average_loss = total_loss / cfg_learner.adaptation_epochs

        print(f"[Learning] 📊 Step {self.learner_step_count} Results:")
        print(f"   Average loss: {average_loss:.6f}")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")

        # 모델 동기화 체크
        if self.learner_step_count % cfg_learner.sync_frequency == 0:
            self._sync_weights()

        return average_loss

    def _analyze_final_batch_composition(self, labels: list, batch_size: int):
        """최종 배치 구성 분석 및 Zero Mask 예측"""
        user_counts = {}
        for label in labels:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        # Positive pairs 분석
        positive_pairs = sum(1 for count in user_counts.values() if count >= 2)
        positive_samples = sum(count for count in user_counts.values() if count >= 2)
        positive_ratio = positive_samples / batch_size
        
        # Single samples 분석
        single_samples = sum(1 for count in user_counts.values() if count == 1)
        single_users = [user_id for user_id, count in user_counts.items() if count == 1]
        
        # Zero mask 예측
        zero_mask_predicted = single_samples
        
        print(f"🔍 [Analysis] Final batch composition analysis:")
        print(f"   Positive pairs: {positive_pairs} pairs ({positive_samples} samples, {positive_ratio:.1%})")
        print(f"   Single samples: {single_samples} samples ({single_samples/batch_size:.1%})")
        print(f"   Unique users: {len(user_counts)}")
        print(f"   Single users: {single_users}")
        print(f"   Zero mask predicted: {zero_mask_predicted} samples")
        print(f"   Target positive ratio: {self.target_positive_ratio:.1%}")
        print(f"   Achievement: {positive_ratio/self.target_positive_ratio:.1f}x target")
        
        # Zero mask 경고
        if zero_mask_predicted > 0:
            print(f"⚠️ [Analysis] Expected {zero_mask_predicted} zero mask warnings in SupCon loss")
        else:
            print(f"✅ [Analysis] No zero mask warnings expected!")

    def _run_headless_learning_step(self, images: list, labels: list):
        """Headless 모드 학습 스텝"""
        print(f"[Learning] 🧠 Headless learning with {len(images)} samples")
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        embeddings = []
        for i, img in enumerate(images):
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            _, embedding = self.learner_net(img)
            embeddings.append(embedding)
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)
        
        print("[Learning] 🎯 Computing SupCon loss (headless mode)...")
        loss = self.contrastive_loss(embeddings_for_loss, labels_tensor)
        
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
        
        embeddings = []
        for i, img in enumerate(images):
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            _, embedding = self.learner_net(img)
            embeddings.append(embedding)
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)
        
        print("[Learning] 🎯 Computing SupCon loss (classification mode)...")
        loss = self.contrastive_loss(embeddings_for_loss, labels_tensor)
        
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            print("[Learning] ✅ Classification gradient update completed")
        else:
            print("[Learning] ⚠️ No gradient - loss computation issue")
        
        print(f"[Learning] ✅ Classification Loss: {loss.item():.6f}")
        return loss.item()

    def _sync_weights(self):
        """가중치 동기화"""
        self.predictor_net.load_state_dict(self.learner_net.state_dict())
        self.predictor_net.eval()
        
        print(f"\\n[Sync] 🔄 MODEL SYNCHRONIZATION ({'Headless' if self.headless_mode else 'Classification'})")
        print(f"[Sync] {'='*60}")
        print(f"[Sync] ✅ Predictor updated at step {self.learner_step_count}!")
        print(f"[Sync] {'='*60}\\n")

    def run_experiment(self):
        """연속학습 실험 실행"""
        print(f"[System] Starting improved continual learning from step {self.learner_step_count}...")
        print(f"[System] Batch configuration: {self.continual_batch_size} samples with {self.target_positive_ratio:.1%} positive, {self.hard_mining_ratio:.1%} hard")

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
        
        for data_offset, (datas, user_id) in enumerate(tqdm(remaining_data, desc="Improved Continual Learning")):
            
            # 전체 데이터셋에서의 현재 위치 업데이트
            self.global_dataset_index = self.global_dataset_index + data_offset
            
            primary_image = datas[0].squeeze(0)
            user_id = user_id.item()

            # 🔥 개선된 프레임 처리
            self.process_single_frame(primary_image, user_id)

            # 설정된 빈도에 따라 체크포인트 저장
            save_frequency = getattr(self.config.continual_learner, 'intermediate_save_frequency', 50)
            if save_frequency > 0 and self.learner_step_count > 0 and self.learner_step_count % save_frequency == 0:
                self._save_complete_checkpoint()

        # 마지막 데이터 처리 후 인덱스 업데이트
        self.global_dataset_index = total_steps

        # 실험 종료 후 최종 체크포인트 저장
        print(f"\\n[System] Improved continual learning experiment finished.")
        print(f"[System] Final stats summary:")
        self._print_final_stats_summary()
        
        self._save_complete_checkpoint()
        self.save_system_state()

    def _print_final_stats_summary(self):
        """최종 통계 요약 출력"""
        if len(self.simple_stats['positive_ratios_achieved']) == 0:
            print("[Stats] No learning steps completed yet.")
            return
        
        # 평균 통계 계산
        avg_positive_ratio = np.mean(self.simple_stats['positive_ratios_achieved'])
        avg_positive_pairs = np.mean(self.simple_stats['positive_pairs_counts'])
        avg_zero_mask = np.mean(self.simple_stats['zero_mask_incidents'])
        total_steps = self.simple_stats['total_learning_steps']
        
        print(f"📊 [Final Stats] Improved Batch Composition Results:")
        print(f"   Total learning steps: {total_steps}")
        print(f"   Average positive ratio: {avg_positive_ratio:.1%} (target: {self.target_positive_ratio:.1%})")
        print(f"   Average positive pairs: {avg_positive_pairs:.1f}")
        print(f"   Average zero mask incidents: {avg_zero_mask:.1f}")
        print(f"   Achievement rate: {avg_positive_ratio/self.target_positive_ratio:.1f}x target")

    def _save_complete_checkpoint(self):
        """완전한 체크포인트 저장"""
        step = self.learner_step_count
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'step_count': step,
            'global_dataset_index': self.global_dataset_index,
            'timestamp': timestamp,
            'learner_state_dict': self.learner_net.state_dict(),
            'predictor_state_dict': self.predictor_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'simple_stats': self.simple_stats,
            'headless_mode': self.headless_mode,
            'verification_method': self.verification_method,
            'continual_batch_size': self.continual_batch_size,
            'target_positive_ratio': self.target_positive_ratio,
            'hard_mining_ratio': self.hard_mining_ratio,
            'enable_hard_mining': self.enable_hard_mining,
            'config_info': {
                'continual_batch_size': self.continual_batch_size,
                'target_positive_ratio': self.target_positive_ratio,
                'hard_mining_ratio': self.hard_mining_ratio,
                'learning_rate': self.config.palm_recognizer.learning_rate,
                'loss_temperature': getattr(self.config.loss, 'temp', 0.07),
                'headless_mode': self.headless_mode,
                'verification_method': self.verification_method,
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        print(f"[Checkpoint] 💾 Complete checkpoint saved:")
        print(f"   📁 Model: checkpoint_step_{step}.pth")

    def save_system_state(self):
        """시스템 상태 저장"""
        custom_save_path = Path('/content/drive/MyDrive/CoCoNut_STAR')
        custom_save_path.mkdir(parents=True, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "headless" if self.headless_mode else "classification"
        batch_suffix = f"batch{self.continual_batch_size}"
        ratio_suffix = f"pos{int(self.target_positive_ratio*100)}hard{int(self.hard_mining_ratio*100)}"
        
        # 상세한 파일명으로 구분
        custom_learner_path = custom_save_path / f'coconut_improved_{mode_suffix}_{batch_suffix}_{ratio_suffix}_model_{timestamp}.pth'
        custom_predictor_path = custom_save_path / f'coconut_improved_{mode_suffix}_{batch_suffix}_{ratio_suffix}_predictor_{timestamp}.pth'
        
        torch.save(self.learner_net.state_dict(), custom_learner_path)
        torch.save(self.predictor_net.state_dict(), custom_predictor_path)
        
        print(f"[System] ✅ CoCoNut Improved 모델 저장 완료:")
        print(f"  🎯 사용자 지정 경로: {custom_save_path}")
        print(f"  📁 Learner 모델: {custom_learner_path.name}")
        print(f"  📁 Predictor 모델: {custom_predictor_path.name}")

    def _resume_from_latest_checkpoint(self):
        """최신 체크포인트에서 복원"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_step_*.pth'))
        
        if not checkpoint_files:
            print("[Resume] 📂 No checkpoints found - starting fresh")
            return
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        step_num = int(latest_checkpoint.stem.split('_')[-1])
        
        print(f"[Resume] 🔄 Found checkpoint: {latest_checkpoint.name}")
        print(f"[Resume] 📍 Resuming from step: {step_num}")
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            # Headless 모드에 맞는 state_dict 필터링
            learner_state_dict = checkpoint['learner_state_dict']
            predictor_state_dict = checkpoint['predictor_state_dict']
            
            if self.headless_mode:
                print("[Resume] 🔪 Filtering out classification head from checkpoint...")
                learner_filtered = {k: v for k, v in learner_state_dict.items() 
                                  if not k.startswith('arclayer_')}
                predictor_filtered = {k: v for k, v in predictor_state_dict.items() 
                                    if not k.startswith('arclayer_')}
                
                removed_count = len(learner_state_dict) - len(learner_filtered)
                print(f"   Removed {removed_count} classification head parameters")
                
                self.learner_net.load_state_dict(learner_filtered, strict=False)
                self.predictor_net.load_state_dict(predictor_filtered, strict=False)
            else:
                self.learner_net.load_state_dict(learner_state_dict)
                self.predictor_net.load_state_dict(predictor_state_dict)
            
            # 옵티마이저 상태 복원
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 학습 상태 복원
            self.learner_step_count = checkpoint['step_count']
            self.global_dataset_index = checkpoint.get('global_dataset_index', 0)
            self.simple_stats = checkpoint.get('simple_stats', self.simple_stats)
            
            print(f"[Resume] ✅ Successfully resumed from step {self.learner_step_count}")
        
        except Exception as e:
            print(f"[Resume] ❌ Failed to resume: {e}")
            print(f"[Resume] 🔄 Starting fresh instead")
            self.learner_step_count = 0
            self.global_dataset_index = 0

print("✅ CoconutSystem Improved Version 완료!")