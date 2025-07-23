"""
=== COCONUT STAGE 2: CONTINUAL LEARNING WITH INTELLIGENT REPLAY BUFFER ===

DESIGN RATIONALE (단순화됨):
1. Focus on Replay Buffer innovation only
2. Use basic SupCon loss for stable continual learning  
3. Maintain checkpoint resume capability
4. Remove all W2ML complexity for clear paper contribution

CORE CONTRIBUTION:
- Diversity-based Intelligent Replay Buffer with Faiss acceleration
- True continual learning with complete state preservation
- Memory-efficient buffer management without catastrophic forgetting
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

from models.ccnet_model import ccnet
from framework.replay_buffer import CoconutReplayBuffer
from loss import SupConLoss  # 기존 CCNet의 단순한 SupConLoss 사용
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class CoconutSystem:
    def __init__(self, config):
        """
        Continual Learning CoCoNut System (단순화됨)
        
        FOCUS: Intelligent Replay Buffer for continual palmprint recognition
        """
        print("="*80)
        print("🥥 COCONUT STAGE 2: INTELLIGENT REPLAY BUFFER")
        print("="*80)
        print("🎯 CORE INNOVATION:")
        print("   - Diversity-based Replay Buffer with Faiss acceleration")
        print("   - True continual learning with checkpoint resume")
        print("   - Memory-efficient sample selection")
        print("   - Basic SupCon loss for stable learning")
        print("="*80)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] Using device: {self.device}")
        print(f"[System] Faiss status: {'Available' if FAISS_AVAILABLE else 'Fallback mode'}")
        
        # 체크포인트 경로 설정
        self.checkpoint_dir = Path('/content/drive/MyDrive/CoCoNut_STAR/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 시스템 구성 요소 초기화
        self._initialize_models()
        self._initialize_replay_buffer()
        self._initialize_basic_learning()
        
        # 학습 상태 초기화
        self.learner_step_count = 0
        self.global_dataset_index = 0
        self._initialize_simple_stats()
        
        # 이전 체크포인트에서 복원
        self._resume_from_latest_checkpoint()
        
        print(f"[System] 🥥 CoCoNut ready!")
        print(f"[System] Starting from step: {self.learner_step_count}")
        print(f"[System] Dataset position: {self.global_dataset_index}")

    def _initialize_models(self):
        """예측기와 학습기 모델을 생성하고 사전 훈련된 가중치를 로드합니다."""
        print("[System] Initializing CCNet models...")
        cfg_model = self.config.palm_recognizer
        
        # 모델 아키텍처 생성
        self.predictor_net = ccnet(num_classes=cfg_model.num_classes, weight=cfg_model.com_weight).to(self.device)
        self.learner_net = ccnet(num_classes=cfg_model.num_classes, weight=cfg_model.com_weight).to(self.device)
        
        # 사전 훈련된 가중치 로드
        weights_path = cfg_model.load_weights_folder
        print(f"[System] Loading pretrained weights from: {weights_path}")
        try:
            self.predictor_net.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.learner_net.load_state_dict(self.predictor_net.state_dict())
            print("[System] ✅ Successfully loaded pretrained weights (Stage 1 → Stage 2)")
        except FileNotFoundError:
            print(f"[System] ⚠️ Pretrained weights not found. Starting with random weights.")
        except Exception as e:
            print(f"[System] ❌ Failed to load weights: {e}")
            
        self.predictor_net.eval()  # 추론용
        self.learner_net.train()   # 학습용

    def _initialize_replay_buffer(self):
        """리플레이 버퍼를 초기화합니다."""
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

    def _initialize_basic_learning(self):
        """기본 연속학습 시스템 초기화 (W2ML 제거)"""
        print("[System] 🎯 Initializing basic continual learning...")
        
        cfg_model = self.config.palm_recognizer
        cfg_loss = self.config.loss
        
        # Adam 옵티마이저
        self.optimizer = optim.Adam(
            self.learner_net.parameters(), 
            lr=cfg_model.learning_rate
        )
        
        # 기본 SupCon 손실 함수
        self.contrastive_loss = SupConLoss(
            temperature=getattr(cfg_loss, 'temp', 0.07)
        )
        
        print("[System] ✅ Basic learning system initialized")
        print(f"[System] Optimizer: Adam (lr={cfg_model.learning_rate})")
        print(f"[System] Loss: SupConLoss (temp={getattr(cfg_loss, 'temp', 0.07)})")
        print(f"[System] Batch size: {cfg_model.batch_size}")

    def _initialize_simple_stats(self):
        """단순한 통계 초기화 (W2ML 통계 제거)"""
        self.simple_stats = {
            'total_learning_steps': 0,
            'buffer_additions': 0,
            'buffer_skips': 0,
            'losses': [],
            'processing_times': [],
            'batch_sizes': [],
            'buffer_diversity_scores': []
        }

    def _resume_from_latest_checkpoint(self):
        """체크포인트에서 시스템 복원"""
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
            
            # 모델 상태 복원
            self.learner_net.load_state_dict(checkpoint['learner_state_dict'])
            self.predictor_net.load_state_dict(checkpoint['predictor_state_dict'])
            
            # 옵티마이저 상태 복원
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 학습 상태 복원
            self.learner_step_count = checkpoint['step_count']
            self.global_dataset_index = checkpoint.get('global_dataset_index', 0)
            self.simple_stats = checkpoint.get('simple_stats', self.simple_stats)
            
            # 리플레이 버퍼 상태 복원
            buffer_checkpoint = self.checkpoint_dir / f'buffer_step_{step_num}.pkl'
            if buffer_checkpoint.exists():
                with open(buffer_checkpoint, 'rb') as f:
                    buffer_data = pickle.load(f)
                    self.replay_buffer.image_storage = buffer_data['image_storage']
                    self.replay_buffer.stored_embeddings = buffer_data.get('stored_embeddings', [])
                    self.replay_buffer.metadata = buffer_data['metadata']
                    if buffer_data.get('faiss_index_data'):
                        self.replay_buffer.faiss_index = faiss.deserialize_index(buffer_data['faiss_index_data'])
                    else:
                        self.replay_buffer.faiss_index = None
            
            print(f"[Resume] ✅ Successfully resumed from step {self.learner_step_count}")
            print(f"   Buffer size: {len(self.replay_buffer.image_storage)}")
            print(f"   Dataset position: {self.global_dataset_index}")
        
        except Exception as e:
            print(f"[Resume] ❌ Failed to resume: {e}")
            print(f"[Resume] 🔄 Starting fresh instead")
            self.learner_step_count = 0
            self.global_dataset_index = 0

    def run_experiment(self):
        """CoCoNut Stage 2 연속학습 실험 실행 (단순화됨)"""
        print(f"[System] Starting continual learning from step {self.learner_step_count}...")

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

            # 중간 결과 로깅
            if self.learner_step_count % 100 == 0 and self.learner_step_count > 0:
                self._log_progress(self.global_dataset_index, total_steps)

        # 마지막 데이터 처리 후 인덱스 업데이트
        self.global_dataset_index = total_steps

        # 실험 종료 후 최종 체크포인트 저장
        print("\n[System] Continual Learning experiment finished.")
        self._save_complete_checkpoint()
        self._final_analysis()
        self.save_system_state()

    def process_single_frame(self, image: torch.Tensor, user_id: int):
        """
        단일 프레임 처리 (단순화됨)
        
        SIMPLIFIED PROCESS:
        1. Extract features using learner network
        2. Add to intelligent replay buffer (diversity-based)
        3. Perform basic continual learning if conditions met
        """
        image = image.to(self.device)

        # 1. 예측기를 통한 실시간 인증
        self.predictor_net.eval()
        with torch.no_grad():
            embedding_from_predictor = self.predictor_net.getFeatureCode(image)
        
        # 2. 학습기를 통한 최신 특징 추출
        self.learner_net.eval()
        with torch.no_grad():
            latest_embedding = self.learner_net.getFeatureCode(image)
        self.learner_net.train()
        
        # 3. 리플레이 버퍼에 추가 (다양성 기반)
        buffer_size_before = len(self.replay_buffer.image_storage)
        self.replay_buffer.add(image, user_id)
        buffer_size_after = len(self.replay_buffer.image_storage)
        
        # 추가 여부 통계 업데이트
        if buffer_size_after > buffer_size_before:
            self.simple_stats['buffer_additions'] += 1
        else:
            self.simple_stats['buffer_skips'] += 1
        
        # 4. 현재 버퍼 상태 확인
        buffer_size = len(self.replay_buffer.image_storage)
        unique_users = len(set([item['user_id'] for item in self.replay_buffer.image_storage]))
        
        # 최소 조건: 2명 이상의 사용자 (대조학습을 위한 최소 다양성)
        if unique_users < 2:
            print(f"[Learning] 📊 Waiting for diversity (Dataset pos: {self.global_dataset_index}):")
            print(f"   Buffer size: {buffer_size}")
            print(f"   Unique users: {unique_users}/2 minimum")
            return
        
        # 5. 첫 번째 학습 시작 알림
        if unique_users == 2 and buffer_size <= 3:
            print(f"\n🎉 [Learning] CONTINUAL LEARNING ACTIVATED!")
            print(f"   Minimum diversity achieved: {unique_users} users")
            print(f"   Target batch size: {self.config.palm_recognizer.batch_size}")
        
        # 6. 기본 연속학습 실행
        self._basic_continual_learning(image, user_id)

    def _basic_continual_learning(self, new_image, new_user_id):
        """
        기본 연속학습 수행 (W2ML 복잡성 제거)
        
        SIMPLIFIED LEARNING:
        1. Create batch with new sample + replay samples
        2. Extract embeddings using learner network
        3. Compute basic SupCon loss
        4. Perform gradient update
        """
        
        # 학습 스텝 증가
        self.learner_step_count += 1
        
        print(f"[Learning] {'='*50}")
        print(f"[Learning] BASIC CONTINUAL STEP {self.learner_step_count}")
        print(f"[Learning] {'='*50}")
        
        cfg_learner = self.config.continual_learner
        cfg_model = self.config.palm_recognizer
        target_batch_size = cfg_model.batch_size

        # 배치 구성: 새 이미지 1장 + 버퍼에서 나머지
        replay_count = target_batch_size - 1
        replay_images, replay_labels = self.replay_buffer.sample_with_replacement(replay_count)
        
        all_images = [new_image] + replay_images
        all_labels = [new_user_id] + replay_labels
        
        actual_batch_size = len(all_images)
        
        print(f"[Learning] Batch Analysis:")
        print(f"   Target batch size: {target_batch_size}")
        print(f"   Actual batch size: {actual_batch_size}")
        print(f"   Current user: {new_user_id}")
        print(f"   Replay samples: {len(replay_images)}")
        
        # 다양성 분석
        unique_users = len(set(all_labels))
        user_distribution = {}
        for label in all_labels:
            user_distribution[label] = user_distribution.get(label, 0) + 1
        
        print(f"   Unique users: {unique_users}")
        print(f"   User distribution: {dict(sorted(user_distribution.items()))}")
        
        # 연속학습 에포크들 실행
        total_loss = 0.0
        processing_start = time.time()
        
        for epoch in range(cfg_learner.adaptation_epochs):
            print(f"[Learning] 🔄 Adaptation epoch {epoch+1}/{cfg_learner.adaptation_epochs}")
            
            epoch_loss = self._run_basic_learning_step(all_images, all_labels)
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
        print(f"   Buffer diversity: {diversity_stats['diversity_score']:.3f}")
        print(f"   Buffer size: {diversity_stats['total_samples']}")
        print(f"   Unique users: {diversity_stats['unique_users']}")
        
        # 모델 동기화 체크
        if self.learner_step_count % cfg_learner.sync_frequency == 0:
            self._sync_weights()

    def _run_basic_learning_step(self, images: list, labels: list):
        """
        기본 학습 스텝 수행 (W2ML 복잡성 제거)
        
        SIMPLE LEARNING:
        1. Extract embeddings from all images
        2. Compute basic SupCon loss
        3. Perform gradient update
        """
        
        print(f"[Learning] 🧠 Processing {len(images)} samples with basic SupCon")
        
        # 1. 학습을 위해 train 모드 설정
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # 2. 임베딩 추출
        embeddings = []
        for i, img in enumerate(images):
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # Forward pass with gradient computation
            embedding = self.learner_net.getFeatureCode(img)
            embeddings.append(embedding)
        
        # 3. 배치 텐서 구성
        embeddings_tensor = torch.cat(embeddings, dim=0)  # [batch_size, feature_dim]
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # 4. SupCon 손실 계산
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        print("[Learning] 🎯 Computing basic SupCon loss...")
        
        # 기본 SupCon 손실 계산
        loss = self.contrastive_loss(embeddings_for_loss, labels_tensor)
        
        # 5. 역전파
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            print("[Learning] ✅ Gradient update completed")
        else:
            print("[Learning] ⚠️ No gradient - loss computation issue")
        
        print(f"[Learning] ✅ Basic Loss: {loss.item():.6f}")
        
        return loss.item()

    def _sync_weights(self):
        """학습기의 가중치를 예측기로 복사 (단순화됨)"""
        
        self.predictor_net.load_state_dict(self.learner_net.state_dict())
        self.predictor_net.eval()
        
        print(f"\n[Sync] 🔄 MODEL SYNCHRONIZATION")
        print(f"[Sync] {'='*50}")
        
        # 최근 성능 분석
        recent_steps = min(10, len(self.simple_stats['losses']))
        if recent_steps > 0:
            recent_losses = self.simple_stats['losses'][-recent_steps:]
            recent_diversity = self.simple_stats['buffer_diversity_scores'][-recent_steps:]
            recent_batch_sizes = self.simple_stats['batch_sizes'][-recent_steps:]
            
            avg_loss = sum(recent_losses) / len(recent_losses)
            avg_diversity = sum(recent_diversity) / len(recent_diversity)
            avg_batch_size = sum(recent_batch_sizes) / len(recent_batch_sizes)
            
            print(f"[Sync] 📊 Recent {recent_steps} steps analysis:")
            print(f"   Average loss: {avg_loss:.6f}")
            print(f"   Average diversity: {avg_diversity:.3f}")
            print(f"   Average batch size: {avg_batch_size:.1f}")
            print(f"   Total buffer additions: {self.simple_stats['buffer_additions']}")
            print(f"   Total buffer skips: {self.simple_stats['buffer_skips']}")
            
        print(f"[Sync] ✅ Predictor updated!")
        print(f"[Sync] {'='*50}\n")

    def _save_complete_checkpoint(self):
        """완전한 체크포인트 저장"""
        
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
            'config_info': {
                'batch_size': self.config.palm_recognizer.batch_size,
                'learning_rate': self.config.palm_recognizer.learning_rate,
                'loss_temperature': getattr(self.config.loss, 'temp', 0.07),
            }
        }
        
        # 메인 체크포인트 저장
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 리플레이 버퍼 상태 저장
        buffer_data = {
            'image_storage': self.replay_buffer.image_storage,
            'stored_embeddings': getattr(self.replay_buffer, 'stored_embeddings', []),
            'metadata': self.replay_buffer.metadata,
            'faiss_index_data': faiss.serialize_index(self.replay_buffer.faiss_index) if self.replay_buffer.faiss_index else None
        }
        buffer_path = self.checkpoint_dir / f'buffer_step_{step}.pkl'
        with open(buffer_path, 'wb') as f:
            pickle.dump(buffer_data, f)
        
        # 상세 통계 저장
        stats_data = {
            'step': step,
            'global_dataset_index': self.global_dataset_index,
            'timestamp': timestamp,
            'learning_progress': {
                'total_steps': self.simple_stats['total_learning_steps'],
                'buffer_additions': self.simple_stats['buffer_additions'],
                'buffer_skips': self.simple_stats['buffer_skips'],
                'avg_loss': sum(self.simple_stats['losses'][-10:]) / min(10, len(self.simple_stats['losses'])) if self.simple_stats['losses'] else 0,
                'avg_diversity': sum(self.simple_stats['buffer_diversity_scores'][-10:]) / min(10, len(self.simple_stats['buffer_diversity_scores'])) if self.simple_stats['buffer_diversity_scores'] else 0,
            },
            'buffer_status': {
                'size': len(self.replay_buffer.image_storage),
                'diversity': len(set([item['user_id'] for item in self.replay_buffer.image_storage])),
                'max_size': self.replay_buffer.buffer_size
            }
        }
        
        stats_path = self.checkpoint_dir / f'stats_step_{step}.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        # 오래된 체크포인트 정리 (최근 5개만 유지)
        self._cleanup_old_checkpoints()
        
        print(f"[Checkpoint] 💾 Complete checkpoint saved:")
        print(f"   📁 Model: checkpoint_step_{step}.pth")
        print(f"   📁 Buffer: buffer_step_{step}.pkl") 
        print(f"   📁 Stats: stats_step_{step}.json")
        print(f"   📍 Dataset position: {self.global_dataset_index}")

    def _cleanup_old_checkpoints(self, keep_last=5):
        """오래된 체크포인트들 정리"""
        
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_step_*.pth'))
        if len(checkpoint_files) <= keep_last:
            return
        
        # 스텝 번호로 정렬하고 오래된 것들 삭제
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        files_to_delete = checkpoint_files[:-keep_last]
        
        for file_path in files_to_delete:
            step_num = int(file_path.stem.split('_')[-1])
            
            # 관련 파일들 모두 삭제
            file_path.unlink()  # checkpoint_step_X.pth
            
            buffer_file = self.checkpoint_dir / f'buffer_step_{step_num}.pkl'
            if buffer_file.exists():
                buffer_file.unlink()
                
            stats_file = self.checkpoint_dir / f'stats_step_{step_num}.json'
            if stats_file.exists():
                stats_file.unlink()
        
        print(f"[Cleanup] 🗑️ Cleaned up {len(files_to_delete)} old checkpoints")

    def _log_progress(self, step, total_steps):
        """진행 상황 로깅 (단순화됨)"""
        
        print(f"\n[Progress] Step {step}/{total_steps} ({step/total_steps*100:.1f}%)")
        
        if len(self.simple_stats['losses']) > 0:
            recent_loss = self.simple_stats['losses'][-1]
            print(f"  - Recent loss: {recent_loss:.6f}")
        
        if len(self.simple_stats['buffer_diversity_scores']) > 0:
            recent_diversity = self.simple_stats['buffer_diversity_scores'][-1]
            print(f"  - Buffer diversity: {recent_diversity:.3f}")
        
        print(f"  - Total learning steps: {self.simple_stats['total_learning_steps']}")
        print(f"  - Buffer additions: {self.simple_stats['buffer_additions']}")
        print(f"  - Buffer skips: {self.simple_stats['buffer_skips']}")

    def _final_analysis(self):
        """최종 분석 (단순화됨)"""
        
        print("\n" + "="*80)
        print("FINAL CONTINUAL LEARNING ANALYSIS")
        print("="*80)
        
        total_steps = self.simple_stats['total_learning_steps']
        total_additions = self.simple_stats['buffer_additions']
        total_skips = self.simple_stats['buffer_skips']
        
        if total_steps > 0:
            avg_loss = sum(self.simple_stats['losses']) / len(self.simple_stats['losses'])
            avg_diversity = sum(self.simple_stats['buffer_diversity_scores']) / len(self.simple_stats['buffer_diversity_scores'])
            avg_batch_size = sum(self.simple_stats['batch_sizes']) / len(self.simple_stats['batch_sizes'])
            
            print(f"📊 Continual Learning Statistics:")
            print(f"   🔄 Total adaptation steps: {total_steps}")
            print(f"   💡 Average loss: {avg_loss:.6f}")
            print(f"   🎯 Average diversity: {avg_diversity:.3f}")
            print(f"   📏 Average batch size: {avg_batch_size:.1f}")
            print(f"   ✅ Buffer additions: {total_additions}")
            print(f"   ⚠️ Buffer skips: {total_skips}")
            print(f"   📍 Final dataset position: {self.global_dataset_index}")
            
            # 버퍼 효율성 분석
            total_processed = total_additions + total_skips
            addition_rate = (total_additions / total_processed * 100) if total_processed > 0 else 0
            print(f"   📈 Buffer addition rate: {addition_rate:.1f}%")
            
            # 배치 크기 달성률
            target_batch_size = self.config.palm_recognizer.batch_size
            batch_size_achievement = (avg_batch_size / target_batch_size) * 100
            print(f"   🎯 Batch size achievement: {batch_size_achievement:.1f}%")
            
            # 최종 평가
            print(f"\n🔬 Intelligent Replay Buffer Performance:")
            print(f"   📖 Diversity-based selection: ✅ {'Excellent' if addition_rate < 70 else 'Good' if addition_rate < 80 else 'Moderate'}")
            print(f"   🚀 Faiss acceleration: ✅ {'Active' if FAISS_AVAILABLE else 'Fallback mode'}")
            print(f"   🎯 Batch size consistency: ✅ {'Excellent' if batch_size_achievement > 95 else 'Good'}")
            print(f"   🔄 Checkpoint resume: ✅ Implemented")
            
            if addition_rate < 70 and batch_size_achievement > 95:
                print(f"   🎉 INTELLIGENT REPLAY BUFFER: EXCELLENT PERFORMANCE!")
            elif addition_rate < 80 and batch_size_achievement > 90:
                print(f"   ✅ INTELLIGENT REPLAY BUFFER: GOOD PERFORMANCE")
            else:
                print(f"   🔧 INTELLIGENT REPLAY BUFFER: NEEDS OPTIMIZATION")
                
        print("="*80)

    def save_system_state(self):
        """시스템 상태 저장 (최종 호출용)"""
        
        # 사용자 지정 저장 경로
        custom_save_path = Path('/content/drive/MyDrive/CoCoNut_STAR')
        custom_save_path.mkdir(parents=True, exist_ok=True)
        
        # 기본 저장 경로도 유지
        storage_path = Path(self.config.replay_buffer.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # 최종 학습된 모델을 사용자 지정 경로에 저장
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 사용자 지정 경로에 저장
        custom_learner_path = custom_save_path / f'coconut_replay_model_{timestamp}.pth'
        custom_predictor_path = custom_save_path / f'coconut_predictor_model_{timestamp}.pth'
        
        torch.save(self.learner_net.state_dict(), custom_learner_path)
        torch.save(self.predictor_net.state_dict(), custom_predictor_path)
        
        # 기본 경로에도 저장 (호환성)
        learner_path = storage_path / 'coconut_replay_learner.pth'
        predictor_path = storage_path / 'coconut_replay_predictor.pth'
        torch.save(self.learner_net.state_dict(), learner_path)
        torch.save(self.predictor_net.state_dict(), predictor_path)
        
        # 통계를 사용자 지정 경로에 저장
        custom_stats_path = custom_save_path / f'coconut_replay_stats_{timestamp}.json'
        stats_path = storage_path / 'coconut_replay_stats.json'
        
        import json
        stats_to_save = {
            'total_learning_steps': self.simple_stats['total_learning_steps'],
            'buffer_additions': self.simple_stats['buffer_additions'],
            'buffer_skips': self.simple_stats['buffer_skips'],
            'losses': self.simple_stats['losses'],
            'processing_times': self.simple_stats['processing_times'],
            'batch_sizes': self.simple_stats['batch_sizes'],
            'buffer_diversity_scores': self.simple_stats['buffer_diversity_scores'],
            # 설정 정보
            'target_batch_size': self.config.palm_recognizer.batch_size,
            'loss_temperature': getattr(self.config.loss, 'temp', 0.07),
            'faiss_available': FAISS_AVAILABLE,
            'gpu_available': torch.cuda.is_available(),
            # 추가 정보
            'save_timestamp': timestamp,
            'total_adaptation_steps': self.learner_step_count,
            'final_dataset_position': self.global_dataset_index,
            'model_architecture': 'CCNet',
            'loss_function': 'SupConLoss',
            'core_innovation': 'Intelligent Replay Buffer',
            'continual_learning': True,
            'checkpoint_resume': True
        }
        
        # 양쪽 경로에 통계 저장
        with open(custom_stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        # README 파일 생성
        readme_content = f"""# CoCoNut Intelligent Replay Buffer Trained Model

## 모델 정보
- 저장 시간: {timestamp}
- 총 적응 스텝: {self.learner_step_count}
- 데이터셋 완료: {self.global_dataset_index}개 처리
- 버퍼 추가: {self.simple_stats['buffer_additions']}개
- 버퍼 스킵: {self.simple_stats['buffer_skips']}개
- 아키텍처: CCNet with Intelligent Replay Buffer
- 체크포인트 복원: 지원됨

## 핵심 기여
- **Diversity-based Replay Strategy**: Faiss 가속 유사도 기반 샘플 선택
- **Intelligent Buffer Management**: 중복 샘플 자동 제거
- **True Continual Learning**: 체크포인트 기반 중단/재개 시스템

## 파일 설명
- `coconut_replay_model_{timestamp}.pth`: 최종 학습된 모델 (learner)
- `coconut_predictor_model_{timestamp}.pth`: 예측용 모델 (predictor)
- `coconut_replay_stats_{timestamp}.json`: 학습 통계 및 성능 지표

## 모델 로드 방법

```python
import torch
from models.ccnet_model import ccnet

# 모델 아키텍처 생성
model = ccnet(num_classes=600, weight=0.8)

# 학습된 가중치 로드
model.load_state_dict(torch.load('coconut_replay_model_{timestamp}.pth'))
model.eval()

# 특징 추출 사용 예시
with torch.no_grad():
    features = model.getFeatureCode(input_image)
```

## 성능 정보
- 총 학습 스텝: {self.simple_stats['total_learning_steps']}
- 버퍼 추가율: {self.simple_stats['buffer_additions']/(self.simple_stats['buffer_additions']+self.simple_stats['buffer_skips'])*100 if (self.simple_stats['buffer_additions']+self.simple_stats['buffer_skips']) > 0 else 0:.1f}%
- Faiss 최적화: {'사용' if FAISS_AVAILABLE else '미사용'}
- 체크포인트 위치: Step {self.learner_step_count}, Data {self.global_dataset_index}

## 연속 학습 재개 방법
```python
# 새로운 데이터로 학습 재개
from framework.coconut import CoconutSystem

system = CoconutSystem(config)  # 자동으로 마지막 체크포인트에서 복원
system.run_experiment()  # 중단된 지점부터 이어서 학습
```

Generated by CoCoNut Intelligent Replay Buffer System
Supports checkpoint resume and never loses progress!
"""
        
        readme_path = custom_save_path / f'README_coconut_{timestamp}.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"[System] ✅ CoCoNut Replay Buffer 모델 저장 완료:")
        print(f"  🎯 사용자 지정 경로: {custom_save_path}")
        print(f"  📁 Learner 모델: {custom_learner_path.name}")
        print(f"  📁 Predictor 모델: {custom_predictor_path.name}")
        print(f"  📊 통계 파일: {custom_stats_path.name}")
        print(f"  📖 README: {readme_path.name}")
        print(f"  🕐 타임스탬프: {timestamp}")
        print(f"  📈 총 적응 스텝: {self.learner_step_count}")
        print(f"  📍 데이터셋 완료: {self.global_dataset_index}")
        print(f"\n[System] 🎉 COCONUT INTELLIGENT REPLAY BUFFER completed!")
        print(f"[System] 🥥 Continual learning with intelligent diversity-based buffer!")
        print(f"[System] 💾 Models saved to: /content/drive/MyDrive/CoCoNut_STAR")
        print(f"[System] 🔄 Next run will auto-resume from checkpoints!")