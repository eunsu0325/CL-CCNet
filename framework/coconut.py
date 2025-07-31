# framework/coconut.py - 배치 기반 재설계 버전

"""
=== COCONUT STAGE 2: BATCH-BASED CONTINUAL LEARNING ===

🔥 MAJOR CHANGES:
- Single sample processing → Batch processing
- Removed positive pair forcing logic
- Simplified learning flow
- Better GPU utilization
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
from collections import defaultdict

from models.ccnet_model import ccnet, HeadlessVerifier
from framework.replay_buffer import SimplifiedReplayBuffer
from .losses import SupConLoss
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class BatchCoconutSystem:
    def __init__(self, config):
        """
        배치 기반 CoCoNut 연속학습 시스템
        
        DESIGN:
        - Process samples in batches per label
        - Automatic positive pairs from batch
        - Simple hard negative mining
        - Efficient GPU utilization
        """
        print("="*80)
        print("🥥 COCONUT STAGE 2: BATCH-BASED CONTINUAL LEARNING")
        print("="*80)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Configuration
        self.headless_mode = getattr(config.palm_recognizer, 'headless_mode', False)
        self.verification_method = getattr(config.palm_recognizer, 'verification_method', 'classification')
        self.metric_type = getattr(config.palm_recognizer, 'metric_type', 'cosine')
        self.similarity_threshold = getattr(config.palm_recognizer, 'similarity_threshold', 0.5)
        
        # Batch configuration
        cfg_learner = self.config.continual_learner
        self.training_batch_size = getattr(cfg_learner, 'training_batch_size', 32)
        self.hard_negative_ratio = getattr(cfg_learner, 'hard_negative_ratio', 0.3)
        self.samples_per_label = getattr(self.config.dataset, 'samples_per_label', 5)
        
        print(f"🔧 BATCH CONFIGURATION:")
        print(f"   Samples per label: {self.samples_per_label}")
        print(f"   Training batch size: {self.training_batch_size}")
        print(f"   Hard negative ratio: {self.hard_negative_ratio:.1%}")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print("="*80)
        
        # Checkpoint directory
        self.checkpoint_dir = Path('/content/drive/MyDrive/CoCoNut_BatchMode/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_models()
        self._initialize_replay_buffer()
        self._initialize_verification_system()
        self._initialize_optimizer()
        
        # Learning state
        self.global_step = 0
        self.processed_users = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        print(f"[System] 🥥 Batch CoCoNut System ready!")
        print(f"[System] Starting from step: {self.global_step}")

    def _initialize_models(self):
        """모델 초기화 (단순화)"""
        print(f"[System] Initializing models...")
        cfg_model = self.config.palm_recognizer
        
        compression_dim = getattr(cfg_model, 'compression_dim', 128)
        
        # Predictor (frozen after sync)
        self.predictor_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode,
            compression_dim=compression_dim
        ).to(self.device)
        
        # Learner (continuously updated)
        self.learner_net = ccnet(
            num_classes=cfg_model.num_classes,
            weight=cfg_model.com_weight,
            headless_mode=self.headless_mode,
            compression_dim=compression_dim
        ).to(self.device)
        
        self.feature_dimension = compression_dim if self.headless_mode else 2048
        
        # Load pretrained weights
        weights_path = cfg_model.load_weights_folder
        if Path(weights_path).exists():
            print(f"[System] Loading pretrained weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            
            if self.headless_mode:
                # Remove classification head
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if not k.startswith('arclayer_')}
                self.predictor_net.load_state_dict(filtered_state_dict, strict=False)
                self.learner_net.load_state_dict(filtered_state_dict, strict=False)
            else:
                self.predictor_net.load_state_dict(state_dict)
                self.learner_net.load_state_dict(state_dict)
                
            print(f"[System] ✅ Weights loaded successfully")
        
        self.predictor_net.eval()
        self.learner_net.train()

    def _initialize_replay_buffer(self):
        """리플레이 버퍼 초기화 (단순화)"""
        print("[System] Initializing replay buffer...")
        cfg_buffer = self.config.replay_buffer
        
        self.replay_buffer = SimplifiedReplayBuffer(
            config=cfg_buffer,
            storage_dir=Path(cfg_buffer.storage_path),
            feature_dimension=self.feature_dimension
        )
        
        # Set feature extractor
        self.replay_buffer.set_feature_extractor(self.learner_net)
        
        # Update hard negative ratio
        self.replay_buffer.update_hard_negative_ratio(self.hard_negative_ratio)

    def _initialize_verification_system(self):
        """검증 시스템 초기화"""
        if self.verification_method == 'metric':
            self.verifier = HeadlessVerifier(
                metric_type=self.metric_type,
                threshold=self.similarity_threshold
            )
            print(f"[System] ✅ Metric verifier initialized")
        else:
            self.verifier = None

    def _initialize_optimizer(self):
        """옵티마이저 초기화"""
        cfg_model = self.config.palm_recognizer
        cfg_loss = self.config.loss
        
        self.optimizer = optim.Adam(
            self.learner_net.parameters(), 
            lr=cfg_model.learning_rate
        )
        
        self.criterion = SupConLoss(
            temperature=getattr(cfg_loss, 'temp', 0.07)
        )
        
        print(f"[System] ✅ Optimizer initialized (lr: {cfg_model.learning_rate})")

    def process_label_batch(self, samples: List[torch.Tensor], user_id: int):
        """
        배치 단위 처리 - 핵심 메서드
        
        Args:
            samples: 한 라벨의 모든 샘플들
            user_id: 사용자 ID
        """
        print(f"\n[Process] 🎯 Processing batch for User {user_id} ({len(samples)} samples)")
        
        # 1. 배치 특징 추출
        batch_embeddings = self._extract_batch_features(samples)
        
        # 2. 학습용 배치 구성
        training_batch = self._construct_training_batch(
            new_samples=samples,
            new_embeddings=batch_embeddings,
            new_user_id=user_id
        )
        
        # 3. 배치 학습
        loss = self._batch_learning(training_batch)
        
        # 4. 선별적 버퍼 저장
        stored_count = self._selective_buffer_storage(
            samples, batch_embeddings, user_id
        )
        
        # 5. 통계 업데이트
        self.global_step += 1
        self.processed_users += 1
        
        # 6. 주기적 동기화
        if self.global_step % self.config.continual_learner.sync_frequency == 0:
            self._sync_weights()
        
        print(f"[Process] ✅ Completed: loss={loss:.4f}, stored={stored_count}/{len(samples)}")
        
        return {
            'loss': loss,
            'stored': stored_count,
            'total': len(samples)
        }

    def _extract_batch_features(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """배치 특징 추출 (GPU 효율적)"""
        self.learner_net.eval()
        
        with torch.no_grad():
            # Stack all samples into a batch
            batch = torch.stack([s.to(self.device) for s in samples])
            
            # Extract features in one forward pass
            features = self.learner_net.getFeatureCode(batch)
        
        self.learner_net.train()
        return features

    def _construct_training_batch(self, new_samples: List[torch.Tensor], 
                                 new_embeddings: torch.Tensor, 
                                 new_user_id: int) -> Dict:
        """학습용 배치 구성 (단순화)"""
        
        # Calculate how many samples we need from buffer
        buffer_samples_needed = max(0, self.training_batch_size - len(new_samples))
        
        print(f"[Batch] Constructing training batch:")
        print(f"   New samples: {len(new_samples)}")
        print(f"   Buffer samples needed: {buffer_samples_needed}")
        
        # Get samples from replay buffer
        if buffer_samples_needed > 0:
            buffer_images, buffer_labels = self.replay_buffer.sample_for_training(
                num_samples=buffer_samples_needed,
                current_embeddings=new_embeddings.cpu().split(1),  # Convert to list
                current_user_id=new_user_id
            )
        else:
            buffer_images, buffer_labels = [], []
        
        # Combine all samples
        all_images = new_samples + buffer_images
        all_labels = [new_user_id] * len(new_samples) + buffer_labels
        
        print(f"[Batch] Final composition: {len(all_images)} samples")
        print(f"   Positive pairs guaranteed: {len(new_samples) * (len(new_samples) - 1) // 2}")
        
        return {
            'images': all_images,
            'labels': all_labels
        }

    def _batch_learning(self, batch_data: Dict) -> float:
        """배치 학습 (단순화)"""
        images = batch_data['images']
        labels = batch_data['labels']
        
        if not images:
            return 0.0
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # Extract features for all images
        embeddings = []
        for img in images:
            img_tensor = img.to(self.device)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            if self.headless_mode:
                _, embedding = self.learner_net(img_tensor)
            else:
                _, embedding = self.learner_net(img_tensor)
            
            embeddings.append(embedding)
        
        # Stack embeddings
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Compute SupCon loss
        embeddings_for_loss = embeddings_tensor.unsqueeze(1)
        loss = self.criterion(embeddings_for_loss, labels_tensor)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _selective_buffer_storage(self, samples: List[torch.Tensor], 
                                 embeddings: torch.Tensor, 
                                 user_id: int) -> int:
        """선별적 버퍼 저장"""
        stored_count = 0
        
        for i, (sample, embedding) in enumerate(zip(samples, embeddings)):
            if self.replay_buffer.add_if_diverse(sample, user_id, embedding):
                stored_count += 1
        
        return stored_count

    def _sync_weights(self):
        """가중치 동기화"""
        self.predictor_net.load_state_dict(self.learner_net.state_dict())
        self.predictor_net.eval()
        
        print(f"\n[Sync] 🔄 Weights synchronized at step {self.global_step}")

    def run_experiment(self):
        """배치 기반 실험 실행"""
        print(f"\n[System] Starting batch-based continual learning...")
        
        # Load dataset
        cfg_dataset = self.config.dataset
        dataset = MyDataset(txt=str(cfg_dataset.dataset_path), train=False)
        
        # Group data by label
        grouped_data = self._group_data_by_label(dataset)
        total_users = len(grouped_data)
        
        print(f"[System] Dataset loaded: {total_users} users")
        print(f"[System] Processing {self.samples_per_label} samples per user")
        
        # Process each user's batch
        for user_id, user_indices in tqdm(grouped_data.items(), desc="Batch Processing"):
            # Skip if already processed
            if self.processed_users > 0 and user_id in self._get_processed_user_ids():
                continue
            
            # Get samples for this user
            samples = []
            for idx in user_indices[:self.samples_per_label]:
                data, _ = dataset[idx]
                samples.append(data[0])  # Use first image from pair
            
            if len(samples) == self.samples_per_label:
                # Process batch
                results = self.process_label_batch(samples, user_id)
                
                # Save checkpoint periodically
                if self.global_step % self.config.continual_learner.intermediate_save_frequency == 0:
                    self._save_checkpoint()
        
        # Final save
        print(f"\n[System] Experiment completed!")
        self._save_checkpoint()
        self._save_final_model()

    def _group_data_by_label(self, dataset) -> Dict[int, List[int]]:
        """데이터를 라벨별로 그룹화"""
        grouped = defaultdict(list)
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            user_id = label.item() if torch.is_tensor(label) else label
            grouped[user_id].append(idx)
        
        return dict(grouped)

    def _get_processed_user_ids(self) -> set:
        """이미 처리된 사용자 ID 반환"""
        # Implementation depends on how you track processed users
        return set()

    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint = {
            'global_step': self.global_step,
            'processed_users': self.processed_users,
            'learner_state_dict': self.learner_net.state_dict(),
            'predictor_state_dict': self.predictor_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'training_batch_size': self.training_batch_size,
                'hard_negative_ratio': self.hard_negative_ratio,
                'samples_per_label': self.samples_per_label,
                'headless_mode': self.headless_mode
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save buffer state
        self.replay_buffer._save_state()
        
        print(f"[Checkpoint] 💾 Saved at step {self.global_step}")

    def _load_checkpoint(self):
        """최신 체크포인트 로드"""
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_step_*.pth'))
        
        if not checkpoint_files:
            print("[Checkpoint] No checkpoint found, starting fresh")
            return
        
        latest = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        print(f"[Checkpoint] Loading from: {latest.name}")
        checkpoint = torch.load(latest, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.processed_users = checkpoint['processed_users']
        
        self.learner_net.load_state_dict(checkpoint['learner_state_dict'])
        self.predictor_net.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"[Checkpoint] ✅ Resumed from step {self.global_step}")

    def _save_final_model(self):
        """최종 모델 저장"""
        save_path = Path(self.config.model_saving.final_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save learner
        learner_path = save_path / f'coconut_batch_learner_{timestamp}.pth'
        torch.save(self.learner_net.state_dict(), learner_path)
        
        # Save predictor
        predictor_path = save_path / f'coconut_batch_predictor_{timestamp}.pth'
        torch.save(self.predictor_net.state_dict(), predictor_path)
        
        # Save metadata
        metadata = {
            'total_steps': self.global_step,
            'total_users': self.processed_users,
            'training_batch_size': self.training_batch_size,
            'hard_negative_ratio': self.hard_negative_ratio,
            'samples_per_label': self.samples_per_label,
            'headless_mode': self.headless_mode,
            'buffer_stats': self.replay_buffer.get_statistics()
        }
        
        metadata_path = save_path / f'coconut_batch_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[System] ✅ Final models saved to: {save_path}")
        print(f"  📁 Learner: {learner_path.name}")
        print(f"  📁 Predictor: {predictor_path.name}")
        print(f"  📁 Metadata: {metadata_path.name}")

# 호환성을 위한 별칭
CoconutSystem = BatchCoconutSystem