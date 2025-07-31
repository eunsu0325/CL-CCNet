# framework/coconut.py - 사용자 노드 시스템 통합 버전

"""
=== COCONUT STAGE 2: USER NODE BASED CONTINUAL LEARNING ===

🔥 MAJOR UPDATES:
- User Node system integration
- Loop Closure mechanism
- Mahalanobis + SupCon alternate training
- ON/OFF modular design
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
from typing import List, Dict, Tuple, Optional

from models.ccnet_model import ccnet, HeadlessVerifier
from framework.replay_buffer import SimplifiedReplayBuffer
from framework.losses import CombinedContrastiveLoss, create_coconut_loss
from framework.user_node import UserNodeManager, UserNode
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class BatchCoconutSystem:
    def __init__(self, config):
        """
        배치 기반 CoCoNut 연속학습 시스템 with User Nodes
        
        DESIGN:
        - User Node based authentication
        - Loop Closure self-correction
        - Alternate training (SupCon ↔ Mahalanobis)
        - Modular ON/OFF design
        """
        print("="*80)
        print("🥥 COCONUT STAGE 2: USER NODE BASED CONTINUAL LEARNING")
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
        
        # 🔥 User Node configuration
        self.user_node_config = getattr(config, 'user_node', None)
        self.user_nodes_enabled = self.user_node_config and self.user_node_config.enable_user_nodes
        
        # 🔥 Loop Closure configuration
        self.loop_closure_config = getattr(config, 'loop_closure', None)
        self.loop_closure_enabled = (self.loop_closure_config and 
                                    self.loop_closure_config.enabled and 
                                    self.user_nodes_enabled)  # 자동 OFF
        
        print(f"🔧 SYSTEM CONFIGURATION:")
        print(f"   Samples per label: {self.samples_per_label}")
        print(f"   Training batch size: {self.training_batch_size}")
        print(f"   Hard negative ratio: {self.hard_negative_ratio:.1%}")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"   🎯 User Nodes: {'ENABLED' if self.user_nodes_enabled else 'DISABLED'}")
        print(f"   🔄 Loop Closure: {'ENABLED' if self.loop_closure_enabled else 'DISABLED'}")
        print("="*80)
        
        # Checkpoint directory
        self.checkpoint_dir = Path('/content/drive/MyDrive/CoCoNut_BatchMode/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_models()
        self._initialize_replay_buffer()
        self._initialize_verification_system()
        self._initialize_optimizer()
        self._initialize_user_node_system()  # 🔥 NEW
        
        # Learning state
        self.global_step = 0
        self.processed_users = 0
        
        # 🔥 Loop Closure queue
        self.loop_closure_queue = []
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        print(f"[System] 🥥 Enhanced CoCoNut System ready!")
        print(f"[System] Starting from step: {self.global_step}")

    def _initialize_models(self):
        """모델 초기화 (기존과 동일)"""
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
        """리플레이 버퍼 초기화 (기존과 동일)"""
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
        """옵티마이저 초기화 - 수정됨"""
        cfg_model = self.config.palm_recognizer
        cfg_loss = self.config.loss
        
        self.optimizer = optim.Adam(
            self.learner_net.parameters(), 
            lr=cfg_model.learning_rate
        )
        
        # 🔥 Combined Loss 사용
        self.criterion = create_coconut_loss(cfg_loss.__dict__)
        
        print(f"[System] ✅ Optimizer initialized (lr: {cfg_model.learning_rate})")

    def _initialize_user_node_system(self):
        """🔥 사용자 노드 시스템 초기화"""
        if self.user_nodes_enabled:
            print("[System] Initializing User Node system...")
            
            # UserNodeManager 생성
            node_config = self.user_node_config.__dict__.copy()
            node_config.pop('config_file', None)  # config_file 제거
            
            self.node_manager = UserNodeManager(
                config=node_config,
                device=self.device
            )
            
            print(f"[System] ✅ User Node system initialized")
        else:
            self.node_manager = None
            print("[System] ⚠️ User Node system is DISABLED")

    def process_label_batch(self, samples: List[torch.Tensor], user_id: int):
        """
        배치 단위 처리 - 사용자 노드 통합 버전
        
        Args:
            samples: 한 라벨의 모든 샘플들
            user_id: 사용자 ID
        """
        print(f"\n[Process] 🎯 Processing batch for User {user_id} ({len(samples)} samples)")
        
        # 🔥 1. Loop Closure 체크
        loop_closure_triggered = False
        collision_user = None
        
        if self.loop_closure_enabled and self.node_manager:
            with torch.no_grad():
                # 첫 번째 샘플로 충돌 체크
                test_features = self._extract_batch_features(samples[:1])
                collision_info = self.node_manager.check_collision(
                    test_features[0], 
                    exclude_user=user_id
                )
                
                if collision_info:
                    collision_user, distance = collision_info
                    print(f"[LoopClosure] ⚠️ Collision detected with User {collision_user}! "
                          f"(distance: {distance:.4f})")
                    loop_closure_triggered = True
        
        # 2. 학습용 배치 구성
        if loop_closure_triggered:
            # 루프 클로저: 충돌한 사용자도 포함
            training_batch = self._construct_loop_closure_batch(
                new_samples=samples,
                new_user_id=user_id,
                collision_user=collision_user
            )
        else:
            # 일반 배치
            training_batch = self._construct_training_batch(
                new_samples=samples,
                new_embeddings=None,  # 나중에 계산
                new_user_id=user_id
            )
        
        # 🔥 3. 교대 학습 (SupCon ↔ Mahalanobis)
        adaptation_epochs = self.config.continual_learner.adaptation_epochs
        if loop_closure_triggered and self.loop_closure_config:
            adaptation_epochs = self.loop_closure_config.retraining_epochs
            print(f"[LoopClosure] Extended training: {adaptation_epochs} epochs")
        
        for epoch in range(adaptation_epochs):
            print(f"[Epoch {epoch+1}/{adaptation_epochs}]")
            
            # Phase 1: SupConLoss
            loss_dict = self._train_step(training_batch, phase='supcon')
            print(f"   SupCon Loss: {loss_dict['supcon']:.4f}")
            
            # Phase 2: Mahalanobis Loss (if enabled)
            if self.criterion.enable_mahalanobis:
                loss_dict = self._train_step(training_batch, phase='mahal')
                print(f"   Mahal Loss: {loss_dict['mahal']:.4f}")
        
        # 🔥 4. 사용자 노드 생성/업데이트
        if self.user_nodes_enabled and self.node_manager:
            final_embeddings = self._extract_batch_features(samples)
            
            if loop_closure_triggered:
                # 루프 클로저: 두 사용자 모두 재구성
                print(f"[LoopClosure] Reconstructing nodes for users {user_id} and {collision_user}")
                self.node_manager.reconstruct_user_node(user_id, final_embeddings)
                
                # 충돌한 사용자도 재계산
                collision_samples = self._get_user_samples_from_buffer(collision_user)
                if collision_samples:
                    collision_embeddings = self._extract_batch_features(collision_samples)
                    self.node_manager.reconstruct_user_node(collision_user, collision_embeddings)
            else:
                # 일반 추가/업데이트
                self.node_manager.add_user(user_id, final_embeddings)
        
        # 5. 선별적 버퍼 저장 (기존과 동일)
        batch_embeddings = self._extract_batch_features(samples)
        stored_count = self._selective_buffer_storage(samples, batch_embeddings, user_id)
        
        # 6. 통계 업데이트
        self.global_step += 1
        self.processed_users += 1
        
        # 7. 주기적 동기화
        if self.global_step % self.config.continual_learner.sync_frequency == 0:
            self._sync_weights()
        
        print(f"[Process] ✅ Completed: stored={stored_count}/{len(samples)}")
        
        return {
            'stored': stored_count,
            'total': len(samples),
            'loop_closure': loop_closure_triggered
        }

    def _train_step(self, batch_data: Dict, phase: str) -> Dict[str, torch.Tensor]:
        """한 스텝 학습 - 수정됨"""
        images = batch_data['images']
        labels = batch_data['labels']
        
        if not images:
            return {'total': 0.0, 'supcon': 0.0, 'mahal': 0.0}
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # Extract features
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
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # 🔥 Combined Loss 사용
        loss_dict = self.criterion(embeddings_tensor, labels_tensor, phase=phase)
        
        # Backward
        loss_dict['total'].backward()
        self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def _construct_loop_closure_batch(self, new_samples: List[torch.Tensor], 
                                    new_user_id: int, 
                                    collision_user: int) -> Dict:
        """🔥 루프 클로저용 배치 구성"""
        print(f"[LoopClosure] Constructing special batch for separation learning")
        
        # 새 사용자 샘플
        all_images = new_samples.copy()
        all_labels = [new_user_id] * len(new_samples)
        
        # 충돌한 사용자의 샘플 가져오기
        collision_samples = self._get_user_samples_from_buffer(collision_user)
        if collision_samples:
            all_images.extend(collision_samples)
            all_labels.extend([collision_user] * len(collision_samples))
            print(f"[LoopClosure] Added {len(collision_samples)} samples from User {collision_user}")
        
        # 추가 버퍼 샘플 (다양성을 위해)
        remaining_slots = max(0, self.training_batch_size - len(all_images))
        if remaining_slots > 0:
            buffer_images, buffer_labels = self.replay_buffer.sample_for_training(
                num_samples=remaining_slots,
                current_embeddings=[],
                current_user_id=new_user_id
            )
            all_images.extend(buffer_images)
            all_labels.extend(buffer_labels)
        
        print(f"[LoopClosure] Batch composition: {len(all_images)} samples")
        return {'images': all_images, 'labels': all_labels}

    def _get_user_samples_from_buffer(self, user_id: int) -> List[torch.Tensor]:
        """특정 사용자의 샘플을 버퍼에서 가져오기"""
        user_samples = []
        
        for item in self.replay_buffer.image_storage:
            if item['user_id'] == user_id:
                user_samples.append(item['image'])
        
        return user_samples

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
        """학습용 배치 구성 (기존과 동일)"""
        
        # Calculate how many samples we need from buffer
        buffer_samples_needed = max(0, self.training_batch_size - len(new_samples))
        
        print(f"[Batch] Constructing training batch:")
        print(f"   New samples: {len(new_samples)}")
        print(f"   Buffer samples needed: {buffer_samples_needed}")
        
        # Get samples from replay buffer
        if buffer_samples_needed > 0:
            if new_embeddings is None:
                new_embeddings = self._extract_batch_features(new_samples)
                
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
        
        return {
            'images': all_images,
            'labels': all_labels
        }

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
        if self.node_manager and self.user_nodes_enabled:
            return set(self.node_manager.nodes.keys())
        return set()

    def _save_checkpoint(self):
        """체크포인트 저장 - 수정됨"""
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
                'headless_mode': self.headless_mode,
                'user_nodes_enabled': self.user_nodes_enabled,
                'loop_closure_enabled': self.loop_closure_enabled
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save buffer state
        self.replay_buffer._save_state()
        
        # 🔥 Save user nodes
        if self.node_manager and self.user_nodes_enabled:
            self.node_manager.save_nodes()
        
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
        """최종 모델 저장 - 수정됨"""
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
            'user_nodes_enabled': self.user_nodes_enabled,
            'loop_closure_enabled': self.loop_closure_enabled,
            'buffer_stats': self.replay_buffer.get_statistics()
        }
        
        # 🔥 Add user node statistics
        if self.node_manager and self.user_nodes_enabled:
            metadata['user_node_stats'] = self.node_manager.get_statistics()
        
        metadata_path = save_path / f'coconut_batch_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[System] ✅ Final models saved to: {save_path}")
        print(f"  📁 Learner: {learner_path.name}")
        print(f"  📁 Predictor: {predictor_path.name}")
        print(f"  📁 Metadata: {metadata_path.name}")

    def verify_user(self, input_palmprint: torch.Tensor) -> Dict:
        """🔥 사용자 검증 (노드 기반)"""
        if not self.user_nodes_enabled or not self.node_manager:
            # Fallback to original verification
            return self.verifier.verify(input_palmprint, self.predictor_net)
        
        # Extract embedding
        with torch.no_grad():
            if len(input_palmprint.shape) == 3:
                input_palmprint = input_palmprint.unsqueeze(0)
            query_embedding = self.predictor_net.getFeatureCode(input_palmprint)[0]
        
        # Find nearest users
        nearest_users = self.node_manager.find_nearest_users(query_embedding, k=5)
        
        if not nearest_users:
            return {
                'is_match': False,
                'matched_user': None,
                'confidence': 0.0,
                'reason': 'No registered users'
            }
        
        # Best match
        best_user_id, best_distance = nearest_users[0]
        
        # Threshold check
        is_match = best_distance < self.similarity_threshold
        
        # Confidence calculation
        confidence = max(0.0, 1.0 - (best_distance / self.similarity_threshold)) if is_match else 0.0
        
        return {
            'is_match': is_match,
            'matched_user': best_user_id if is_match else None,
            'distance': best_distance,
            'confidence': confidence,
            'top_k_results': nearest_users,
            'method': 'user_node_mahalanobis'
        }

# 호환성을 위한 별칭
CoconutSystem = BatchCoconutSystem