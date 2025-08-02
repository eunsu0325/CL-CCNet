# framework/coconut.py - CCNet 스타일로 수정된 버전

"""
=== COCONUT STAGE 2: CONTINUAL LEARNING ===

🔥 CCNet Style Implementation:
- Use both images from dataset pairs
- SupCon loss with proper multi-view format
- User Node system maintained
- Fixed NaN issues
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
import numpy as np

from models.ccnet_model import ccnet, HeadlessVerifier
from framework.replay_buffer import ReplayBuffer
from framework.losses import create_coconut_loss
from framework.user_node import UserNodeManager, UserNode
from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader

class CoconutSystem:
    def __init__(self, config):
        """
        배치 기반 CoCoNut 연속학습 시스템 - CCNet 스타일
        
        DESIGN:
        - SupCon loss with proper 2-view format
        - User Node based authentication
        - Even-count buffer management
        """
        print("="*80)
        print("🥥 COCONUT: CONTINUAL LEARNING (CCNet Style)")
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
        self.samples_per_label = getattr(self.config.dataset, 'samples_per_label', 10)
        
        # User Node configuration
        self.user_node_config = getattr(config, 'user_node', None)
        self.user_nodes_enabled = self.user_node_config and self.user_node_config.enable_user_nodes
        
        # Loop Closure configuration
        self.loop_closure_config = getattr(config, 'loop_closure', None)
        self.loop_closure_enabled = self.loop_closure_config and self.loop_closure_config.enabled
        
        print(f"🔧 SYSTEM CONFIGURATION:")
        print(f"   Samples per label: {self.samples_per_label}")
        print(f"   Training batch size: {self.training_batch_size}")
        print(f"   Hard negative ratio: {self.hard_negative_ratio:.1%}")
        print(f"   Mode: {'Headless' if self.headless_mode else 'Classification'}")
        print(f"   🎯 User Nodes: {'ENABLED' if self.user_nodes_enabled else 'DISABLED'}")
        print(f"   📊 Loss: SupCon (CCNet style)")
        print("="*80)
        
        # Checkpoint directory
        self.checkpoint_dir = Path('/content/drive/MyDrive/CL-CCNet_nodemode/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_models()
        self._initialize_replay_buffer()
        self._initialize_verification_system()
        self._initialize_optimizer()
        self._initialize_user_node_system()
        
        # Learning state
        self.global_step = 0
        self.processed_users = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        print(f"[System] 🥥 CoCoNut System ready!")
        print(f"[System] Starting from step: {self.global_step}")

    def _initialize_models(self):
        """모델 초기화"""
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
        """리플레이 버퍼 초기화"""
        print("[System] Initializing replay buffer...")
        cfg_buffer = self.config.replay_buffer
        
        self.replay_buffer = ReplayBuffer(
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
        
        # 손실 함수 (SupCon)
        self.criterion = create_coconut_loss(cfg_loss.__dict__)
        
        print(f"[System] ✅ Optimizer initialized (lr: {cfg_model.learning_rate})")
        print(f"[System] ✅ Loss: SupCon (CCNet style)")

    def _initialize_user_node_system(self):
        """사용자 노드 시스템 초기화"""
        if self.user_nodes_enabled:
            print("[System] Initializing User Node system...")
            
            # UserNodeManager 생성
            node_config = self.user_node_config.__dict__.copy()
            node_config.pop('config_file', None)
            node_config['feature_dimension'] = self.feature_dimension
            
            self.node_manager = UserNodeManager(
                config=node_config,
                device=self.device
            )
            
            print(f"[System] ✅ User Node system initialized")
        else:
            self.node_manager = None
            print("[System] ⚠️ User Node system is DISABLED")

    def _prepare_registration_image(self, sample_tensor):
        """등록 이미지 준비"""
        try:
            # 텐서를 numpy로 변환
            image_np = sample_tensor.cpu().numpy()
            
            # 형태 확인 및 변환
            if len(image_np.shape) == 3:
                # (C, H, W) -> (H, W, C)
                if image_np.shape[0] in [1, 3]:
                    image_np = image_np.transpose(1, 2, 0)
            
            # 값 범위 정규화 (0-1 -> 0-255)
            if image_np.dtype in [np.float32, np.float64]:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            
            # 그레이스케일 처리
            if len(image_np.shape) == 3 and image_np.shape[2] == 1:
                image_np = image_np.squeeze(2)
            
            return image_np
            
        except Exception as e:
            print(f"[System] ❌ Error preparing registration image: {e}")
            # 더미 이미지 생성
            dummy_image = np.full((128, 128), 128, dtype=np.uint8)
            return dummy_image

    def process_label_batch(self, sample_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
                        user_id: int,
                        raw_images: List[Tuple[np.ndarray, np.ndarray]] = None):
        """
        배치 단위 처리 - CCNet 스타일
        
        Args:
            sample_pairs: [(img1, img2), ...] 형태의 정규화된 이미지 페어
            user_id: 사용자 ID
            raw_images: [(raw1, raw2), ...] 형태의 원본 이미지 페어
        """
        print(f"\n[Process] 🎯 Processing batch for User {user_id} ({len(sample_pairs)} pairs)")
        
        # 1. 훈련 배치 구성
        training_batch = self._construct_training_batch(
            sample_pairs=sample_pairs,
            user_id=user_id
        )
        
        # 2. 학습 수행
        adaptation_epochs = self.config.continual_learner.adaptation_epochs
        
        for epoch in range(adaptation_epochs):
            print(f"[Epoch {epoch+1}/{adaptation_epochs}]")
            loss_dict = self._train_step_ccnet_style(training_batch)
            
            # NaN 체크
            if torch.isnan(torch.tensor(loss_dict['total'])):
                print(f"   ⚠️ NaN detected! Skipping this batch.")
                return {'stored': 0, 'total': len(sample_pairs)}
            
            print(f"   Loss: {loss_dict['total']:.4f}")
        
        # 3. 사용자 노드 생성/업데이트
        if self.user_nodes_enabled and self.node_manager:
            # 모든 이미지의 특징 추출
            all_embeddings = []
            all_normalized_tensors = []  # 정규화된 텐서 수집
            
            for img1, img2 in sample_pairs:
                emb1 = self._extract_feature(img1)
                emb2 = self._extract_feature(img2)
                all_embeddings.extend([emb1, emb2])
                
                # 정규화된 텐서도 저장 (Loop Closure용)
                all_normalized_tensors.extend([img1.cpu(), img2.cpu()])
            
            final_embeddings = torch.stack(all_embeddings)  # [20, feature_dim]
            
            # 원본 이미지는 시각화용
            registration_image = None
            if raw_images and len(raw_images) > 0:
                registration_image = raw_images[0][0]  # 첫 번째 원본 이미지
            
            # User Node 업데이트 (정규화된 텐서 포함)
            self.node_manager.add_user(
                user_id, 
                final_embeddings, 
                registration_image=registration_image,
                normalized_tensors=all_normalized_tensors  # Loop Closure용
            )
        
        # 4. 선별적 버퍼 저장 (짝수 유지)
        stored_count = self._store_to_buffer_even(sample_pairs, user_id)
        
        # 5. 통계 업데이트
        self.global_step += 1
        self.processed_users += 1
        
        # 6. 주기적 동기화
        if self.global_step % self.config.continual_learner.sync_frequency == 0:
            self._sync_weights()
        
        # 7. Loop Closure 체크 (옵션)
        if self.loop_closure_enabled and self.global_step % 10 == 0:
            self._check_loop_closure()
        
        print(f"[Process] ✅ Completed: stored={stored_count}/{len(sample_pairs)*2}")
        
        return {
            'stored': stored_count,
            'total': len(sample_pairs) * 2
        }
    
    def _check_loop_closure(self):
        """Loop Closure 체크 및 실행"""
        if not self.node_manager:
            return
        
        print("\n[Loop Closure] Checking for candidates...")
        
        # Loop Closure 후보 찾기
        candidates = self.node_manager.get_loop_closure_candidates(
            similarity_threshold=0.8
        )
        
        if not candidates:
            print("[Loop Closure] No candidates found")
            return
        
        print(f"[Loop Closure] Found {len(candidates)} candidate pairs")
        
        # 상위 2개만 처리 (시간 절약)
        max_pairs = 2
        for user1, user2, similarity in candidates[:max_pairs]:
            print(f"[Loop Closure] Processing pair: User {user1} <-> User {user2} (sim: {similarity:.3f})")
            
            # 두 사용자의 데이터 가져오기
            loop_data = self.node_manager.get_loop_closure_data([user1, user2])
            
            if user1 in loop_data and user2 in loop_data:
                # 정규화된 텐서들로 재학습
                _, tensors1 = loop_data[user1]
                _, tensors2 = loop_data[user2]
                
                # 재학습을 위한 배치 구성
                combined_pairs = []
                for t in tensors1[:3]:  # 각 사용자에서 최대 3개
                    combined_pairs.append((t, t))  # 같은 이미지로 페어 구성
                for t in tensors2[:3]:
                    combined_pairs.append((t, t))
                
                # 재학습 실행
                if combined_pairs:
                    print(f"[Loop Closure] Retraining with {len(combined_pairs)} pairs")
                    training_batch = self._construct_training_batch(
                        sample_pairs=combined_pairs,
                        user_id=-1  # 특별한 ID로 Loop Closure 표시
                    )
                    
                    # 1 epoch만 학습
                    loss_dict = self._train_step_ccnet_style(training_batch)
                    print(f"[Loop Closure] Loss: {loss_dict['total']:.4f}")
                    
    def _train_step_ccnet_style(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """CCNet 스타일 학습 스텝"""
        sample_pairs = batch_data['sample_pairs']  # [(img1, img2), ...]
        buffer_samples = batch_data['buffer_samples']  # [(img, label), ...]
        
        if not sample_pairs and not buffer_samples:
            return {'total': 0.0, 'supcon': 0.0}
        
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        # CCNet 스타일로 특징 추출
        features_list = []
        labels_list = []
        
        # 1. 새 사용자의 페어들 처리
        for (img1, img2), label in sample_pairs:
            # 각 이미지에서 특징 추출
            img1_tensor = img1.to(self.device)
            img2_tensor = img2.to(self.device)
            
            if len(img1_tensor.shape) == 3:
                img1_tensor = img1_tensor.unsqueeze(0)
            if len(img2_tensor.shape) == 3:
                img2_tensor = img2_tensor.unsqueeze(0)
            
            _, feat1 = self.learner_net(img1_tensor)
            _, feat2 = self.learner_net(img2_tensor)
            
            # [2, feature_dim] 형태로 묶기
            paired_features = torch.stack([feat1.squeeze(0), feat2.squeeze(0)], dim=0)
            features_list.append(paired_features)
            labels_list.append(label)
        
        # 2. 버퍼 샘플들을 라벨별로 그룹화하여 페어 만들기
        if buffer_samples:
            label_groups = defaultdict(list)
            for img, lbl in buffer_samples:
                label_groups[lbl].append(img)
            
            # 각 라벨에서 짝수개씩 선택하여 페어 구성
            for lbl, imgs in label_groups.items():
                # 짝수개로 만들기
                num_imgs = len(imgs)
                if num_imgs >= 2:
                    # 짝수개만 사용
                    for i in range(0, num_imgs - 1, 2):
                        img1_tensor = imgs[i].to(self.device)
                        img2_tensor = imgs[i+1].to(self.device)
                        
                        if len(img1_tensor.shape) == 3:
                            img1_tensor = img1_tensor.unsqueeze(0)
                        if len(img2_tensor.shape) == 3:
                            img2_tensor = img2_tensor.unsqueeze(0)
                        
                        _, feat1 = self.learner_net(img1_tensor)
                        _, feat2 = self.learner_net(img2_tensor)
                        
                        paired_features = torch.stack([feat1.squeeze(0), feat2.squeeze(0)], dim=0)
                        features_list.append(paired_features)
                        labels_list.append(lbl)
        
        if not features_list:
            return {'total': 0.0, 'supcon': 0.0}
        
        # [batch_size, 2, feature_dim] 형태로 스택
        features_tensor = torch.stack(features_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)
        
        print(f"[Train] Batch shape: {features_tensor.shape}, Labels: {labels_tensor.shape}")
        
        # SupCon Loss 계산
        loss_dict = self.criterion(features_tensor, labels_tensor)
        
        # Backward with gradient clipping
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.learner_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def _construct_training_batch(self, sample_pairs: List[Tuple], user_id: int) -> Dict:
        """CCNet 스타일 배치 구성"""
        
        # 새 사용자의 페어들
        new_pairs = [(pair, user_id) for pair in sample_pairs]
        
        # 버퍼에서 샘플 가져오기
        # 목표: 전체 배치가 적절한 크기가 되도록
        num_new_samples = len(sample_pairs) * 2  # 각 페어는 2개 이미지
        buffer_samples_needed = max(0, self.training_batch_size - num_new_samples)
        
        # 짝수로 맞추기 (페어를 만들기 위해)
        if buffer_samples_needed % 2 == 1:
            buffer_samples_needed += 1
        
        print(f"[Batch] Constructing training batch:")
        print(f"   New pairs: {len(sample_pairs)} ({num_new_samples} images)")
        print(f"   Buffer samples needed: {buffer_samples_needed}")
        
        buffer_samples = []
        if buffer_samples_needed > 0:
            buffer_images, buffer_labels = self.replay_buffer.sample_for_training_even(
                num_samples=buffer_samples_needed,
                current_user_id=user_id
            )
            
            buffer_samples = list(zip(buffer_images, buffer_labels))
            print(f"   Buffer samples retrieved: {len(buffer_samples)}")
        
        print(f"[Batch] Final composition: {len(new_pairs)} pairs + {len(buffer_samples)} buffer samples")
        
        return {
            'sample_pairs': new_pairs,
            'buffer_samples': buffer_samples
        }

    def _store_to_buffer_even(self, sample_pairs: List[Tuple], user_id: int) -> int:
        """버퍼에 짝수개로 저장"""
        stored_count = 0
        user_embeddings = []
        user_images = []
        
        # 모든 이미지와 임베딩 수집
        for img1, img2 in sample_pairs:
            emb1 = self._extract_feature(img1)
            emb2 = self._extract_feature(img2)
            
            user_embeddings.extend([emb1, emb2])
            user_images.extend([img1, img2])
        
        # 다양성 점수 계산
        diversity_scores = []
        for i, (img, emb) in enumerate(zip(user_images, user_embeddings)):
            # 다른 임베딩들과의 평균 유사도
            similarities = []
            for j, other_emb in enumerate(user_embeddings):
                if i != j:
                    sim = F.cosine_similarity(emb.unsqueeze(0), other_emb.unsqueeze(0)).item()
                    similarities.append(sim)
            avg_sim = np.mean(similarities) if similarities else 0
            diversity_scores.append((i, avg_sim, img, emb))
        
        # 다양성이 높은 순으로 정렬 (유사도가 낮은 순)
        diversity_scores.sort(key=lambda x: x[1])
        
        # 짝수개만 저장 (최대 samples_per_user_limit까지)
        max_to_store = min(len(diversity_scores), self.replay_buffer.samples_per_user_limit)
        if max_to_store % 2 == 1:
            max_to_store -= 1  # 짝수로 만들기
        
        for i in range(max_to_store):
            idx, sim, img, emb = diversity_scores[i]
            if self.replay_buffer.add_sample_direct(img, user_id, emb):
                stored_count += 1
        
        return stored_count

    def _extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """단일 이미지에서 특징 추출"""
        self.learner_net.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            features = self.learner_net.getFeatureCode(image)
        
        self.learner_net.train()
        return features.squeeze(0)

    def _extract_batch_features(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """배치 특징 추출"""
        self.learner_net.eval()
        
        with torch.no_grad():
            # Stack all samples into a batch
            batch = torch.stack([s.to(self.device) for s in samples])
            
            # Extract features in one forward pass
            features = self.learner_net.getFeatureCode(batch)
        
        self.learner_net.train()
        return features

    def _sync_weights(self):
        """가중치 동기화"""
        self.predictor_net.load_state_dict(self.learner_net.state_dict())
        self.predictor_net.eval()
        
        print(f"\n[Sync] 🔄 Weights synchronized at step {self.global_step}")

    def verify_user(self, probe_image: torch.Tensor, top_k: int = 10) -> Dict:
        """사용자 인증"""
        if not self.node_manager:
            return {
                'is_match': False,
                'error': 'No node manager available'
            }
        
        start_time = time.time()
        
        # 1. 프로브 이미지 특징 추출
        self.predictor_net.eval()
        with torch.no_grad():
            if len(probe_image.shape) == 3:
                probe_image = probe_image.unsqueeze(0)
            probe_image = probe_image.to(self.device)
            probe_feature = self.predictor_net.getFeatureCode(probe_image).squeeze(0)
        
        # 2. User Node Manager를 통한 인증
        auth_result = self.node_manager.verify_user(probe_feature, top_k=top_k)
        
        # 3. 결과에 추가 정보 포함
        auth_result['computation_time'] = time.time() - start_time
        
        return auth_result

    def run_experiment(self):
        """배치 기반 실험 실행 - CCNet 스타일"""
        print(f"\n[System] Starting CCNet-style continual learning...")
        
        # Load dataset with return_raw=True for raw images
        cfg_dataset = self.config.dataset
        dataset = MyDataset(txt=str(cfg_dataset.train_set_file), train=False, return_raw=True)  # 🔥 FIX
        
        # Group data by label
        grouped_data = self._group_data_by_label(dataset)
        total_users = len(grouped_data)
        
        print(f"[System] Dataset loaded: {total_users} users")
        print(f"[System] Processing {self.samples_per_label} samples per user")
        print(f"[System] Using 2 images per sample (CCNet style)")
        
        # Process each user's batch
        for user_id, user_indices in tqdm(grouped_data.items(), desc="Batch Processing"):
            # Skip if already processed
            if self.processed_users > 0 and user_id in self._get_processed_user_ids():
                continue
            
            # Get sample pairs with raw images
            sample_pairs = []
            raw_images = []
            
            for idx in user_indices[:self.samples_per_label]:
                data, _, raw_data = dataset[idx]  # 원본도 받음
                sample_pairs.append((data[0], data[1]))
                raw_images.append((raw_data[0], raw_data[1]))
            
            if len(sample_pairs) == self.samples_per_label:
                # Process batch with raw images
                results = self.process_label_batch(sample_pairs, user_id, raw_images)
                
                # Save checkpoint periodically
                if self.global_step % self.config.continual_learner.intermediate_save_frequency == 0:
                    self._save_checkpoint()
        
        # Final save
        print(f"\n[System] Experiment completed!")
        self._save_checkpoint()
        self._save_final_model()
        
        # End-to-End 평가 실행
        if hasattr(self.config.dataset, 'test_set_file'):
            print("\n[System] Running End-to-End evaluation...")
            self.run_evaluation()

    def _group_data_by_label(self, dataset) -> Dict[int, List[int]]:
        """데이터를 라벨별로 그룹화"""
        grouped = defaultdict(list)
        
        for idx in range(len(dataset)):
            if dataset.return_raw:
                _, label, _ = dataset[idx]
            else:
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
                'headless_mode': self.headless_mode,
                'user_nodes_enabled': self.user_nodes_enabled
            }
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save buffer state
        self.replay_buffer._save_state()
        
        # Save user nodes
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
        """최종 모델 저장"""
        save_path = Path(self.config.model_saving.final_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save learner
        learner_path = save_path / f'coconut_learner_{timestamp}.pth'
        torch.save(self.learner_net.state_dict(), learner_path)
        
        # Save predictor
        predictor_path = save_path / f'coconut_predictor_{timestamp}.pth'
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
            'loss_type': 'SupCon (CCNet style)',
            'buffer_stats': self.replay_buffer.get_statistics()
        }
        
        # Add user node statistics
        if self.node_manager and self.user_nodes_enabled:
            metadata['user_node_stats'] = self.node_manager.get_statistics()
        
        metadata_path = save_path / f'coconut_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[System] ✅ Final models saved to: {save_path}")
        print(f"  📁 Learner: {learner_path.name}")
        print(f"  📁 Predictor: {predictor_path.name}")
        print(f"  📁 Metadata: {metadata_path.name}")

    def run_evaluation(self):
        """End-to-End 평가 실행"""
        try:
            from evaluation.eval_utils import CoconutEvaluator
            
            test_file = getattr(self.config.dataset, 'test_set_file', None)
            if not test_file:
                print("⚠️ No test file specified in config")
                return None
            
            print("\n" + "="*80)
            print("🔍 Starting End-to-End Authentication Evaluation")
            print("="*80)
            
            # 평가기 생성
            evaluator = CoconutEvaluator(
                model=self.predictor_net,
                node_manager=self.node_manager,
                device=self.device
            )
            
            # 평가 실행
            results = evaluator.run_end_to_end_evaluation(
                test_file_path=test_file,
                batch_size=32,
                save_results=True,
                output_dir="./evaluation_results"
            )
            
            return results
            
        except ImportError as e:
            print(f"⚠️ Evaluation module not found: {e}")
            print("📝 Skipping end-to-end evaluation")
            return None
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return None