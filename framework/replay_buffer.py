# framework/replay_buffer.py - Artificial Positive Pairs 지원
"""
CoCoNut Intelligent Replay Buffer with Advanced Sampling

CORE FEATURES:
- Diversity-based sample selection using Faiss similarity search
- Hard sample mining with fixed ratio
- 3-type augmentation: Geometric + Resolution + Noise
- 🔥 NEW: Artificial positive pairs for single-sample users
- Complete state management for checkpoint resume
"""

import os
import pickle
import random
from pathlib import Path

import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class CoconutReplayBuffer:
    def __init__(self, config, storage_dir: Path, feature_dimension: int = 2048):
        """
        지능형 리플레이 버퍼 초기화
        """
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension

        # Faiss 인덱스 및 저장소
        self.image_storage = []
        self.faiss_index = None
        self.stored_embeddings = []
        self.metadata = {}
        self.feature_extractor = None

        # 🔥 Hard Mining 설정 (나중에 CoconutSystem에서 설정됨)
        self.enable_hard_mining = False  # 기본값
        self.hard_ratio = 0.3  # 기본값
        
        # 🔥 데이터 증강 설정 (나중에 CoconutSystem에서 설정됨)
        self.enable_augmentation = False  # 기본값
        self.aug_config = None
        self._setup_augmentation_transforms()
        
        # 상태 파일 경로
        self.state_file = self.storage_dir / 'buffer_state.pkl'
        self._load_state()

        print(f"[Buffer] 🥥 CoCoNut Replay Buffer initialized")
        print(f"[Buffer] Max size: {self.buffer_size}, Threshold: {self.similarity_threshold}")
        print(f"[Buffer] Hard mining: {self.enable_hard_mining} (ratio: {self.hard_ratio})")
        print(f"[Buffer] Augmentation: {self.enable_augmentation}")
        print(f"[Buffer] Current size: {len(self.image_storage)}")

    def update_hard_mining_config(self, enable_hard_mining, hard_ratio):
        """🔥 Hard Mining 설정 업데이트 (CoconutSystem에서 호출)"""
        self.enable_hard_mining = enable_hard_mining
        self.hard_ratio = hard_ratio
        print(f"[Buffer] 🔥 Hard Mining updated: {self.enable_hard_mining} (ratio: {self.hard_ratio})")

    def update_augmentation_config(self, enable_augmentation, aug_config):
        """🔥 데이터 증강 설정 업데이트 (CoconutSystem에서 호출)"""
        self.enable_augmentation = enable_augmentation
        self.aug_config = aug_config
        self._setup_augmentation_transforms()
        print(f"[Buffer] 🎨 Augmentation updated: {self.enable_augmentation}")

    def _setup_augmentation_transforms(self):
        """3가지 증강 변환 설정"""
        if not self.enable_augmentation or not self.aug_config:
            self.geometric_transform = None
            self.noise_transform = None
            return
            
        # 1. 기하학적 변환
        if getattr(self.aug_config, 'enable_geometric', False):
            max_rotation = getattr(self.aug_config, 'max_rotation_degrees', 3)
            max_translation = getattr(self.aug_config, 'max_translation_ratio', 0.05)
            
            self.geometric_transform = transforms.RandomAffine(
                degrees=(-max_rotation, max_rotation),
                translate=(max_translation, max_translation),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        else:
            self.geometric_transform = None
            
        # 2. 해상도 적응 - 동적으로 처리
        self.resolution_config = {
            'enable': getattr(self.aug_config, 'enable_resolution_adaptation', False),
            'probability': getattr(self.aug_config, 'resolution_probability', 0.3),
            'intermediate_sizes': getattr(self.aug_config, 'intermediate_resolutions', [[64, 64], [96, 96], [160, 160]]),
            'methods': getattr(self.aug_config, 'resize_methods', ['bilinear', 'bicubic'])
        }
        
        # 3. 노이즈 - 동적으로 처리
        self.noise_config = {
            'enable': getattr(self.aug_config, 'enable_noise', False),
            'probability': getattr(self.aug_config, 'noise_probability', 0.3),
            'std_range': getattr(self.aug_config, 'noise_std_range', [0.01, 0.03])
        }

    def set_feature_extractor(self, model):
        """특징 추출을 위한 모델 설정"""
        self.feature_extractor = model

    def _initialize_faiss(self):
        """Faiss 인덱스 초기화"""
        index = faiss.IndexFlatIP(self.feature_dimension)
        self.faiss_index = faiss.IndexIDMap(index)
        print(f"[Buffer] Faiss index initialized with dimension {self.feature_dimension}")

    def _extract_feature_for_diversity(self, image):
        """다양성 측정을 위한 특징 벡터 추출"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor.getFeatureCode(image)
        
        return features

    def add(self, image: torch.Tensor, user_id: int):
        """새로운 경험을 버퍼에 추가 (다양성 기반)"""
        with torch.no_grad():
            embedding = self._extract_feature_for_diversity(image)
            embedding_np = embedding.cpu().numpy().astype('float32')
            faiss.normalize_L2(embedding_np)

        if self.faiss_index is None:
            self._initialize_faiss()

        # 다양성 확인
        if self.faiss_index.ntotal == 0:
            max_similarity = 0.0
        else:
            distances, _ = self.faiss_index.search(embedding_np, k=1)
            max_similarity = distances[0][0]

        # 새로운 경험이면 저장
        if max_similarity < self.similarity_threshold:
            if len(self.image_storage) >= self.buffer_size:
                self._cull()
            
            unique_id = len(self.image_storage)
            self.image_storage.append({
                'image': image.cpu().clone(),
                'user_id': user_id,
                'id': unique_id
            })
            
            self.stored_embeddings.append(embedding_np.copy())
            self.faiss_index.add_with_ids(embedding_np, np.array([unique_id]))
            self.metadata[unique_id] = {'user_id': user_id}
            
            print(f"[Buffer] ✅ Added diverse sample (ID: {unique_id}, User: {user_id}). "
                  f"Buffer: {len(self.image_storage)}/{self.buffer_size}")
        else:
            print(f"[Buffer] ⚠️ Similar sample skipped (similarity: {max_similarity:.4f} >= {self.similarity_threshold})")

    def _cull(self):
        """가장 중복되는 데이터 제거"""
        if self.faiss_index.ntotal < 2:
            return

        print(f"[Buffer] 🔄 Buffer full. Finding most redundant sample...")
        
        if len(self.stored_embeddings) == 0:
            return
        
        all_vectors = np.vstack(self.stored_embeddings)
        k = min(self.faiss_index.ntotal, 50)
        similarities, _ = self.faiss_index.search(all_vectors, k=k)
        
        diversity_scores = similarities.sum(axis=1) - 1.0
        cull_idx_in_storage = np.argmax(diversity_scores)
        cull_unique_id = self.image_storage[cull_idx_in_storage]['id']

        try:
            self.faiss_index.remove_ids(np.array([cull_unique_id]))
        except Exception:
            self._rebuild_faiss_index_after_removal(cull_idx_in_storage)
        
        if cull_unique_id in self.metadata:
            del self.metadata[cull_unique_id]
        
        del self.image_storage[cull_idx_in_storage]
        del self.stored_embeddings[cull_idx_in_storage]
        
        print(f"[Buffer] 🗑️ Removed redundant sample (ID: {cull_unique_id})")

    def _rebuild_faiss_index_after_removal(self, removed_idx):
        """Faiss 인덱스 재구축"""
        print("[Buffer] 🔧 Rebuilding Faiss index...")
        self._initialize_faiss()
        
        for i, (item, embedding) in enumerate(zip(self.image_storage, self.stored_embeddings)):
            if i != removed_idx:
                self.faiss_index.add_with_ids(
                    embedding.reshape(1, -1), 
                    np.array([item['id']])
                )

    def _select_hard_samples(self, num_hard, new_embedding, current_user_id):
        """어려운 샘플 선택"""
        if len(self.stored_embeddings) == 0:
            return []
            
        buffer_embeddings = np.array(self.stored_embeddings)
        new_emb = new_embedding.cpu().numpy().flatten()
        
        # 유사도 계산
        similarities = []
        for i, buffer_emb in enumerate(buffer_embeddings):
            similarity = np.dot(new_emb, buffer_emb.flatten()) / (
                np.linalg.norm(new_emb) * np.linalg.norm(buffer_emb.flatten())
            )
            
            is_same_user = self.image_storage[i]['user_id'] == current_user_id
            
            # Hard score 계산
            if is_same_user:
                hard_score = 1.0 - similarity  # 같은 유저인데 멀리 있음
            else:
                hard_score = similarity       # 다른 유저인데 가까이 있음
                
            similarities.append({
                'index': i,
                'similarity': similarity,
                'hard_score': hard_score,
                'is_same_user': is_same_user
            })
        
        # Hard score 기준 정렬
        similarities.sort(key=lambda x: x['hard_score'], reverse=True)
        
        # 상위 어려운 샘플들 선택
        hard_samples = similarities[:num_hard]
        hard_indices = [item['index'] for item in hard_samples]
        
        # 통계
        hard_positives = sum(1 for item in hard_samples if item['is_same_user'])
        hard_negatives = len(hard_samples) - hard_positives
        
        print(f"[HardMining] 💪 Selected {num_hard} hard samples:")
        print(f"   Hard positives (same user, far): {hard_positives}")
        print(f"   Hard negatives (diff user, close): {hard_negatives}")
        
        return hard_indices

    def _apply_resolution_augmentation(self, image):
        """해상도 적응 증강 (크기는 128x128 유지)"""
        # 중간 해상도 선택
        intermediate_size = random.choice(self.resolution_config['intermediate_sizes'])
        method1 = random.choice(self.resolution_config['methods'])
        method2 = random.choice(self.resolution_config['methods'])
        
        # PIL 변환
        pil_image = transforms.ToPILImage()(image)
        
        # 중간 해상도로 변환
        method1_mode = getattr(transforms.InterpolationMode, method1.upper())
        intermediate = transforms.Resize(intermediate_size, interpolation=method1_mode)(pil_image)
        
        # 원래 크기로 복원
        method2_mode = getattr(transforms.InterpolationMode, method2.upper())
        final_image = transforms.Resize((128, 128), interpolation=method2_mode)(intermediate)
        
        return transforms.ToTensor()(final_image)

    def _apply_differentiated_augmentation(self, image, sample_info="", intensity="normal"):
        """🔥 NEW: 차별화된 증강 강도 적용"""
        if not self.enable_augmentation:
            return image
            
        result = image.clone()
        applied_augs = []
        
        # 강도별 확률 조정
        if intensity == "strong":
            geo_prob = getattr(self.aug_config, 'geometric_probability', 0.3) * 1.5  # 1.5배 증가
            res_prob = self.resolution_config.get('probability', 0.3) * 1.5
            noise_prob = self.noise_config.get('probability', 0.3) * 1.5
        else:
            geo_prob = getattr(self.aug_config, 'geometric_probability', 0.3)
            res_prob = self.resolution_config.get('probability', 0.3)
            noise_prob = self.noise_config.get('probability', 0.3)
        
        # 1. 기하학적 증강
        if (self.geometric_transform and 
            getattr(self.aug_config, 'enable_geometric', False) and
            np.random.random() < geo_prob):
            
            result = self.geometric_transform(result)
            applied_augs.append("Geometric")
        
        # 2. 해상도 적응 증강
        if (self.resolution_config['enable'] and 
            np.random.random() < res_prob):
            
            result = self._apply_resolution_augmentation(result)
            applied_augs.append("Resolution")
        
        # 3. 노이즈 증강
        if (self.noise_config['enable'] and 
            np.random.random() < noise_prob):
            
            # 강도별 노이즈 레벨 조정
            if intensity == "strong":
                noise_std = np.random.uniform(0.02, 0.05)  # 더 강한 노이즈
            else:
                noise_std = np.random.uniform(*self.noise_config['std_range'])
            
            noise = torch.randn_like(result) * noise_std
            result = result + noise
            applied_augs.append(f"Noise(σ={noise_std:.3f})")
        
        if applied_augs:
            print(f"[Augmentation] 🎨 {sample_info}: {', '.join(applied_augs)}")
        
        return result

    def sample_with_replacement(self, batch_size, new_embedding=None, current_user_id=None):
        """
        🔥 Step 3 수정: Artificial Positive Pair 생성
        
        핵심 개선:
        - Single-sample 사용자도 positive pair 생성 (같은 이미지, 다른 증강)
        - 모든 샘플이 학습에 참여 가능
        - SupCon Loss Warning 제거
        """
        if len(self.image_storage) == 0:
            return [], []

        print(f"[Sampling] 🎯 Smart sampling with artificial positive pairs (batch_size: {batch_size}, buffer_size: {len(self.image_storage)})")
        
        # 1. 사용자별 샘플 그룹화
        user_groups = {}
        for i, item in enumerate(self.image_storage):
            user_id = item['user_id']
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(i)
        
        # 2. 사용자 분류
        multi_sample_users = {uid: indices for uid, indices in user_groups.items() 
                             if len(indices) >= 2}
        single_sample_users = {uid: indices for uid, indices in user_groups.items() 
                              if len(indices) == 1}
        
        print(f"[Sampling] 👥 User analysis:")
        print(f"   Multi-sample users: {len(multi_sample_users)} (≥2 samples each)")
        print(f"   Single-sample users: {len(single_sample_users)} (1 sample each)")
        
        # 3. 샘플링 전략 결정
        buffer_size = len(self.image_storage)
        base_allow_duplicates = buffer_size < batch_size
        
        print(f"[Sampling] 📋 Sampling strategy:")
        print(f"   Buffer size vs Batch size: {buffer_size} vs {batch_size}")
        print(f"   Base allow duplicates: {base_allow_duplicates}")
        
        selected_indices = []
        
        # 4. 🔥 NEW: 모든 사용자에 대해 positive pair 우선 확보
        total_users = len(user_groups)
        max_pairs_possible = batch_size // 2  # 최대 가능한 pair 수
        
        # 4a. Multi-sample 사용자 처리
        natural_pairs = min(len(multi_sample_users), max_pairs_possible)
        pair_users = list(multi_sample_users.keys())[:natural_pairs]
        
        for user_id in pair_users:
            indices = multi_sample_users[user_id]
            # 각 사용자에서 2개 선택
            selected = np.random.choice(indices, min(2, len(indices)), replace=False)
            selected_indices.extend(selected)
            print(f"[Sampling] ✅ User {user_id}: selected {len(selected)} samples (natural positive pair)")
        
        # 4b. 🔥 NEW: Single-sample 사용자를 artificial positive pair로 변환
        remaining_pair_slots = max_pairs_possible - natural_pairs
        artificial_pairs = min(len(single_sample_users), remaining_pair_slots)
        
        artificial_users = list(single_sample_users.keys())[:artificial_pairs]
        for user_id in artificial_users:
            idx = single_sample_users[user_id][0]
            # 🔥 같은 인덱스를 2번 추가 (나중에 서로 다른 증강 적용됨)
            selected_indices.extend([idx, idx])
            print(f"[Sampling] 🎨 User {user_id}: created artificial positive pair (same image, different augmentation)")
        
        # 5. 나머지 슬롯 채우기
        remaining_slots = batch_size - len(selected_indices)
        print(f"[Sampling] 📋 Remaining slots to fill: {remaining_slots}")
        
        if remaining_slots > 0:
            # 아직 사용되지 않은 사용자들
            used_users = set(pair_users + artificial_users)
            unused_users = [uid for uid in user_groups.keys() if uid not in used_users]
            
            if unused_users:
                print(f"[Sampling] 📊 Unused users available: {len(unused_users)}")
                # 남은 사용자들에서 샘플 선택
                unused_indices = []
                for uid in unused_users:
                    unused_indices.extend(user_groups[uid])
                
                if len(unused_indices) >= remaining_slots:
                    if base_allow_duplicates:
                        additional_indices = np.random.choice(unused_indices, remaining_slots, replace=True)
                    else:
                        additional_indices = np.random.choice(unused_indices, remaining_slots, replace=False)
                    selected_indices.extend(additional_indices)
                    print(f"[Sampling] 📋 Added {remaining_slots} samples from unused users")
                else:
                    # 부족하면 전체에서 선택
                    all_indices = list(range(len(self.image_storage)))
                    additional_indices = np.random.choice(all_indices, remaining_slots, replace=True)
                    selected_indices.extend(additional_indices)
                    print(f"[Sampling] 🔄 Added {remaining_slots} samples from all users (with replacement)")
            else:
                # 사용되지 않은 사용자가 없으면 전체에서 선택
                all_indices = list(range(len(self.image_storage)))
                additional_indices = np.random.choice(all_indices, remaining_slots, replace=True)
                selected_indices.extend(additional_indices)
                print(f"[Sampling] 🔄 Added {remaining_slots} samples from all users (with replacement)")

        # 6. 🔥 NEW: 최종 샘플 구성 및 차별화된 증강
        images = []
        labels = []
        
        # 중복 인덱스 추적 (artificial positive pair용)
        index_count = {}
        for idx in selected_indices:
            index_count[idx] = index_count.get(idx, 0) + 1
        
        # 각 인덱스별로 몇 번째 사용인지 추적
        index_usage = {}
        
        for i, idx in enumerate(selected_indices):
            item = self.image_storage[idx]
            base_image = item['image'].clone()
            
            # 🔥 중복 사용 시 더 강한 증강 적용
            usage_count = index_usage.get(idx, 0)
            index_usage[idx] = usage_count + 1
            
            if index_count[idx] > 1:
                # Artificial positive pair: 차별화된 증강
                augmentation_intensity = "strong" if usage_count > 0 else "normal"
                sample_info = f"Sample{i+1}(User{item['user_id']}, {augmentation_intensity}_aug)"
            else:
                # Natural sample: 일반 증강
                sample_info = f"Sample{i+1}(User{item['user_id']})"
            
            # 증강 적용 (차별화된 강도)
            augmented_image = self._apply_differentiated_augmentation(
                base_image, 
                sample_info=sample_info,
                intensity=augmentation_intensity if index_count[idx] > 1 else "normal"
            )
            
            images.append(augmented_image)
            labels.append(item['user_id'])

        # 7. 통계 출력 및 검증
        unique_users = len(set(labels))
        user_counts = {}
        for label in labels:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        positive_pairs = sum(1 for count in user_counts.values() if count >= 2)
        artificial_pairs_count = sum(1 for idx, count in index_count.items() if count > 1)
        total_duplicates = len(labels) - len(set(selected_indices))
        
        print(f"[Sampling] 📊 Final batch composition:")
        print(f"   Total samples: {len(images)}")
        print(f"   Unique users: {unique_users}")
        print(f"   Users with ≥2 samples: {positive_pairs} (positive pairs)")
        print(f"   Natural positive pairs: {natural_pairs}")
        print(f"   Artificial positive pairs: {artificial_pairs_count}")
        print(f"   Total duplicates: {total_duplicates}")
        print(f"   User distribution: {dict(sorted(user_counts.items()))}")
        
        # 8. 🔥 Positive pair 보장 검증
        if positive_pairs == 0:
            print(f"🚨 WARNING: No positive pairs in batch! This may cause SupCon loss issues.")
        else:
            print(f"✅ Positive pairs guaranteed: {positive_pairs} users have multiple samples")
            print(f"✅ SupCon Loss Warning should be eliminated!")
        
        return images, labels

    def sample(self, batch_size, **kwargs):
        """기본 샘플링 인터페이스"""
        return self.sample_with_replacement(batch_size, **kwargs)

    def get_diversity_stats(self):
        """버퍼 다양성 통계"""
        if len(self.image_storage) == 0:
            return {
                'total_samples': 0,
                'unique_users': 0,
                'user_distribution': {},
                'diversity_score': 0.0
            }
        
        user_counts = {}
        for item in self.image_storage:
            user_id = item['user_id']
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        unique_users = len(user_counts)
        diversity_score = unique_users / len(self.image_storage)
        
        return {
            'total_samples': len(self.image_storage),
            'unique_users': unique_users,
            'user_distribution': dict(sorted(user_counts.items())),
            'diversity_score': diversity_score
        }

    def save_state(self):
        """상태 저장"""
        if self.faiss_index is None:
            return
            
        data_to_save = {
            'faiss_index_data': faiss.serialize_index(self.faiss_index),
            'metadata': self.metadata,
            'image_storage': self.image_storage,
            'stored_embeddings': self.stored_embeddings
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        diversity_stats = self.get_diversity_stats()
        print(f"[Buffer] 💾 Saved {len(self.image_storage)} samples")
        print(f"[Buffer] Diversity: {diversity_stats['unique_users']} users, "
              f"{diversity_stats['diversity_score']:.2f} score")

    def _load_state(self):
        """상태 로드"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'rb') as f:
                saved_data = pickle.load(f)
                
                self.faiss_index = faiss.deserialize_index(saved_data['faiss_index_data'])
                self.metadata = saved_data['metadata']
                self.image_storage = saved_data.get('image_storage', [])
                self.stored_embeddings = saved_data.get('stored_embeddings', [])
                
                diversity_stats = self.get_diversity_stats()
                print(f"[Buffer] ✅ Restored {len(self.image_storage)} samples")
                print(f"[Buffer] Diversity: {diversity_stats['unique_users']} users")
                
        except Exception as e:
            print(f"[Buffer] ❌ Failed to load state: {e}")
            self.faiss_index = None
            self.metadata = {}
            self.image_storage = []
            self.stored_embeddings = []

print("✅ ReplayBuffer with Artificial Positive Pairs 완료!")