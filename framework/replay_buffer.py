# framework/replay_buffer.py - 완전 제어된 배치 구성 버전

"""
CoCoNut Intelligent Replay Buffer with Controlled Batch Composition

CORE FEATURES:
- 🎯 Precise positive/hard sample ratios (e.g., 30%/30%)  
- 💪 Real hard sample mining with similarity-based selection
- 🎨 Differentiated augmentation for artificial positive pairs
- 📊 Comprehensive batch composition reporting
- 🔧 Config-driven ratio control
- 🔥 Complete GPU/CPU compatibility
"""

import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# 🔥 Faiss import 안전하게 처리
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
    print("[Buffer] 🚀 Faiss available - Buffer optimization enabled")
except ImportError:
    FAISS_AVAILABLE = False
    print("[Buffer] ⚠️ Faiss not found - using PyTorch fallback")
    import numpy as np

import torch
import torchvision.transforms as transforms
from PIL import Image

class CoconutReplayBuffer:
    def __init__(self, config, storage_dir: Path, feature_dimension: int = 128):
        """
        지능형 리플레이 버퍼 초기화 (완전 제어된 배치 구성)
        """
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension
        
        # 🔥 샘플링 전략 설정
        self.sampling_strategy = getattr(self.config, 'sampling_strategy', 'controlled')
        self.force_positive_pairs = getattr(self.config, 'force_positive_pairs', True)
        self.min_positive_pairs = getattr(self.config, 'min_positive_pairs', 1)
        self.max_positive_ratio = getattr(self.config, 'max_positive_ratio', 0.5)
        
        # Faiss 인덱스 및 저장소
        self.image_storage = []
        self.faiss_index = None
        self.stored_embeddings = []
        self.metadata = {}
        self.feature_extractor = None
        
        # 🔥 GPU/CPU 호환성을 위한 device 추적
        self.device = 'cpu'  # 기본값, feature_extractor 설정시 업데이트

        # 배치 구성 비율 (나중에 CoconutSystem에서 설정)
        self.target_positive_ratio = 0.3  # 기본값
        self.hard_ratio = 0.3             # 기본값
        self.enable_hard_mining = False   # 기본값
        
        # 데이터 증강 설정 (나중에 설정)
        self.enable_augmentation = False
        self.aug_config = None
        self._setup_augmentation_transforms()
        
        # 상태 파일 경로
        self.state_file = self.storage_dir / 'buffer_state.pkl'
        self._load_state()

        print(f"[Buffer] 🥥 CoCoNut Controlled Batch Replay Buffer initialized")
        print(f"[Buffer] Strategy: {self.sampling_strategy}")
        print(f"[Buffer] Max buffer size: {self.buffer_size}")
        print(f"[Buffer] Current size: {len(self.image_storage)}")

    def update_batch_composition_config(self, target_positive_ratio: float, hard_mining_ratio: float):
        """🔥 배치 구성 비율 업데이트 (CoconutSystem에서 호출)"""
        self.target_positive_ratio = target_positive_ratio
        self.hard_ratio = hard_mining_ratio
        print(f"[Buffer] 🎯 Batch composition config updated:")
        print(f"   Target positive ratio: {self.target_positive_ratio:.1%}")
        print(f"   Hard mining ratio: {self.hard_ratio:.1%}")

    def update_hard_mining_config(self, enable_hard_mining: bool, hard_ratio: float):
        """Hard Mining 설정 업데이트"""
        self.enable_hard_mining = enable_hard_mining
        self.hard_ratio = hard_ratio
        print(f"[Buffer] 🔥 Hard Mining updated: {self.enable_hard_mining} (ratio: {self.hard_ratio:.1%})")

    def update_augmentation_config(self, enable_augmentation: bool, aug_config):
        """데이터 증강 설정 업데이트"""
        self.enable_augmentation = enable_augmentation
        self.aug_config = aug_config
        self._setup_augmentation_transforms()
        print(f"[Buffer] 🎨 Augmentation updated: {self.enable_augmentation}")

    def set_feature_extractor(self, model):
        """특징 추출을 위한 모델 설정"""
        self.feature_extractor = model
        # 🔥 모델의 device 추적
        if model is not None:
            self.device = next(model.parameters()).device
            print(f"[Buffer] 🔧 Feature extractor device: {self.device}")

    def _initialize_faiss(self):
        """Faiss 인덱스 초기화"""
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(self.feature_dimension)
            self.faiss_index = faiss.IndexIDMap(index)
            print(f"[Buffer] Faiss index initialized with dimension {self.feature_dimension}")
        else:
            self.faiss_index = None
            print(f"[Buffer] Faiss not available - using PyTorch fallback")

    def _extract_feature_for_diversity(self, image):
        """다양성 측정을 위한 특징 벡터 추출 (GPU 호환성 보장)"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
        
        # 🔥 이미지를 모델과 같은 device로 강제 이동
        image = image.to(self.device)
        
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            # getFeatureCode 메서드 사용 (128차원 반환)
            features = self.feature_extractor.getFeatureCode(image)
        
        return features

    def add(self, image: torch.Tensor, user_id: int):
        """새로운 경험을 버퍼에 추가 (GPU/CPU 완전 호환)"""
        
        with torch.no_grad():
            # 🔥 이미지를 모델 device로 강제 이동
            image_for_extraction = image.to(self.device)
            
            embedding = self._extract_feature_for_diversity(image_for_extraction)
            embedding_np = embedding.cpu().numpy().astype('float32')
            
            # Faiss 정규화 (CPU에서 수행)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embedding_np)

        if self.faiss_index is None:
            self._initialize_faiss()

        # 다양성 확인
        if self.faiss_index is None or (FAISS_AVAILABLE and self.faiss_index.ntotal == 0):
            max_similarity = 0.0
        elif FAISS_AVAILABLE:
            distances, _ = self.faiss_index.search(embedding_np, k=1)
            max_similarity = distances[0][0]
        else:
            # Faiss 없을 때 PyTorch 기반 유사도 계산
            max_similarity = self._compute_pytorch_similarity(embedding_np)

        # 새로운 경험이면 저장
        if max_similarity < self.similarity_threshold:
            if len(self.image_storage) >= self.buffer_size:
                self._cull()
            
            unique_id = len(self.image_storage)
            self.image_storage.append({
                'image': image.cpu().clone(),  # 🔥 항상 CPU에 저장
                'user_id': user_id,
                'id': unique_id
            })
            
            self.stored_embeddings.append(embedding_np.copy())
            if FAISS_AVAILABLE and self.faiss_index is not None:
                self.faiss_index.add_with_ids(embedding_np, np.array([unique_id]))
            self.metadata[unique_id] = {'user_id': user_id}
            
            print(f"[Buffer] ✅ Added diverse sample (ID: {unique_id}, User: {user_id}). "
                  f"Buffer: {len(self.image_storage)}/{self.buffer_size}")
        else:
            print(f"[Buffer] ⚠️ Similar sample skipped (similarity: {max_similarity:.4f} >= {self.similarity_threshold})")

    def _compute_pytorch_similarity(self, new_embedding):
        """Faiss 없을 때 PyTorch로 유사도 계산"""
        if len(self.stored_embeddings) == 0:
            return 0.0
        
        new_emb = torch.tensor(new_embedding).flatten()
        max_sim = 0.0
        
        for stored_emb in self.stored_embeddings:
            stored_tensor = torch.tensor(stored_emb).flatten()
            # 코사인 유사도 계산
            similarity = torch.cosine_similarity(new_emb.unsqueeze(0), stored_tensor.unsqueeze(0)).item()
            max_sim = max(max_sim, similarity)
        
        return max_sim

    def _cull(self):
        """가장 중복되는 데이터 제거"""
        if len(self.stored_embeddings) < 2:
            return

        print(f"[Buffer] 🔄 Buffer full. Finding most redundant sample...")
        
        if FAISS_AVAILABLE and self.faiss_index is not None and self.faiss_index.ntotal >= 2:
            all_vectors = np.vstack(self.stored_embeddings)
            k = min(self.faiss_index.ntotal, 50)
            similarities, _ = self.faiss_index.search(all_vectors, k=k)
            diversity_scores = similarities.sum(axis=1) - 1.0
            cull_idx_in_storage = np.argmax(diversity_scores)
        else:
            # PyTorch 기반 다양성 계산
            cull_idx_in_storage = self._find_most_redundant_pytorch()
        
        cull_unique_id = self.image_storage[cull_idx_in_storage]['id']

        # Faiss 인덱스에서 제거
        if FAISS_AVAILABLE and self.faiss_index is not None:
            try:
                self.faiss_index.remove_ids(np.array([cull_unique_id]))
            except Exception:
                self._rebuild_faiss_index_after_removal(cull_idx_in_storage)
        
        # 메타데이터 정리
        if cull_unique_id in self.metadata:
            del self.metadata[cull_unique_id]
        
        del self.image_storage[cull_idx_in_storage]
        del self.stored_embeddings[cull_idx_in_storage]
        
        print(f"[Buffer] 🗑️ Removed redundant sample (ID: {cull_unique_id})")

    def _find_most_redundant_pytorch(self):
        """PyTorch로 가장 중복되는 샘플 찾기"""
        max_similarity = -1
        most_redundant_idx = 0
        
        for i, emb1 in enumerate(self.stored_embeddings):
            total_similarity = 0
            tensor1 = torch.tensor(emb1).flatten()
            
            for j, emb2 in enumerate(self.stored_embeddings):
                if i != j:
                    tensor2 = torch.tensor(emb2).flatten()
                    sim = torch.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()
                    total_similarity += sim
            
            if total_similarity > max_similarity:
                max_similarity = total_similarity
                most_redundant_idx = i
        
        return most_redundant_idx

    def _rebuild_faiss_index_after_removal(self, removed_idx):
        """Faiss 인덱스 재구축"""
        if not FAISS_AVAILABLE:
            return
            
        print("[Buffer] 🔧 Rebuilding Faiss index...")
        self._initialize_faiss()
        
        for i, (item, embedding) in enumerate(zip(self.image_storage, self.stored_embeddings)):
            if i != removed_idx and self.faiss_index is not None:
                self.faiss_index.add_with_ids(
                    embedding.reshape(1, -1), 
                    np.array([item['id']])
                )

    def sample_with_controlled_composition(self, batch_size: int, new_embedding: torch.Tensor = None, 
                                         current_user_id: int = None) -> Tuple[List, List]:
        """
        🔥 핵심 메서드: 제어된 배치 구성으로 정확한 비율 샘플링
        """
        print(f"🎯 [Controlled] Creating controlled batch (size: {batch_size})")
        print(f"   Target positive ratio: {self.target_positive_ratio:.1%}")
        print(f"   Hard mining ratio: {self.hard_ratio:.1%}")
        
        if len(self.image_storage) == 0:
            return [], []

        # 1. 🎯 배치 구성 계획 수립
        target_positive_samples = max(2, int(batch_size * self.target_positive_ratio))
        if target_positive_samples % 2 == 1:
            target_positive_samples += 1
        target_positive_pairs = target_positive_samples // 2
        
        target_hard_samples = int(batch_size * self.hard_ratio)
        remaining_regular = batch_size - target_positive_samples - target_hard_samples
        
        if remaining_regular < 0:
            print(f"[Controlled] ⚠️ Batch size too small, adjusting targets...")
            target_hard_samples = max(0, batch_size - target_positive_samples)
            remaining_regular = batch_size - target_positive_samples - target_hard_samples
        
        print(f"📋 [Controlled] Planned composition:")
        print(f"   Positive pairs: {target_positive_pairs} pairs ({target_positive_samples} samples)")
        print(f"   Hard samples: {target_hard_samples} samples")
        print(f"   Regular samples: {remaining_regular} samples")

        selected_indices = []
        selected_labels = []
        used_user_ids = set()
        
        # 2. 🎯 제한된 Positive Pairs 생성
        pairs_created = self._create_limited_positive_pairs(
            target_positive_pairs, selected_indices, selected_labels, used_user_ids)
        
        # 3. 💪 Hard Samples 선택
        hard_added = 0
        if self.enable_hard_mining and new_embedding is not None and target_hard_samples > 0:
            hard_added = self._select_controlled_hard_samples(
                target_hard_samples, new_embedding, current_user_id,
                selected_indices, selected_labels, used_user_ids)
        
        # 4. 📦 Regular Samples로 나머지 채우기
        remaining_slots = batch_size - len(selected_indices)
        regular_added = self._fill_with_regular_samples(
            remaining_slots, selected_indices, selected_labels, used_user_ids)
        
        # 5. 📊 실제 구성 검증 및 리포트
        final_images = self._convert_indices_to_samples(selected_indices)
        self._report_final_composition(selected_labels, batch_size, pairs_created, hard_added, regular_added)
        
        return final_images, selected_labels

    def _create_limited_positive_pairs(self, target_pairs: int, selected_indices: List,
                                     selected_labels: List, used_user_ids: set) -> int:
        """제한된 수의 Positive Pairs만 생성"""
        print(f"🎯 [Positive] Creating exactly {target_pairs} positive pairs...")
        
        if target_pairs == 0:
            return 0
        
        user_groups = {}
        for i, item in enumerate(self.image_storage):
            user_id = item['user_id']
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(i)
        
        multi_sample_users = {uid: indices for uid, indices in user_groups.items() 
                             if len(indices) >= 2}
        
        if len(multi_sample_users) == 0:
            print(f"[Positive] ⚠️ No multi-sample users available")
            return 0
        
        available_users = list(multi_sample_users.keys())
        pairs_created = 0
        
        for _ in range(target_pairs):
            if not available_users:
                break
                
            unused_users = [uid for uid in available_users if uid not in used_user_ids]
            selected_user = random.choice(unused_users if unused_users else available_users)
            
            user_samples = multi_sample_users[selected_user]
            if len(user_samples) >= 2:
                pair = random.sample(user_samples, 2)
                selected_indices.extend(pair)
                selected_labels.extend([selected_user, selected_user])
                used_user_ids.add(selected_user)
                pairs_created += 1
                
                print(f"   ✅ Pair {pairs_created}: User {selected_user} (indices: {pair})")
        
        print(f"✅ [Positive] Created {pairs_created}/{target_pairs} positive pairs")
        return pairs_created

    def _select_controlled_hard_samples(self, target_hard: int, new_embedding: torch.Tensor,
                                      current_user_id: int, selected_indices: List,
                                      selected_labels: List, used_user_ids: set) -> int:
        """제어된 Hard Sample 선택"""
        print(f"💪 [Hard] Selecting {target_hard} hard samples...")
        
        if len(self.stored_embeddings) == 0:
            return 0
        
        available_indices = [i for i in range(len(self.image_storage)) 
                           if i not in selected_indices]
        
        if len(available_indices) == 0:
            return 0
        
        # 🔥 Hard Score 계산
        buffer_embeddings = np.array([self.stored_embeddings[i] for i in available_indices])
        new_emb = new_embedding.cpu().numpy().flatten()
        
        hard_candidates = []
        for idx, buffer_idx in enumerate(available_indices):
            buffer_emb = buffer_embeddings[idx].flatten()
            
            similarity = np.dot(new_emb, buffer_emb) / (
                np.linalg.norm(new_emb) * np.linalg.norm(buffer_emb) + 1e-8)
            
            user_id = self.image_storage[buffer_idx]['user_id']
            is_same_user = user_id == current_user_id
            
            # Hard Score 정의
            if is_same_user:
                hard_score = 1.0 - similarity  # 낮은 유사도일수록 높은 hard score
            else:
                hard_score = similarity       # 높은 유사도일수록 높은 hard score
                
            hard_candidates.append({
                'buffer_index': buffer_idx,
                'hard_score': hard_score,
                'similarity': similarity,
                'is_same_user': is_same_user,
                'user_id': user_id
            })
        
        hard_candidates.sort(key=lambda x: x['hard_score'], reverse=True)
        
        hard_added = 0
        for candidate in hard_candidates[:target_hard]:
            selected_indices.append(candidate['buffer_index'])
            selected_labels.append(candidate['user_id'])
            hard_added += 1
            
            sample_type = "Hard Positive" if candidate['is_same_user'] else "Hard Negative"
            print(f"   💪 Hard {hard_added}: User {candidate['user_id']} ({sample_type})")
        
        print(f"✅ [Hard] Selected {hard_added}/{target_hard} hard samples")
        return hard_added

    def _fill_with_regular_samples(self, remaining_slots: int, selected_indices: List,
                                 selected_labels: List, used_user_ids: set) -> int:
        """Regular Samples로 나머지 슬롯 채우기"""
        if remaining_slots <= 0:
            return 0
            
        print(f"📦 [Regular] Filling {remaining_slots} remaining slots...")
        
        available_indices = [i for i in range(len(self.image_storage)) 
                           if i not in selected_indices]
        
        if len(available_indices) == 0:
            return 0
        
        unused_user_indices = []
        used_user_indices = []
        
        for idx in available_indices:
            user_id = self.image_storage[idx]['user_id']
            if user_id not in used_user_ids:
                unused_user_indices.append(idx)
            else:
                used_user_indices.append(idx)
        
        priority_indices = unused_user_indices + used_user_indices
        selected_count = min(remaining_slots, len(priority_indices))
        chosen_indices = priority_indices[:selected_count]
        
        regular_added = 0
        for idx in chosen_indices:
            user_id = self.image_storage[idx]['user_id']
            selected_indices.append(idx)
            selected_labels.append(user_id)
            regular_added += 1
            used_user_ids.add(user_id)
        
        print(f"✅ [Regular] Added {regular_added}/{remaining_slots} regular samples")
        return regular_added

    def _report_final_composition(self, labels: List, batch_size: int, 
                                pairs_created: int, hard_added: int, regular_added: int):
        """최종 배치 구성 상세 리포트"""
        user_counts = {}
        for label in labels:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        actual_positive_pairs = sum(1 for count in user_counts.values() if count >= 2)
        actual_positive_samples = sum(count for count in user_counts.values() if count >= 2)
        
        positive_ratio = actual_positive_samples / batch_size if batch_size > 0 else 0
        hard_ratio = hard_added / batch_size if batch_size > 0 else 0
        
        print(f"📊 [Final] Achieved batch composition:")
        print(f"   Positive pairs: {actual_positive_pairs} pairs ({actual_positive_samples} samples, {positive_ratio:.1%})")
        print(f"   Hard samples: {hard_added} samples ({hard_ratio:.1%})")
        print(f"   Regular samples: {regular_added} samples")

    def sample_with_replacement(self, batch_size: int, new_embedding: torch.Tensor = None, 
                              current_user_id: int = None) -> Tuple[List, List]:
        """메인 샘플링 인터페이스"""
        if self.sampling_strategy == "controlled":
            return self.sample_with_controlled_composition(batch_size, new_embedding, current_user_id)
        else:
            return self.sample_with_controlled_composition(batch_size, new_embedding, current_user_id)

    def _convert_indices_to_samples(self, indices: List) -> List:
        """인덱스를 실제 이미지 샘플로 변환 (GPU 호환성 보장)"""
        images = []
        
        for idx in indices:
            base_image = self.image_storage[idx]['image'].clone()  # CPU에서 로드
            augmented = self._apply_differentiated_augmentation(base_image)
            # 🔥 모델 device로 이동
            augmented = augmented.to(self.device)
            images.append(augmented)
        return images

    def _setup_augmentation_transforms(self):
        """증강 변환 설정"""
        if not self.enable_augmentation or not self.aug_config:
            self.geometric_transform = None
            return
            
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

    def _apply_differentiated_augmentation(self, image, sample_info="", intensity="normal"):
        """차별화된 증강 적용"""
        if not self.enable_augmentation:
            return image
            
        result = image.clone()
        
        if (self.geometric_transform and 
            getattr(self.aug_config, 'enable_geometric', False) and
            np.random.random() < 0.3):
            result = self.geometric_transform(result)
        
        return result

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
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
            
        try:
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
        except Exception as e:
            print(f"[Buffer] ❌ Save failed: {e}")

    def _load_state(self):
        """상태 로드"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'rb') as f:
                saved_data = pickle.load(f)
                
            if FAISS_AVAILABLE and 'faiss_index_data' in saved_data:
                self.faiss_index = faiss.deserialize_index(saved_data['faiss_index_data'])
                self.metadata = saved_data.get('metadata', {})
                self.image_storage = saved_data.get('image_storage', [])
                self.stored_embeddings = saved_data.get('stored_embeddings', [])
                
                diversity_stats = self.get_diversity_stats()
                print(f"[Buffer] ✅ Restored {len(self.image_storage)} samples")
                
        except Exception as e:
            print(f"[Buffer] ❌ Load failed: {e}")
            self.faiss_index = None
            self.metadata = {}
            self.image_storage = []
            self.stored_embeddings = []

print("✅ CoconutReplayBuffer with Complete GPU/CPU Compatibility!")