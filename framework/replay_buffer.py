# framework/replay_buffer.py - 배치 기반 단순화 버전

"""
CoCoNut Simplified Replay Buffer

🔥 CHANGES:
- Removed all positive pair forcing logic
- Simplified batch composition
- Optimized for batch processing
- Added per-user sample limit
"""

import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class SimplifiedReplayBuffer:
    def __init__(self, config, storage_dir: Path, feature_dimension: int = 128):
        """단순화된 리플레이 버퍼"""
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension
        self.samples_per_user_limit = getattr(self.config, 'samples_per_user_limit', 3)
        
        # Faiss 인덱스 및 저장소
        self.image_storage = []
        self.faiss_index = None
        self.stored_embeddings = []
        self.metadata = {}
        self.feature_extractor = None
        self.device = 'cpu'
        
        # 하드 네거티브 비율 (CoconutSystem에서 설정)
        self.hard_negative_ratio = 0.3
        
        # 데이터 증강 설정
        self.enable_augmentation = False
        self.aug_config = None
        self._setup_augmentation_transforms()
        
        # 상태 로드
        self.state_file = self.storage_dir / 'buffer_state.pkl'
        self._load_state()
        
        print(f"[Buffer] 🥥 Simplified Replay Buffer initialized")
        print(f"[Buffer] Max size: {self.buffer_size}")
        print(f"[Buffer] Per-user limit: {self.samples_per_user_limit}")
        print(f"[Buffer] Current size: {len(self.image_storage)}")

    def add_if_diverse(self, image: torch.Tensor, user_id: int, embedding: torch.Tensor = None):
        """
        다양성 기반 추가 (단순화)
        
        Args:
            image: 이미지
            user_id: 사용자 ID
            embedding: 이미 계산된 임베딩 (없으면 계산)
        """
        # 사용자별 샘플 수 확인
        user_samples = [item for item in self.image_storage if item['user_id'] == user_id]
        
        if len(user_samples) >= self.samples_per_user_limit:
            print(f"[Buffer] User {user_id} already has {len(user_samples)} samples (limit: {self.samples_per_user_limit})")
            return False
        
        # 임베딩 계산
        if embedding is None:
            with torch.no_grad():
                embedding = self._extract_feature(image)
        
        # 다양성 체크
        if len(user_samples) > 0:
            max_similarity = self._compute_max_similarity_to_user(embedding, user_id)
            
            if max_similarity >= self.similarity_threshold:
                print(f"[Buffer] Sample too similar ({max_similarity:.3f} >= {self.similarity_threshold})")
                return False
        
        # 버퍼 공간 확보
        if len(self.image_storage) >= self.buffer_size:
            self._remove_least_diverse()
        
        # 저장
        self._store_sample(image, user_id, embedding)
        return True

    def sample_for_training(self, num_samples: int, current_embeddings: List[torch.Tensor], 
                          current_user_id: int) -> Tuple[List, List]:
        """
        학습을 위한 샘플링 (단순화)
        
        Args:
            num_samples: 필요한 샘플 수
            current_embeddings: 현재 배치의 임베딩들
            current_user_id: 현재 사용자 ID
            
        Returns:
            (images, labels)
        """
        if len(self.image_storage) == 0:
            return [], []
        
        # 하드 네거티브 수 계산
        num_hard = int(num_samples * self.hard_negative_ratio)
        num_random = num_samples - num_hard
        
        sampled_images = []
        sampled_labels = []
        used_indices = set()
        
        # 1. 하드 네거티브 마이닝
        if num_hard > 0 and current_embeddings:
            hard_samples = self._mine_hard_negatives_batch(
                current_embeddings, current_user_id, num_hard
            )
            
            for item in hard_samples:
                sampled_images.append(item['image'])
                sampled_labels.append(item['user_id'])
                used_indices.add(self.image_storage.index(item))
        
        # 2. 랜덤 샘플링
        available_indices = [i for i in range(len(self.image_storage)) 
                           if i not in used_indices]
        
        if available_indices and num_random > 0:
            random_indices = random.choices(available_indices, k=min(num_random, len(available_indices)))
            
            for idx in random_indices:
                item = self.image_storage[idx]
                sampled_images.append(item['image'])
                sampled_labels.append(item['user_id'])
        
        print(f"[Buffer] Sampled {len(sampled_images)} samples: "
              f"{len(hard_samples) if num_hard > 0 else 0} hard, "
              f"{len(sampled_images) - (len(hard_samples) if num_hard > 0 else 0)} random")
        
        return sampled_images, sampled_labels

    def _mine_hard_negatives_batch(self, query_embeddings: List[torch.Tensor], 
                                  exclude_user: int, num_samples: int) -> List[Dict]:
        """배치 쿼리로 하드 네거티브 마이닝"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
        
        # 평균 임베딩으로 쿼리
        query_tensor = torch.stack(query_embeddings).mean(dim=0, keepdim=True)
        query_np = query_tensor.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        # FAISS 검색
        k = min(num_samples * 3, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_np, k)
        
        # 다른 사용자의 어려운 샘플들 선택
        hard_samples = []
        for idx in indices[0]:
            if idx < len(self.image_storage):
                item = self.image_storage[idx]
                if item['user_id'] != exclude_user:
                    hard_samples.append(item)
                    if len(hard_samples) >= num_samples:
                        break
        
        return hard_samples

    def _compute_max_similarity_to_user(self, embedding: torch.Tensor, user_id: int) -> float:
        """특정 사용자의 샘플들과의 최대 유사도"""
        user_indices = [i for i, item in enumerate(self.image_storage) 
                       if item['user_id'] == user_id]
        
        if not user_indices:
            return 0.0
        
        if FAISS_AVAILABLE and self.faiss_index:
            # FAISS 사용
            query = embedding.cpu().numpy().astype('float32').reshape(1, -1)
            faiss.normalize_L2(query)
            
            # 사용자 샘플들만 검색하도록 임시 인덱스 생성
            user_embeddings = np.array([self.stored_embeddings[i] for i in user_indices])
            temp_index = faiss.IndexFlatIP(self.feature_dimension)
            temp_index.add(user_embeddings)
            
            similarities, _ = temp_index.search(query, 1)
            return similarities[0][0]
        else:
            # PyTorch 폴백
            max_sim = 0.0
            query_norm = F.normalize(embedding.flatten(), dim=0)
            
            for idx in user_indices:
                stored_emb = torch.tensor(self.stored_embeddings[idx])
                stored_norm = F.normalize(stored_emb.flatten(), dim=0)
                sim = torch.cosine_similarity(query_norm.unsqueeze(0), 
                                            stored_norm.unsqueeze(0)).item()
                max_sim = max(max_sim, sim)
            
            return max_sim

    def _store_sample(self, image: torch.Tensor, user_id: int, embedding: torch.Tensor):
        """샘플 저장"""
        unique_id = len(self.image_storage)
        
        # 이미지 저장
        self.image_storage.append({
            'image': image.cpu().clone(),
            'user_id': user_id,
            'id': unique_id
        })
        
        # 임베딩 저장
        embedding_np = embedding.cpu().numpy().astype('float32')
        if FAISS_AVAILABLE:
            faiss.normalize_L2(embedding_np)
        self.stored_embeddings.append(embedding_np.copy())
        
        # Faiss 인덱스 업데이트
        if self.faiss_index is None:
            self._initialize_faiss()
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            self.faiss_index.add_with_ids(embedding_np.reshape(1, -1), np.array([unique_id]))
        
        # 메타데이터
        self.metadata[unique_id] = {'user_id': user_id}
        
        print(f"[Buffer] Stored sample {unique_id} for user {user_id}. "
              f"Buffer: {len(self.image_storage)}/{self.buffer_size}")

    def _remove_least_diverse(self):
        """가장 중복되는 샘플 제거"""
        if len(self.image_storage) < 2:
            return
        
        # 각 샘플의 평균 유사도 계산
        diversity_scores = []
        
        for i in range(len(self.image_storage)):
            if FAISS_AVAILABLE and self.faiss_index:
                query = self.stored_embeddings[i].reshape(1, -1)
                similarities, _ = self.faiss_index.search(query, min(10, len(self.image_storage)))
                avg_similarity = similarities[0][1:].mean()  # 자기 자신 제외
            else:
                avg_similarity = 0.0
            
            diversity_scores.append(avg_similarity)
        
        # 가장 유사도가 높은 (다양성이 낮은) 샘플 제거
        remove_idx = np.argmax(diversity_scores)
        removed_item = self.image_storage[remove_idx]
        
        # 제거
        del self.image_storage[remove_idx]
        del self.stored_embeddings[remove_idx]
        if removed_item['id'] in self.metadata:
            del self.metadata[removed_item['id']]
        
        # Faiss 인덱스 재구성
        self._rebuild_faiss_index()
        
        print(f"[Buffer] Removed least diverse sample from user {removed_item['user_id']}")

    def update_hard_negative_ratio(self, ratio: float):
        """하드 네거티브 비율 업데이트"""
        self.hard_negative_ratio = ratio
        print(f"[Buffer] Hard negative ratio updated to {ratio:.1%}")

    def set_feature_extractor(self, model):
        """특징 추출기 설정"""
        self.feature_extractor = model
        if model is not None:
            self.device = next(model.parameters()).device
            print(f"[Buffer] Feature extractor set (device: {self.device})")

    def _extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """특징 추출"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
        
        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor.getFeatureCode(image)
        
        return features.squeeze(0)

    def _initialize_faiss(self):
        """Faiss 인덱스 초기화"""
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(self.feature_dimension)
            self.faiss_index = faiss.IndexIDMap(index)
            print(f"[Buffer] Faiss index initialized")
        else:
            self.faiss_index = None

    def _rebuild_faiss_index(self):
        """Faiss 인덱스 재구성"""
        if not FAISS_AVAILABLE or not self.stored_embeddings:
            return
        
        self._initialize_faiss()
        
        for i, (embedding, item) in enumerate(zip(self.stored_embeddings, self.image_storage)):
            embedding_np = np.array(embedding).astype('float32').reshape(1, -1)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embedding_np)
            self.faiss_index.add_with_ids(embedding_np, np.array([item['id']]))

    def _setup_augmentation_transforms(self):
        """데이터 증강 설정 (기존과 동일)"""
        self.augmentation_transforms = None

    def get_statistics(self) -> Dict:
        """버퍼 통계"""
        user_distribution = {}
        for item in self.image_storage:
            user_id = item['user_id']
            user_distribution[user_id] = user_distribution.get(user_id, 0) + 1
        
        return {
            'total_samples': len(self.image_storage),
            'unique_users': len(user_distribution),
            'user_distribution': user_distribution,
            'buffer_utilization': len(self.image_storage) / self.buffer_size,
            'avg_samples_per_user': len(self.image_storage) / len(user_distribution) if user_distribution else 0
        }

    def _save_state(self):
        """상태 저장"""
        save_data = {
            'image_storage': self.image_storage,
            'stored_embeddings': self.stored_embeddings,
            'metadata': self.metadata,
            'feature_dim': self.feature_dimension
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[Buffer] State saved: {len(self.image_storage)} samples")

    def _load_state(self):
        """상태 로드"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'rb') as f:
                save_data = pickle.load(f)
            
            self.image_storage = save_data.get('image_storage', [])
            self.stored_embeddings = save_data.get('stored_embeddings', [])
            self.metadata = save_data.get('metadata', {})
            
            if self.image_storage:
                self._rebuild_faiss_index()
            
            print(f"[Buffer] State loaded: {len(self.image_storage)} samples")
        except Exception as e:
            print(f"[Buffer] Failed to load state: {e}")

# 호환성을 위한 별칭
CoconutReplayBuffer = SimplifiedReplayBuffer