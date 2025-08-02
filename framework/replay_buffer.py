# framework/replay_buffer.py - CCNet 스타일로 수정된 버전

"""
CoCoNut Replay Buffer with Loop Closure Support

🔥 FEATURES:
- Priority queue for loop closure samples
- User-specific sample retrieval
- Enhanced sampling strategies
- Even-count maintenance for SupConLoss
- CCNet-style pair handling
"""

import os
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

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
from PIL import Image

class ReplayBuffer:
    def __init__(self, config, storage_dir: Path, feature_dimension: int = 128):
        """리플레이 버퍼 with Loop Closure support + CCNet style"""
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension
        self.samples_per_user_limit = getattr(self.config, 'samples_per_user_limit', 4)  # 짝수로 설정
        self.min_samples_new_user = getattr(self.config, 'min_samples_new_user', 2)     # 짝수로 설정
        
        # Faiss 인덱스 및 저장소
        self.image_storage = []
        self.faiss_index = None
        self.stored_embeddings = []
        self.metadata = {}
        self.feature_extractor = None
        self.device = 'cpu'
        
        # 하드 네거티브 비율
        self.hard_negative_ratio = 0.3
        
        # 🔥 Loop Closure Priority Queue
        self.priority_queue = []  # 우선순위 샘플들
        self.priority_users = set()  # 우선순위 사용자 ID들
        
        # 데이터 증강 설정
        self.enable_augmentation = False
        self.aug_config = None
        self._setup_augmentation_transforms()
        
        # 상태 로드
        self.state_file = self.storage_dir / 'buffer_state.pkl'
        self._load_state()
        
        print(f"[Buffer] 🥥 Replay Buffer initialized (CCNet style)")
        print(f"[Buffer] Max size: {self.buffer_size}")
        print(f"[Buffer] Per-user limit: {self.samples_per_user_limit} (even count)")
        print(f"[Buffer] Min samples for new user: {self.min_samples_new_user}")
        print(f"[Buffer] Current size: {len(self.image_storage)}")
        print(f"[Buffer] 🔥 Loop Closure support: ENABLED")

    def add_if_diverse(self, image: torch.Tensor, user_id: int, embedding: torch.Tensor = None):
        """
        다양성 기반 추가 - 짝수 유지 고려
        
        Args:
            image: 이미지
            user_id: 사용자 ID
            embedding: 이미 계산된 임베딩 (없으면 계산)
        """
        # 사용자별 샘플 수 확인
        user_samples = [item for item in self.image_storage if item['user_id'] == user_id]
        current_count = len(user_samples)
        
        if current_count >= self.samples_per_user_limit:
            print(f"[Buffer] User {user_id} already has {current_count} samples (limit: {self.samples_per_user_limit})")
            return False
        
        # 임베딩 계산
        if embedding is None:
            with torch.no_grad():
                embedding = self._extract_feature(image)
        
        # 홀수인 경우 무조건 추가 (짝수로 만들기)
        if current_count % 2 == 1:
            # 버퍼 공간 확보
            if len(self.image_storage) >= self.buffer_size:
                self._remove_least_diverse_even()
            
            self._store_sample(image, user_id, embedding)
            return True
        
        # 짝수인 경우 다양성 체크
        if current_count > 0:
            max_similarity = self._compute_max_similarity_to_user(embedding, user_id)
            
            if max_similarity >= self.similarity_threshold:
                print(f"[Buffer] Sample too similar ({max_similarity:.3f} >= {self.similarity_threshold})")
                return False
        
        # 버퍼 공간 확보
        if len(self.image_storage) >= self.buffer_size:
            self._remove_least_diverse_even()
        
        # 저장
        self._store_sample(image, user_id, embedding)
        return True

    def add_sample_direct(self, image: torch.Tensor, user_id: int, embedding: torch.Tensor):
        """직접 샘플 추가 (다양성 체크 없이)"""
        # 버퍼 공간 확보
        if len(self.image_storage) >= self.buffer_size:
            self._remove_least_diverse_even()
        
        # 저장
        self._store_sample(image, user_id, embedding)
        return True

    def sample_for_training(self, num_samples: int, current_embeddings: List[torch.Tensor], 
                          current_user_id: int) -> Tuple[List, List]:
        """학습을 위한 샘플링 - 🔥 Loop Closure 우선순위 지원"""
        if len(self.image_storage) == 0:
            return [], []
        
        sampled_images = []
        sampled_labels = []
        used_indices = set()
        
        # 🔥 1. Priority Queue 처리 (Loop Closure)
        if self.priority_queue:
            print(f"[Buffer] Processing {len(self.priority_queue)} priority samples")
            
            for priority_item in self.priority_queue[:num_samples]:
                sampled_images.append(priority_item['image'])
                sampled_labels.append(priority_item['user_id'])
                
            # 사용한 것들은 제거
            self.priority_queue = self.priority_queue[len(sampled_images):]
            
            if len(sampled_images) >= num_samples:
                return sampled_images[:num_samples], sampled_labels[:num_samples]
        
        # 2. 하드 네거티브 마이닝
        remaining_samples = num_samples - len(sampled_images)
        num_hard = int(remaining_samples * self.hard_negative_ratio)
        
        if num_hard > 0 and current_embeddings:
            hard_samples = self._mine_hard_negatives_batch(
                current_embeddings, current_user_id, num_hard
            )
            
            for item in hard_samples:
                if len(sampled_images) < num_samples:
                    sampled_images.append(item['image'])
                    sampled_labels.append(item['user_id'])
                    used_indices.add(item['id'])
        
        # 3. 랜덤 샘플링
        remaining_samples = num_samples - len(sampled_images)
        if remaining_samples > 0:
            available_indices = []
            for i, storage_item in enumerate(self.image_storage):
                if storage_item['id'] not in used_indices:
                    available_indices.append(i)
            
            if available_indices:
                random_indices = random.choices(available_indices, 
                                              k=min(remaining_samples, len(available_indices)))
                
                for idx in random_indices:
                    if len(sampled_images) < num_samples:
                        item = self.image_storage[idx]
                        sampled_images.append(item['image'])
                        sampled_labels.append(item['user_id'])
        
        print(f"[Buffer] Sampled {len(sampled_images)} samples: "
              f"{len(self.priority_queue)} priority, "
              f"{num_hard} hard, "
              f"{len(sampled_images) - len(self.priority_queue) - num_hard} random")
        
        return sampled_images, sampled_labels

    def sample_for_training_even(self, num_samples: int, current_user_id: int) -> Tuple[List, List]:
        """
        학습을 위한 샘플링 - CCNet 스타일 (짝수 보장)
        
        Args:
            num_samples: 요청된 샘플 수
            current_user_id: 현재 사용자 ID (제외용)
            
        Returns:
            (images, labels) - 짝수개 보장
        """
        if len(self.image_storage) == 0:
            return [], []
        
        sampled_images = []
        sampled_labels = []
        used_indices = set()
        
        # 1. Priority Queue 처리
        if self.priority_queue:
            for priority_item in self.priority_queue[:num_samples]:
                sampled_images.append(priority_item['image'])
                sampled_labels.append(priority_item['user_id'])
                
            self.priority_queue = self.priority_queue[len(sampled_images):]
            
            if len(sampled_images) >= num_samples:
                return self._ensure_even_count(sampled_images[:num_samples], 
                                              sampled_labels[:num_samples])
        
        # 2. 라벨별 그룹화
        label_groups = defaultdict(list)
        for i, item in enumerate(self.image_storage):
            if item['user_id'] != current_user_id:  # 현재 사용자 제외
                label_groups[item['user_id']].append(i)
        
        # 3. 각 라벨에서 짝수개씩 샘플링
        for user_id, indices in label_groups.items():
            if len(sampled_images) >= num_samples:
                break
                
            if len(indices) >= 2:
                # 짝수개 선택
                num_to_sample = min(len(indices) // 2 * 2, 4)  # 최대 4개
                selected_indices = random.sample(indices, num_to_sample)
                
                for idx in selected_indices:
                    if len(sampled_images) < num_samples:
                        item = self.image_storage[idx]
                        sampled_images.append(item['image'])
                        sampled_labels.append(item['user_id'])
                        used_indices.add(idx)
        
        # 4. 부족하면 추가 샘플링
        remaining = num_samples - len(sampled_images)
        if remaining > 0:
            available_indices = [i for i in range(len(self.image_storage)) 
                               if i not in used_indices and 
                               self.image_storage[i]['user_id'] != current_user_id]
            
            if available_indices:
                # 짝수개로 맞추기
                if remaining % 2 == 1:
                    remaining += 1
                    
                additional = random.sample(available_indices, 
                                         min(remaining, len(available_indices)))
                
                for idx in additional:
                    if len(sampled_images) < num_samples:
                        item = self.image_storage[idx]
                        sampled_images.append(item['image'])
                        sampled_labels.append(item['user_id'])
        
        # 5. 최종적으로 짝수로 조정
        return self._ensure_even_count(sampled_images, sampled_labels)

    def _ensure_even_count(self, images: List, labels: List) -> Tuple[List, List]:
        """짝수개로 보장"""
        if len(images) % 2 == 1 and len(images) > 0:
            images.pop()
            labels.pop()
        
        print(f"[Buffer] Sampled {len(images)} samples (even count ensured)")
        return images, labels

    def _remove_least_diverse_even(self):
        """다양성 기반 삭제 - 짝수 유지"""
        if len(self.image_storage) < 2:
            return
        
        # 사용자별 그룹화
        user_samples = defaultdict(list)
        for i, item in enumerate(self.image_storage):
            user_samples[item['user_id']].append(i)
        
        # 삭제 대상 선정
        candidate_users = []
        for user_id, indices in user_samples.items():
            if user_id in self.priority_users or len(indices) < 2:
                continue
                
            # 평균 다양성 계산
            avg_div = self._calculate_user_average_diversity(indices)
            candidate_users.append((user_id, indices, avg_div))
        
        if not candidate_users:
            # 모든 사용자가 보호되거나 1개씩만 있음
            # 가장 오래된 비우선순위 사용자 찾기
            for user_id, indices in user_samples.items():
                if user_id not in self.priority_users and len(indices) >= 2:
                    candidate_users.append((user_id, indices, 1.0))
                    break
        
        if not candidate_users:
            print("[Buffer] No removable samples while maintaining even counts")
            return
        
        # 가장 다양성 낮은 사용자 선택
        candidate_users.sort(key=lambda x: x[2])
        selected_user, indices, _ = candidate_users[0]
        
        # 삭제 개수 결정 (짝수 유지)
        current_count = len(indices)
        if current_count % 2 == 0:
            # 짝수 → 2개 삭제
            num_to_remove = 2
        else:
            # 홀수 → 1개 삭제 (짝수로)
            num_to_remove = 1
        
        # 가장 유사한 샘플들 찾기
        if num_to_remove == 1:
            remove_indices = [self._find_most_redundant_single(indices)]
        else:
            remove_indices = self._find_most_similar_pair(indices)
        
        # 삭제 실행
        for idx in sorted(remove_indices, reverse=True):
            removed_id = self.image_storage[idx]['id']
            del self.image_storage[idx]
            del self.stored_embeddings[idx]
            if removed_id in self.metadata:
                del self.metadata[removed_id]
        
        print(f"[Buffer] Removed {num_to_remove} samples from user {selected_user} "
              f"({current_count} → {current_count-num_to_remove})")
        
        self._rebuild_faiss_index()

    def _find_most_redundant_single(self, indices: List[int]) -> int:
        """가장 중복된 단일 샘플 찾기"""
        max_avg_sim = -1
        most_redundant = indices[0]
        
        for i, idx in enumerate(indices):
            similarities = []
            for j, other_idx in enumerate(indices):
                if i != j:
                    sim = self._compute_similarity_by_idx(idx, other_idx)
                    similarities.append(sim)
            
            avg_sim = np.mean(similarities) if similarities else 0
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                most_redundant = idx
        
        return most_redundant

    def _find_most_similar_pair(self, indices: List[int]) -> List[int]:
        """가장 유사한 페어 찾기"""
        max_sim = -1
        best_pair = [indices[0], indices[1]] if len(indices) >= 2 else indices[:1]
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = self._compute_similarity_by_idx(indices[i], indices[j])
                if sim > max_sim:
                    max_sim = sim
                    best_pair = [indices[i], indices[j]]
        
        return best_pair

    def _compute_similarity_by_idx(self, idx1: int, idx2: int) -> float:
        """인덱스로 유사도 계산"""
        emb1 = torch.tensor(self.stored_embeddings[idx1])
        emb2 = torch.tensor(self.stored_embeddings[idx2])
        
        similarity = F.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()
        
        return similarity

    def _calculate_user_average_diversity(self, indices: List[int]) -> float:
        """사용자의 평균 다양성 계산"""
        if len(indices) < 2:
            return 1.0  # 높은 다양성으로 가정
        
        total_sim = 0
        count = 0
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = self._compute_similarity_by_idx(indices[i], indices[j])
                total_sim += sim
                count += 1
        
        avg_sim = total_sim / count if count > 0 else 0
        return 1 - avg_sim  # 다양성 = 1 - 유사도

    def add_priority_samples(self, user_ids: List[int], priority_weight: float = 2.0):
        """
        🔥 Loop Closure용 우선순위 샘플 추가
        
        Args:
            user_ids: 우선순위로 추가할 사용자 ID들
            priority_weight: 우선순위 가중치
        """
        print(f"[Buffer] Adding priority samples for users: {user_ids}")
        
        for user_id in user_ids:
            user_samples = self.get_user_samples(user_id)
            
            for sample_dict in user_samples:
                # 우선순위 큐에 추가
                priority_item = sample_dict.copy()
                priority_item['priority_weight'] = priority_weight
                self.priority_queue.append(priority_item)
            
            self.priority_users.add(user_id)
            print(f"[Buffer] Added {len(user_samples)} priority samples for user {user_id}")
        
        # 우선순위 큐 정렬 (가중치 높은 것부터)
        self.priority_queue.sort(key=lambda x: x.get('priority_weight', 1.0), reverse=True)

    def get_user_samples(self, user_id: int) -> List[Dict]:
        """
        🔥 특정 사용자의 모든 샘플 반환
        
        Returns:
            List of sample dictionaries
        """
        user_samples = []
        for item in self.image_storage:
            if item['user_id'] == user_id:
                user_samples.append(item)
        return user_samples

    def get_user_sample_images(self, user_id: int) -> List[torch.Tensor]:
        """
        🔥 특정 사용자의 이미지만 반환 (Loop Closure용)
        
        Returns:
            List of image tensors
        """
        return [item['image'] for item in self.image_storage if item['user_id'] == user_id]

    def clear_priority_queue(self):
        """🔥 우선순위 큐 초기화"""
        self.priority_queue = []
        self.priority_users.clear()
        print("[Buffer] Priority queue cleared")

    def _mine_hard_negatives_batch(self, query_embeddings: List[torch.Tensor], 
                                  exclude_user: int, num_samples: int) -> List[Dict]:
        """배치 쿼리로 하드 네거티브 마이닝"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
        
        # 평균 임베딩으로 쿼리
        query_tensor = torch.stack(query_embeddings).mean(dim=0, keepdim=True)
        query_np = query_tensor.cpu().numpy().astype('float32')
        
        if len(query_np.shape) == 3:
            # (1, 1, feature_dim) -> (1, feature_dim)
            query_np = query_np.squeeze(0)
        elif len(query_np.shape) == 1:
            # (feature_dim,) -> (1, feature_dim)
            query_np = query_np.reshape(1, -1)
        
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
            query = embedding.cpu().numpy().astype('float32')
            if len(query.shape) == 1:
                query = query.reshape(1, -1)
            elif len(query.shape) == 3:
                query = query.squeeze(0)
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
            'id': unique_id,
            'timestamp': len(self.image_storage)  # 추가 순서
        })
        
        # 임베딩 저장
        embedding_np = embedding.cpu().numpy().astype('float32')
        
        if len(embedding_np.shape) == 1:
            embedding_np = embedding_np.reshape(1, -1)
        
        if FAISS_AVAILABLE:
            faiss.normalize_L2(embedding_np)
        
        self.stored_embeddings.append(embedding_np.squeeze().copy())
        
        # Faiss 인덱스 업데이트
        if self.faiss_index is None:
            self._initialize_faiss()
        
        if FAISS_AVAILABLE and self.faiss_index is not None:
            self.faiss_index.add_with_ids(embedding_np, np.array([unique_id]))
        
        # 메타데이터
        self.metadata[unique_id] = {
            'user_id': user_id,
            'priority': user_id in self.priority_users
        }
        
        # 현재 사용자의 샘플 수 확인
        user_count = sum(1 for item in self.image_storage if item['user_id'] == user_id)
        
        print(f"[Buffer] Stored sample {unique_id} for user {user_id}. "
              f"Buffer: {len(self.image_storage)}/{self.buffer_size}, "
              f"User samples: {user_count}")

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
            embedding_np = np.array(embedding).astype('float32')
            if len(embedding_np.shape) == 1:
                embedding_np = embedding_np.reshape(1, -1)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embedding_np)
            self.faiss_index.add_with_ids(embedding_np, np.array([item['id']]))
        
        print(f"[Buffer] Faiss index rebuilt with {len(self.stored_embeddings)} samples")

    def _setup_augmentation_transforms(self):
        """데이터 증강 설정"""
        self.augmentation_transforms = None

    def get_statistics(self) -> Dict:
        """버퍼 통계 - 🔥 Loop Closure 정보 추가"""
        user_distribution = {}
        even_count_users = 0
        
        for item in self.image_storage:
            user_id = item['user_id']
            user_distribution[user_id] = user_distribution.get(user_id, 0) + 1
        
        # 짝수 개수를 가진 사용자 수 계산
        for count in user_distribution.values():
            if count % 2 == 0:
                even_count_users += 1
        
        return {
            'total_samples': len(self.image_storage),
            'unique_users': len(user_distribution),
            'user_distribution': user_distribution,
            'buffer_utilization': len(self.image_storage) / self.buffer_size,
            'avg_samples_per_user': len(self.image_storage) / len(user_distribution) if user_distribution else 0,
            'priority_queue_size': len(self.priority_queue),
            'priority_users': list(self.priority_users),
            'even_count_users': even_count_users,
            'even_count_percentage': even_count_users / len(user_distribution) if user_distribution else 0
        }

    def _save_state(self):
        """상태 저장 - 🔥 우선순위 정보 포함"""
        save_data = {
            'image_storage': self.image_storage,
            'stored_embeddings': self.stored_embeddings,
            'metadata': self.metadata,
            'feature_dim': self.feature_dimension,
            'priority_queue': self.priority_queue,
            'priority_users': list(self.priority_users),
            'buffer_config': {
                'max_size': self.buffer_size,
                'samples_per_user_limit': self.samples_per_user_limit,
                'similarity_threshold': self.similarity_threshold,
                'hard_negative_ratio': self.hard_negative_ratio
            }
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        stats = self.get_statistics()
        print(f"[Buffer] State saved: {len(self.image_storage)} samples, "
              f"{len(self.priority_queue)} priority items, "
              f"{stats['even_count_percentage']:.1%} even count users")

    def _load_state(self):
        """상태 로드 - 🔥 우선순위 정보 포함"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'rb') as f:
                save_data = pickle.load(f)
            
            self.image_storage = save_data.get('image_storage', [])
            self.stored_embeddings = save_data.get('stored_embeddings', [])
            self.metadata = save_data.get('metadata', {})
            self.priority_queue = save_data.get('priority_queue', [])
            self.priority_users = set(save_data.get('priority_users', []))
            
            # 설정 복원
            buffer_config = save_data.get('buffer_config', {})
            if buffer_config:
                self.hard_negative_ratio = buffer_config.get('hard_negative_ratio', 0.3)
            
            if self.image_storage:
                self._rebuild_faiss_index()
            
            stats = self.get_statistics()
            print(f"[Buffer] State loaded: {len(self.image_storage)} samples, "
                  f"{len(self.priority_queue)} priority items, "
                  f"{stats['even_count_percentage']:.1%} even count users")
        except Exception as e:
            print(f"[Buffer] Failed to load state: {e}")
            print("[Buffer] Starting with empty buffer")