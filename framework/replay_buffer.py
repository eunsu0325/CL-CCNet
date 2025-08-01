import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch.nn.functional as F
import pickle
import random

# Faiss 임포트 시도
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[Buffer] Warning: Faiss not available, using CPU-based similarity computation")


class EnhancedReplayBuffer:
    """
    개선된 리플레이 버퍼
    - 새 사용자는 2장 저장
    - 다양성 기반 샘플링
    - 하드 네거티브 마이닝
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # 기본 설정
        self.max_buffer_size = config.get('max_buffer_size', 1000)
        self.samples_per_user_limit = config.get('samples_per_user_limit', 5)
        self.min_samples_new_user = config.get('min_samples_new_user', 2)  # 새 사용자는 2장
        
        # 다양성 설정
        self.maximize_diversity = config.get('maximize_diversity', True)
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        
        # 저장 경로
        self.storage_path = config.get('storage_path', './replay_buffer')
        os.makedirs(self.storage_path, exist_ok=True)
        
        # 저장소
        self.image_storage = []  # {'image': tensor, 'label': int, 'embedding': tensor}
        self.user_sample_count = defaultdict(int)
        self.feature_extractor = None
        
        # 하드 네거티브 마이닝
        self.hard_negative_ratio = 0.3
        
        # Faiss 인덱스 (빠른 유사도 검색)
        self.use_faiss = FAISS_AVAILABLE and config.get('use_faiss', True)
        self.faiss_index = None
        self.feature_dim = 128  # CCNet feature dimension
        
        # 통계
        self.total_added = 0
        self.total_rejected = 0
        
        self._load_buffer()
    
    def set_feature_extractor(self, model):
        """특징 추출 모델 설정"""
        self.feature_extractor = model
        print(f"[Buffer] Feature extractor set (device: {next(model.parameters()).device})")
    
    def add_if_diverse(self, image: torch.Tensor, label: int, 
                      embedding: Optional[torch.Tensor] = None) -> bool:
        """
        다양성 체크 후 버퍼에 추가
        새 사용자는 최소 2장 보장
        """
        # 특징 추출
        if embedding is None and self.feature_extractor is not None:
            with torch.no_grad():
                embedding = self.feature_extractor(image.unsqueeze(0))
                embedding = F.normalize(embedding, p=2, dim=1).squeeze()
        
        # 해당 사용자의 현재 샘플 수
        user_samples = [item for item in self.image_storage if item['label'] == label]
        user_count = len(user_samples)
        
        # 새 사용자인 경우 최소 샘플 수 보장
        if user_count < self.min_samples_new_user:
            # 첫 번째 샘플은 무조건 저장
            if user_count == 0:
                return self._add_sample(image, label, embedding)
            
            # 두 번째 샘플은 약간의 다양성 체크
            if self.maximize_diversity and embedding is not None:
                max_similarity = self._compute_max_similarity_to_user(embedding, label)
                # 새 사용자는 더 낮은 임계값 사용 (더 쉽게 통과)
                relaxed_threshold = min(0.95, self.similarity_threshold + 0.1)
                
                if max_similarity >= relaxed_threshold:
                    print(f"[Buffer] New user sample too similar ({max_similarity:.3f} >= {relaxed_threshold})")
                    self.total_rejected += 1
                    return False
            
            return self._add_sample(image, label, embedding)
        
        # 기존 사용자의 경우 일반 다양성 체크
        if user_count >= self.samples_per_user_limit:
            # 사용자별 한계 도달
            if self.maximize_diversity and embedding is not None:
                # 가장 유사한 샘플 교체 고려
                return self._replace_least_diverse(image, label, embedding)
            return False
        
        # 다양성 체크
        if self.maximize_diversity and embedding is not None:
            max_similarity = self._compute_max_similarity_to_user(embedding, label)
            
            if max_similarity >= self.similarity_threshold:
                print(f"[Buffer] Sample too similar ({max_similarity:.3f} >= {self.similarity_threshold})")
                self.total_rejected += 1
                return False
        
        return self._add_sample(image, label, embedding)
    
    def _add_sample(self, image: torch.Tensor, label: int, 
                    embedding: Optional[torch.Tensor]) -> bool:
        """샘플을 버퍼에 추가"""
        # 버퍼 크기 확인
        if len(self.image_storage) >= self.max_buffer_size:
            # 가장 오래된 샘플 제거 (FIFO)
            removed = self.image_storage.pop(0)
            self.user_sample_count[removed['label']] -= 1
            
            # Faiss 인덱스 재구성 필요
            if self.use_faiss and self.faiss_index is not None:
                self._rebuild_faiss_index()
        
        # 새 샘플 추가
        sample = {
            'image': image.cpu(),
            'label': label,
            'embedding': embedding.cpu() if embedding is not None else None,
            'index': len(self.image_storage)
        }
        
        self.image_storage.append(sample)
        self.user_sample_count[label] += 1
        self.total_added += 1
        
        # Faiss 인덱스 업데이트
        if self.use_faiss and embedding is not None:
            self._update_faiss_index(embedding)
        
        print(f"[Buffer] Stored sample {len(self.image_storage)-1} for user {label}. "
              f"Buffer: {len(self.image_storage)}/{self.max_buffer_size}")
        
        return True
    
    def _compute_max_similarity_to_user(self, embedding: torch.Tensor, user_id: int) -> float:
        """특정 사용자의 샘플들과의 최대 유사도 계산"""
        user_samples = [item for item in self.image_storage 
                       if item['label'] == user_id and item['embedding'] is not None]
        
        if not user_samples:
            return 0.0
        
        max_similarity = 0.0
        embedding = F.normalize(embedding.view(1, -1), p=2, dim=1)
        
        for sample in user_samples:
            stored_embedding = sample['embedding']
            if stored_embedding is not None:
                stored_embedding = F.normalize(stored_embedding.view(1, -1), p=2, dim=1)
                similarity = torch.mm(embedding, stored_embedding.t()).item()
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _replace_least_diverse(self, image: torch.Tensor, label: int, 
                              embedding: torch.Tensor) -> bool:
        """가장 덜 다양한 샘플을 새 샘플로 교체"""
        user_samples = [(i, item) for i, item in enumerate(self.image_storage) 
                       if item['label'] == label]
        
        if len(user_samples) < self.samples_per_user_limit:
            return self._add_sample(image, label, embedding)
        
        # 각 샘플의 다양성 점수 계산
        diversity_scores = []
        for idx, sample in user_samples:
            if sample['embedding'] is not None:
                # 다른 샘플들과의 평균 거리
                distances = []
                for _, other in user_samples:
                    if other['embedding'] is not None and other != sample:
                        sim = F.cosine_similarity(
                            sample['embedding'].view(1, -1),
                            other['embedding'].view(1, -1)
                        ).item()
                        distances.append(1 - sim)
                
                diversity_score = np.mean(distances) if distances else 0
                diversity_scores.append((idx, diversity_score))
        
        if diversity_scores:
            # 가장 낮은 다양성 점수를 가진 샘플 제거
            idx_to_remove = min(diversity_scores, key=lambda x: x[1])[0]
            self.image_storage.pop(idx_to_remove)
            self.user_sample_count[label] -= 1
            
            # 새 샘플 추가
            return self._add_sample(image, label, embedding)
        
        return False
    
    def sample_batch(self, batch_size: int, current_batch_labels: List[int]) -> List[Dict]:
        """
        학습을 위한 배치 샘플링
        하드 네거티브 마이닝 포함
        """
        if len(self.image_storage) == 0:
            return []
        
        sampled = []
        current_labels_set = set(current_batch_labels)
        
        # 하드 네거티브 샘플 수 계산
        num_hard = int(batch_size * self.hard_negative_ratio)
        num_random = batch_size - num_hard
        
        # 하드 네거티브 샘플링
        if num_hard > 0 and self.use_faiss and self.faiss_index is not None:
            hard_samples = self._sample_hard_negatives(current_labels_set, num_hard)
            sampled.extend(hard_samples)
        
        # 랜덤 샘플링 (나머지)
        num_random = batch_size - len(sampled)
        if num_random > 0:
            # 현재 배치와 다른 라벨 우선
            other_label_samples = [s for s in self.image_storage 
                                 if s['label'] not in current_labels_set]
            
            if len(other_label_samples) >= num_random:
                random_samples = random.sample(other_label_samples, num_random)
            else:
                # 부족하면 전체에서 샘플링
                random_samples = random.sample(self.image_storage, 
                                             min(num_random, len(self.image_storage)))
            
            sampled.extend(random_samples)
        
        return sampled[:batch_size]
    
    def _sample_hard_negatives(self, current_labels: set, num_samples: int) -> List[Dict]:
        """하드 네거티브 샘플링 (가장 헷갈리는 샘플)"""
        # 구현 단순화 - 랜덤 샘플링으로 대체
        other_samples = [s for s in self.image_storage if s['label'] not in current_labels]
        if len(other_samples) > num_samples:
            return random.sample(other_samples, num_samples)
        return other_samples
    
    def _update_faiss_index(self, embedding: torch.Tensor):
        """Faiss 인덱스 업데이트"""
        if not self.use_faiss:
            return
        
        if self.faiss_index is None:
            # 첫 번째 샘플에서 인덱스 초기화
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)  # Inner Product
            print("[Buffer] Faiss index initialized")
        
        # 정규화된 임베딩 추가
        embedding_np = F.normalize(embedding, p=2, dim=0).cpu().numpy()
        self.faiss_index.add(embedding_np.reshape(1, -1))
    
    def _rebuild_faiss_index(self):
        """Faiss 인덱스 재구성"""
        if not self.use_faiss or self.faiss_index is None:
            return
        
        self.faiss_index.reset()
        
        for sample in self.image_storage:
            if sample['embedding'] is not None:
                embedding_np = F.normalize(sample['embedding'], p=2, dim=0).numpy()
                self.faiss_index.add(embedding_np.reshape(1, -1))
    
    def save_buffer(self):
        """버퍼 상태 저장"""
        # 이미지와 임베딩 저장
        buffer_file = os.path.join(self.storage_path, 'buffer_state.pkl')
        
        state = {
            'image_storage': self.image_storage,
            'user_sample_count': dict(self.user_sample_count),
            'total_added': self.total_added,
            'total_rejected': self.total_rejected
        }
        
        with open(buffer_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[Buffer] State saved: {len(self.image_storage)} samples")
    
    def _load_buffer(self):
        """저장된 버퍼 로드"""
        buffer_file = os.path.join(self.storage_path, 'buffer_state.pkl')
        
        if os.path.exists(buffer_file):
            try:
                with open(buffer_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.image_storage = state['image_storage']
                self.user_sample_count = defaultdict(int, state['user_sample_count'])
                self.total_added = state.get('total_added', 0)
                self.total_rejected = state.get('total_rejected', 0)
                
                # Faiss 인덱스 재구성
                if self.use_faiss:
                    self._rebuild_faiss_index()
                
                print(f"[Buffer] Loaded {len(self.image_storage)} samples")
            except Exception as e:
                print(f"[Buffer] Failed to load state: {e}")
    
    def get_statistics(self) -> dict:
        """버퍼 통계 반환"""
        unique_users = len(set(item['label'] for item in self.image_storage))
        
        return {
            'total_samples': len(self.image_storage),
            'unique_users': unique_users,
            'buffer_utilization': len(self.image_storage) / self.max_buffer_size,
            'total_added': self.total_added,
            'total_rejected': self.total_rejected,
            'acceptance_rate': self.total_added / (self.total_added + self.total_rejected) 
                              if (self.total_added + self.total_rejected) > 0 else 0
        }
    
    def __repr__(self):
        return f"EnhancedReplayBuffer(size={len(self.image_storage)}/{self.max_buffer_size})"