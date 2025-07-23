# framework/replay_buffer.py - 로깅 단순화된 버전
"""
CoCoNut Intelligent Replay Buffer 

CORE CONTRIBUTION:
- Diversity-based sample selection using Faiss similarity search
- Original images stored for exact replay (no degradation)
- Metadata management for rich context information
- Efficient culling strategy to maintain buffer size
- Proper batch size support with replacement sampling

DESIGN PHILOSOPHY:
- Focus on buffer intelligence, not learning complexity
- Faiss-accelerated for practical deployment
- Memory-efficient with automatic duplicate removal
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import torch

class CoconutReplayBuffer:
    def __init__(self, config, storage_dir: Path, feature_dimension: int = 2048):
        """
        CoCoNut의 지능형 리플레이 버퍼 초기화
        
        CORE INNOVATION:
        - Faiss index for efficient high-dimensional similarity search
        - Diversity threshold prevents redundant samples
        - Original image storage ensures perfect replay quality
        - Support for replacement sampling to guarantee batch sizes
        """
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension

        # Faiss: 고차원 벡터의 유사도 검색을 위한 인덱스
        self.image_storage = []  # 원본 이미지들을 저장할 리스트
        self.faiss_index = None  # 다양성 측정용
        
        # 임베딩을 별도로 저장하여 reconstruction 문제 해결
        self.stored_embeddings = []

        # 버퍼에 저장된 데이터의 메타정보 관리
        self.metadata = {}

        # 특징 추출을 위한 모델 참조
        self.feature_extractor = None

        # 시스템 재시작을 위한 상태 파일 경로
        self.state_file = self.storage_dir / 'buffer_state.pkl'
        self._load_state()

        print(f"[Buffer] 🥥 CoCoNut Replay Buffer initialized")
        print(f"[Buffer] Max size: {self.buffer_size}, Similarity threshold: {self.similarity_threshold}")
        print(f"[Buffer] Current buffer size: {self.faiss_index.ntotal if self.faiss_index else 0}")

    def set_feature_extractor(self, model):
        """특징 추출을 위한 모델 설정"""
        self.feature_extractor = model

    def _initialize_faiss(self):
        """Faiss 인덱스를 초기화합니다."""
        index = faiss.IndexFlatIP(self.feature_dimension)
        self.faiss_index = faiss.IndexIDMap(index)
        print(f"[Buffer] Faiss index initialized with dimension {self.feature_dimension}")

    def _extract_feature_for_diversity(self, image):
        """다양성 측정을 위한 특징 벡터 추출"""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set. Call set_feature_extractor() first.")
        
        # 배치 차원 추가 (필요한 경우)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # CCNet의 getFeatureCode 메서드 사용
        with torch.no_grad():
            features = self.feature_extractor.getFeatureCode(image)
        
        return features

    def add(self, image: torch.Tensor, user_id: int):
        """
        새로운 경험(이미지)을 버퍼에 추가할지 결정하고 추가합니다.
        
        DIVERSITY-BASED ADDITION (핵심 기여):
        1. Extract feature embedding from new image
        2. Calculate similarity with existing samples using Faiss
        3. Add only if sufficiently diverse (below threshold)
        """
        # 1. 다양성 측정을 위한 임베딩 추출
        with torch.no_grad():
            embedding = self._extract_feature_for_diversity(image)
            embedding_np = embedding.cpu().numpy().astype('float32')
            faiss.normalize_L2(embedding_np)

        if self.faiss_index is None:
            self._initialize_faiss()

        # 2. 다양성 확인
        if self.faiss_index.ntotal == 0:
            max_similarity = 0.0
        else:
            distances, _ = self.faiss_index.search(embedding_np, k=1)
            max_similarity = distances[0][0]

        # 3. 새로운 경험이면 원본 이미지 저장
        if max_similarity < self.similarity_threshold:
            if len(self.image_storage) >= self.buffer_size:
                self._cull()  # 오래된 이미지 제거
            
            unique_id = len(self.image_storage)
            self.image_storage.append({
                'image': image.cpu().clone(),  # 원본 이미지 저장
                'user_id': user_id,
                'id': unique_id
            })
            
            # 임베딩도 별도로 저장
            self.stored_embeddings.append(embedding_np.copy())
            
            # 다양성 측정용 faiss 인덱스도 업데이트
            self.faiss_index.add_with_ids(embedding_np, np.array([unique_id]))
            self.metadata[unique_id] = {'user_id': user_id}
            
            print(f"[Buffer] ✅ Added diverse sample (ID: {unique_id}, User: {user_id}). "
                  f"Buffer: {len(self.image_storage)}/{self.buffer_size}")
        else:
            print(f"[Buffer] ⚠️ Similar sample skipped (similarity: {max_similarity:.4f} >= {self.similarity_threshold})")
            
    def _cull(self):
        """
        버퍼 내에서 가장 중복되는 데이터를 제거합니다.
        
        INTELLIGENT CULLING (핵심 기여):
        - Faiss를 사용하여 중복도가 가장 높은 샘플 식별
        - 저장된 임베딩을 직접 사용하여 안정성 확보
        """
        if self.faiss_index.ntotal < 2:
            return

        print(f"[Buffer] 🔄 Buffer full ({self.faiss_index.ntotal}). Finding most redundant sample...")
        
        # 저장된 임베딩들을 NumPy 배열로 변환
        if len(self.stored_embeddings) == 0:
            print("[Buffer] ⚠️ No stored embeddings for culling")
            return
        
        all_vectors = np.vstack(self.stored_embeddings)
        
        k = min(self.faiss_index.ntotal, 50) 
        similarities, _ = self.faiss_index.search(all_vectors, k=k)
        
        # 각 벡터의 유사도 총합 계산 (자기 자신과의 유사도(1.0)는 제외)
        diversity_scores = similarities.sum(axis=1) - 1.0
        
        cull_idx_in_storage = np.argmax(diversity_scores)
        cull_unique_id = self.image_storage[cull_idx_in_storage]['id']

        # Faiss 인덱스에서 제거
        try:
            self.faiss_index.remove_ids(np.array([cull_unique_id]))
        except Exception as e:
            print(f"[Buffer] ⚠️ Faiss removal failed, rebuilding index...")
            self._rebuild_faiss_index_after_removal(cull_idx_in_storage)
        
        # 메타데이터에서 제거
        if cull_unique_id in self.metadata:
            del self.metadata[cull_unique_id]
        
        # 이미지 저장소에서도 제거
        del self.image_storage[cull_idx_in_storage]
        del self.stored_embeddings[cull_idx_in_storage]
        
        print(f"[Buffer] 🗑️ Removed redundant sample (ID: {cull_unique_id})")

    def _rebuild_faiss_index_after_removal(self, removed_idx):
        """Faiss 인덱스 재구축 (제거 실패 시 백업 방법)"""
        print("[Buffer] 🔧 Rebuilding Faiss index...")
        
        # 새로운 인덱스 생성
        self._initialize_faiss()
        
        # 제거될 항목을 제외하고 다시 추가
        for i, (item, embedding) in enumerate(zip(self.image_storage, self.stored_embeddings)):
            if i != removed_idx:  # 제거될 항목 제외
                self.faiss_index.add_with_ids(
                    embedding.reshape(1, -1), 
                    np.array([item['id']])
                )

    def sample_with_replacement(self, batch_size: int):
        """
        복원 추출로 충분한 샘플 확보 (핵심 기능)
        
        KEY CONTRIBUTION: 버퍼 크기에 관계없이 요청된 개수만큼 반환
        - 항상 설정된 batch_size 달성
        - 연속학습이 충분한 대조 쌍으로 학습 가능
        """
        if len(self.image_storage) == 0:
            print("[Buffer] ⚠️ Empty buffer - returning empty lists")
            return [], []
        
        if len(self.image_storage) == 1:
            # 버퍼에 1개만 있으면 그것만 반복
            item = self.image_storage[0]
            images = [item['image'].clone() for _ in range(batch_size)]
            labels = [item['user_id'] for _ in range(batch_size)]
            
            print(f"[Buffer] 🔄 Single sample replication: {batch_size} copies of User {item['user_id']}")
            return images, labels
        
        # 복원 추출로 필요한 개수만큼 샘플링
        sampled_indices = np.random.choice(
            len(self.image_storage), 
            size=batch_size, 
            replace=True  # 핵심: 복원 추출 허용
        )
        
        images = []
        labels = []
        sampled_users = []
        
        for idx in sampled_indices:
            item = self.image_storage[idx]
            images.append(item['image'].clone())  # 복사본 생성 (중요!)
            labels.append(item['user_id'])
            sampled_users.append(item['user_id'])
        
        # 샘플링 품질 분석
        unique_users = len(set(sampled_users))
        user_counts = {}
        for user_id in sampled_users:
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        print(f"[Buffer] 🎯 Sampled {len(images)} replay samples:")
        print(f"   Unique users: {unique_users}, Distribution: {dict(sorted(user_counts.items()))}")
        
        return images, labels

    def sample(self, batch_size: int):
        """기본 샘플링 - 복원 추출 사용"""
        return self.sample_with_replacement(batch_size)

    def get_diversity_stats(self):
        """버퍼 다양성 통계 반환"""
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
        """버퍼의 현재 상태를 파일로 저장합니다."""
        if self.faiss_index is None:
            print("[Buffer] No data to save.")
            return
            
        print(f"[Buffer] 💾 Saving buffer state...")
        
        data_to_save = {
            'faiss_index_data': faiss.serialize_index(self.faiss_index),
            'metadata': self.metadata,
            'image_storage': self.image_storage,
            'stored_embeddings': self.stored_embeddings
        }
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        # 다양성 통계도 저장
        diversity_stats = self.get_diversity_stats()
        stats_file = self.storage_dir / 'buffer_diversity_stats.json'
        
        import json
        with open(stats_file, 'w') as f:
            json.dump(diversity_stats, f, indent=2)
        
        print(f"[Buffer] ✅ Saved {len(self.image_storage)} samples")
        print(f"[Buffer] Diversity: {diversity_stats['unique_users']} users, "
              f"{diversity_stats['diversity_score']:.2f} score")

    def _load_state(self):
        """파일에서 버퍼 상태를 복원합니다."""
        if not self.state_file.exists():
            print(f"[Buffer] 📂 No previous state found - starting fresh")
            return

        print(f"[Buffer] 🔄 Loading previous buffer state...")
        try:
            with open(self.state_file, 'rb') as f:
                saved_data = pickle.load(f)
                
                # Faiss 인덱스 복원
                self.faiss_index = faiss.deserialize_index(saved_data['faiss_index_data'])
                self.metadata = saved_data['metadata']
                
                # 이미지 저장소도 복원
                if 'image_storage' in saved_data:
                    self.image_storage = saved_data['image_storage']
                
                # 저장된 임베딩도 복원
                if 'stored_embeddings' in saved_data:
                    self.stored_embeddings = saved_data['stored_embeddings']
                else:
                    self.stored_embeddings = []
                    print("[Buffer] ⚠️ No embeddings in saved state")
                
                # 복원된 다양성 통계 출력
                diversity_stats = self.get_diversity_stats()
                print(f"[Buffer] ✅ Restored {len(self.image_storage)} samples")
                print(f"[Buffer] Diversity: {diversity_stats['unique_users']} users")
                
        except Exception as e:
            print(f"[Buffer] ❌ Failed to load state: {e}. Starting fresh.")
            self.faiss_index = None
            self.metadata = {}
            self.image_storage = []
            self.stored_embeddings = []