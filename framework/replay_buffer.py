# framework/replay_buffer.py - 스마트 버퍼 저장 로직 추가 (수정 버전)

"""
CoCoNut Intelligent Replay Buffer with Smart Storage Logic

🔥 NEW FEATURES:
- 스마트 저장 로직 (smart_add)
- 긍정쌍 확보를 위한 강제 저장
- 학습 여부 기반 저장 결정
- 사용자별 샘플 수 관리
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
        지능형 리플레이 버퍼 초기화 (스마트 저장 로직 포함)
        """
        self.config = config
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 설정
        self.buffer_size = self.config.max_buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.feature_dimension = feature_dimension
        
        # 🔥 NEW: 스마트 저장을 위한 설정
        self.min_samples_per_user = getattr(config, 'min_samples_per_user', 2)
        
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

        print(f"[Buffer] 🥥 CoCoNut Smart Replay Buffer initialized")
        print(f"[Buffer] Strategy: {self.sampling_strategy}")
        print(f"[Buffer] Max buffer size: {self.buffer_size}")
        print(f"[Buffer] Min samples per user: {self.min_samples_per_user}")
        print(f"[Buffer] Current size: {len(self.image_storage)}")

    def smart_add(self, image: torch.Tensor, user_id: int, learned: bool = False):
        """
        🔥 NEW: 스마트 버퍼 저장 - 학습 여부를 고려한 저장 결정
        
        Args:
            image: 저장할 이미지
            user_id: 사용자 ID
            learned: 이 샘플이 학습에 사용되었는지 여부
            
        Returns:
            str: 저장 결정 이유
        """
        
        existing_user_samples = [item for item in self.image_storage 
                               if item['user_id'] == user_id]
        
        # Case 1: 완전 새로운 사용자 - 무조건 저장
        if len(existing_user_samples) == 0:
            reason = "new_user_first_sample"
            self._force_store(image, user_id, reason)
            return reason
        
        # Case 2: 기존 사용자, 샘플 부족 - 긍정쌍 위해 저장
        if len(existing_user_samples) < self.min_samples_per_user:
            reason = "ensuring_positive_pairs"
            self._force_store(image, user_id, reason)
            return reason
        
        # Case 3: 충분한 샘플 보유 - 다양성 기반 판단
        if learned:
            # 이미 학습에 활용했으므로 엄격한 다양성 기준 적용
            try:
                max_similarity = self._compute_max_similarity_for_user(image, user_id)
                if max_similarity < self.similarity_threshold:
                    reason = "diversity_based_storage_after_learning"
                    self._force_store(image, user_id, reason)
                    return reason
                else:
                    reason = f"too_similar_after_learning_{max_similarity:.3f}"
                    return reason
            except Exception as e:
                print(f"[Buffer] ⚠️ Similarity computation failed: {e}")
                reason = "diversity_check_failed_store_anyway"
                self._force_store(image, user_id, reason)
                return reason
        else:
            # 학습 안했으면 저장해서 나중에 활용 기회 제공
            reason = "not_learned_store_for_future"
            self._force_store(image, user_id, reason)
            return reason

    def _force_store(self, image: torch.Tensor, user_id: int, reason: str):
        """🔥 NEW: 강제 저장 (버퍼 공간 확보 후 저장)"""
        # 버퍼가 가득 찬 경우 공간 확보
        if len(self.image_storage) >= self.buffer_size:
            self._smart_cull_for_positive_pairs()

        unique_id = len(self.image_storage)
        self.image_storage.append({
            'image': image.cpu().clone(),
            'user_id': user_id,
            'id': unique_id,
            'storage_reason': reason
        })
        
        # 임베딩도 저장 (기존 방식)
        if hasattr(self, 'stored_embeddings'):
            try:
                with torch.no_grad():
                    embedding = self._extract_feature_for_diversity(image.to(self.device))
                    embedding_np = embedding.cpu().numpy().astype('float32')
                    self.stored_embeddings.append(embedding_np.copy())
                    
                    # Faiss 인덱스 업데이트
                    if FAISS_AVAILABLE and self.faiss_index is not None:
                        faiss.normalize_L2(embedding_np)
                        self.faiss_index.add_with_ids(embedding_np.reshape(1, -1), np.array([unique_id]))
                        
            except Exception as e:
                print(f"[Buffer] ⚠️ Embedding extraction failed: {e}")
        
        # 메타데이터 업데이트
        if hasattr(self, 'metadata'):
            self.metadata[unique_id] = {'user_id': user_id, 'reason': reason}

        print(f"[Buffer] ✅ Force stored sample (ID: {unique_id}, User: {user_id}, Reason: {reason})")

    def _compute_max_similarity_for_user(self, image: torch.Tensor, user_id: int):
        """🔥 NEW: 특정 사용자의 샘플들과의 최대 유사도 계산"""
        user_samples = [item for item in self.image_storage if item['user_id'] == user_id]
        
        if len(user_samples) == 0:
            return 0.0
        
        # 새로운 이미지의 특징 추출
        with torch.no_grad():
            new_embedding = self._extract_feature_for_diversity(image.to(self.device))
            new_embedding_np = new_embedding.cpu().numpy().flatten()
        
        max_sim = 0.0
        for sample in user_samples:
            try:
                # 기존 샘플의 특징 추출
                stored_image = sample['image'].to(self.device)
                with torch.no_grad():
                    stored_embedding = self._extract_feature_for_diversity(stored_image)
                    stored_embedding_np = stored_embedding.cpu().numpy().flatten()
                
                # 코사인 유사도 계산
                similarity = np.dot(new_embedding_np, stored_embedding_np) / (
                    np.linalg.norm(new_embedding_np) * np.linalg.norm(stored_embedding_np) + 1e-8
                )
                max_sim = max(max_sim, similarity)
            except Exception as e:
                print(f"[Buffer] ⚠️ Similarity calculation failed for sample: {e}")
                continue
        
        return max_sim

    def _smart_cull_for_positive_pairs(self):
        """🔥 NEW: 긍정쌍을 보존하면서 지능적 큐레이션"""
        print(f"[Buffer] 🔄 Smart culling to preserve positive pairs...")
        
        # 1. 사용자별 샘플 수 분석
        user_counts = {}
        for item in self.image_storage:
            user_id = item['user_id']
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        # 2. 샘플이 많은 사용자부터 제거 대상 선정
        over_sampled_users = [uid for uid, count in user_counts.items() 
                             if count > self.min_samples_per_user]
        
        if over_sampled_users:
            # 가장 많은 샘플을 가진 사용자의 가장 유사한 샘플 제거
            victim_user = max(over_sampled_users, key=lambda uid: user_counts[uid])
            victim_sample_idx = self._find_most_similar_sample_for_user(victim_user)
            
            if victim_sample_idx is not None:
                victim_id = self.image_storage[victim_sample_idx]['id']
                
                # 제거 실행
                del self.image_storage[victim_sample_idx]
                if victim_sample_idx < len(self.stored_embeddings):
                    del self.stored_embeddings[victim_sample_idx]
                if victim_id in self.metadata:
                    del self.metadata[victim_id]
                
                print(f"[Buffer] 🗑️ Removed redundant sample (User: {victim_user}, ID: {victim_id})")
            else:
                # 기존 방식으로 폴백
                self._cull()
        else:
            # 모든 사용자가 최소 샘플 이하면 기존 방식 사용
            self._cull()

    def _find_most_similar_sample_for_user(self, user_id: int):
        """특정 사용자의 가장 유사한 샘플 찾기"""
        user_samples = [(i, item) for i, item in enumerate(self.image_storage) 
                       if item['user_id'] == user_id]
        
        if len(user_samples) < 2:
            return None
        
        max_similarity = -1
        most_similar_idx = None
        
        for i, (idx1, sample1) in enumerate(user_samples):
            for j, (idx2, sample2) in enumerate(user_samples[i+1:], i+1):
                try:
                    # 두 샘플 간 유사도 계산
                    img1 = sample1['image'].to(self.device)
                    img2 = sample2['image'].to(self.device)
                    
                    with torch.no_grad():
                        emb1 = self._extract_feature_for_diversity(img1).cpu().numpy().flatten()
                        emb2 = self._extract_feature_for_diversity(img2).cpu().numpy().flatten()
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        # 더 최근 샘플을 제거 (인덱스가 큰 것)
                        most_similar_idx = max(idx1, idx2)
                        
                except Exception as e:
                    print(f"[Buffer] ⚠️ Similarity calculation failed: {e}")
                    continue
        
        return most_similar_idx

    def update_batch_composition_config(self, target_positive_ratio: float, hard_mining_ratio: float):
        """배치 구성 비율 업데이트 (CoconutSystem에서 호출)"""
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
        """새로운 경험을 버퍼에 추가 (기존 방식 - 호환성 유지)"""
        
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
        """가장 중복되는 데이터 제거 (기존 방식)"""
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
        제어된 배치 구성으로 정확한 비율 샘플링 (기존 방식 유지)
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
        use