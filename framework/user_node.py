# framework/user_node.py - Loop Closure를 위한 정규화 이미지 저장 버전

import torch
import numpy as np
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import faiss
import base64
import io
from PIL import Image
import torch.nn.functional as F


class UserNode:
    """
    사용자별 노드 - 임베딩과 이미지 저장
    
    Features:
    - Mean embedding for fast matching
    - Registration image (raw) for visualization
    - Normalized tensors for Loop Closure
    - Update history tracking
    """
    
    def __init__(self, user_id: int, feature_dimension: int):
        self.user_id = user_id
        self.feature_dimension = feature_dimension
        
        # 임베딩 관련
        self.mean_embedding = None
        self.embeddings = []
        
        # 이미지 저장
        self.registration_image = None  # 원본 이미지 (시각화용)
        self.normalized_tensors = []    # 정규화된 텐서 (Loop Closure용)
        self.max_stored_tensors = 10    # 메모리 관리를 위한 최대 저장 수
        
        # 메타데이터
        self.sample_count = 0
        self.last_update = None
        self.creation_time = datetime.now()
        
    def update(self, embeddings: torch.Tensor, 
               registration_image: Optional[np.ndarray] = None,
               normalized_tensors: Optional[List[torch.Tensor]] = None):
        """
        노드 업데이트
        
        Args:
            embeddings: [N, feature_dim] 특징 벡터
            registration_image: 원본 이미지 (시각화용)
            normalized_tensors: 정규화된 텐서들 (Loop Closure용)
        """
        # 임베딩 업데이트
        if isinstance(embeddings, torch.Tensor):
            embeddings_list = embeddings.cpu().numpy()
        else:
            embeddings_list = embeddings
            
        self.embeddings.extend(embeddings_list)
        self.sample_count += len(embeddings_list)
        
        # Mean embedding 재계산
        self.mean_embedding = torch.tensor(
            np.mean(self.embeddings, axis=0),
            dtype=torch.float32
        )
        
        # 시각화용 원본 이미지
        if registration_image is not None:
            self.registration_image = registration_image
        
        # Loop Closure용 정규화된 텐서 저장
        if normalized_tensors is not None:
            self._store_normalized_tensors(normalized_tensors)
        
        self.last_update = datetime.now()
    
    def _store_normalized_tensors(self, new_tensors: List[torch.Tensor]):
        """정규화된 텐서 저장 (다양성 기반 선택)"""
        # 기존 텐서와 새 텐서 합치기
        all_tensors = self.normalized_tensors + new_tensors
        
        if len(all_tensors) <= self.max_stored_tensors:
            # 저장 공간이 충분하면 모두 저장
            self.normalized_tensors = [t.cpu() for t in all_tensors]
        else:
            # 다양성 기반으로 선택
            selected_indices = self._select_diverse_samples(all_tensors, self.max_stored_tensors)
            self.normalized_tensors = [all_tensors[i].cpu() for i in selected_indices]
            
        print(f"[UserNode {self.user_id}] Stored {len(self.normalized_tensors)} normalized tensors")
    
    def _select_diverse_samples(self, tensors: List[torch.Tensor], n_select: int) -> List[int]:
        """다양성 기반 샘플 선택"""
        if len(tensors) <= n_select:
            return list(range(len(tensors)))
        
        # 모든 텐서를 특징 벡터로 변환 (이미 특징이면 그대로 사용)
        features = []
        for t in tensors:
            if len(t.shape) > 1:  # 이미지 텐서인 경우
                # 간단한 평균 풀링으로 특징 추출
                feat = t.view(t.size(0), -1).mean(dim=1) if len(t.shape) == 3 else t.mean()
                features.append(feat)
            else:
                features.append(t)
        
        # Greedy 선택: 가장 멀리 떨어진 샘플들 선택
        selected = [0]  # 첫 번째 샘플 선택
        
        while len(selected) < n_select:
            max_min_dist = -1
            best_idx = -1
            
            for i in range(len(features)):
                if i in selected:
                    continue
                    
                # 선택된 샘플들과의 최소 거리 계산
                min_dist = float('inf')
                for j in selected:
                    dist = torch.norm(features[i] - features[j]).item()
                    min_dist = min(min_dist, dist)
                
                # 최소 거리가 가장 큰 샘플 선택
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(best_idx)
        
        return selected
    
    def get_loop_closure_data(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Loop Closure를 위한 데이터 반환
        
        Returns:
            mean_embedding: 평균 임베딩
            normalized_tensors: 정규화된 텐서들
        """
        return self.mean_embedding, self.normalized_tensors
    
    def image_to_base64(self) -> Optional[str]:
        """원본 이미지를 base64로 변환 (시각화용)"""
        if self.registration_image is None:
            return None
        
        try:
            image_array = self.registration_image
            
            # numpy 배열 확인
            if not isinstance(image_array, np.ndarray):
                if hasattr(image_array, 'cpu'):
                    image_array = image_array.cpu().numpy()
            
            # uint8 형태 확인
            if image_array.dtype == np.uint8:
                # 그대로 사용
                if len(image_array.shape) == 3 and image_array.shape[2] == 1:
                    image_array = image_array.squeeze(2)
            else:
                # float 형태면 변환
                print(f"[UserNode] Converting from {image_array.dtype} to uint8")
                min_val = image_array.min()
                max_val = image_array.max()
                if max_val - min_val > 0:
                    image_array = (image_array - min_val) / (max_val - min_val)
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.full_like(image_array, 128, dtype=np.uint8)
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(image_array, mode='L')
            
            # Base64 인코딩
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"[UserNode] Error converting image to base64: {e}")
            return None
    
    def to_dict(self) -> dict:
        """직렬화를 위한 딕셔너리 변환"""
        return {
            'user_id': self.user_id,
            'mean_embedding': self.mean_embedding.numpy().tolist() if self.mean_embedding is not None else None,
            'embeddings': [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in self.embeddings],
            'sample_count': self.sample_count,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'creation_time': self.creation_time.isoformat(),
            'feature_dimension': self.feature_dimension,
            'num_stored_tensors': len(self.normalized_tensors),
            'registration_image_shape': self.registration_image.shape if self.registration_image is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: dict, feature_dimension: int) -> 'UserNode':
        """딕셔너리에서 UserNode 복원"""
        node = cls(data['user_id'], feature_dimension)
        
        if data.get('mean_embedding'):
            node.mean_embedding = torch.tensor(data['mean_embedding'], dtype=torch.float32)
        
        if data.get('embeddings'):
            node.embeddings = [np.array(emb) for emb in data['embeddings']]
        
        node.sample_count = data.get('sample_count', 0)
        
        if data.get('last_update'):
            node.last_update = datetime.fromisoformat(data['last_update'])
        
        if data.get('creation_time'):
            node.creation_time = datetime.fromisoformat(data['creation_time'])
        
        return node


class UserNodeManager:
    """
    사용자 노드 관리자 - Loop Closure 지원
    
    Features:
    - User node creation and updates
    - Fast similarity search with Faiss
    - Loop closure data management
    - Persistence support
    """
    
    def __init__(self, config: Dict, device='cuda'):
        self.config = config
        self.device = device
        
        # 설정
        self.feature_dimension = config.get('feature_dimension', 128)
        self.distance_threshold = config.get('distance_threshold', 0.5)
        self.storage_path = Path(config.get('storage_path', './user_nodes'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 사용자 노드 저장소
        self.nodes: Dict[int, UserNode] = {}
        
        # Faiss 인덱스
        self.index = faiss.IndexFlatL2(self.feature_dimension)
        self.user_id_map = []  # 인덱스 -> user_id 매핑
        
        # 통계
        self.total_verifications = 0
        self.successful_verifications = 0
        
        # 기존 노드 로드
        self.load_nodes()
        
        print(f"[NodeManager] ✅ Initialized")
        print(f"  Feature dimension: {self.feature_dimension}")
        print(f"  Distance threshold: {self.distance_threshold}")
        print(f"  Loaded nodes: {len(self.nodes)}")
    
    def add_user(self, user_id: int, embeddings: torch.Tensor, 
                 registration_image: Optional[np.ndarray] = None,
                 normalized_tensors: Optional[List[torch.Tensor]] = None):
        """
        새 사용자 추가 또는 업데이트
        
        Args:
            user_id: 사용자 ID
            embeddings: 특징 벡터들
            registration_image: 원본 이미지 (시각화용)
            normalized_tensors: 정규화된 텐서들 (Loop Closure용)
        """
        if user_id in self.nodes:
            # 기존 사용자 업데이트
            node = self.nodes[user_id]
            node.update(embeddings, registration_image, normalized_tensors)
            print(f"[NodeManager] Updated user {user_id}")
        else:
            # 새 사용자 생성
            node = UserNode(user_id, self.feature_dimension)
            node.update(embeddings, registration_image, normalized_tensors)
            self.nodes[user_id] = node
            print(f"[NodeManager] Added new user {user_id}")
        
        # Faiss 인덱스 업데이트
        self._update_faiss_index()
    
    def get_loop_closure_candidates(self, similarity_threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        Loop Closure 후보 찾기
        
        Returns:
            List of (user_id1, user_id2, similarity) tuples
        """
        candidates = []
        user_ids = list(self.nodes.keys())
        
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                user1, user2 = user_ids[i], user_ids[j]
                node1, node2 = self.nodes[user1], self.nodes[user2]
                
                if node1.mean_embedding is not None and node2.mean_embedding is not None:
                    # 코사인 유사도 계산
                    similarity = F.cosine_similarity(
                        node1.mean_embedding.unsqueeze(0),
                        node2.mean_embedding.unsqueeze(0)
                    ).item()
                    
                    if similarity > similarity_threshold:
                        candidates.append((user1, user2, similarity))
        
        # 유사도 순으로 정렬
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates
    
    def get_loop_closure_data(self, user_ids: List[int]) -> Dict[int, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Loop Closure를 위한 데이터 반환
        
        Args:
            user_ids: 사용자 ID 리스트
            
        Returns:
            Dict mapping user_id to (mean_embedding, normalized_tensors)
        """
        data = {}
        for user_id in user_ids:
            if user_id in self.nodes:
                node = self.nodes[user_id]
                data[user_id] = node.get_loop_closure_data()
        return data
    
    def _update_faiss_index(self):
        """Faiss 인덱스 재구성"""
        # 인덱스 초기화
        self.index = faiss.IndexFlatL2(self.feature_dimension)
        self.user_id_map = []
        
        # 모든 평균 임베딩 추가
        embeddings_list = []
        for user_id, node in self.nodes.items():
            if node.mean_embedding is not None:
                embeddings_list.append(node.mean_embedding.numpy())
                self.user_id_map.append(user_id)
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            self.index.add(embeddings_array)
    
    def find_nearest_users(self, query_embedding: torch.Tensor, k: int = 10) -> List[Tuple[int, float]]:
        """가장 가까운 사용자 찾기"""
        if self.index.ntotal == 0:
            return []
        
        # 쿼리 준비
        if isinstance(query_embedding, torch.Tensor):
            query = query_embedding.cpu().numpy().reshape(1, -1).astype('float32')
        else:
            query = query_embedding.reshape(1, -1).astype('float32')
        
        # 검색
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query, k)
        
        # 결과 변환
        results = []
        for i in range(k):
            if indices[0][i] >= 0:
                user_id = self.user_id_map[indices[0][i]]
                distance = float(distances[0][i])
                results.append((user_id, distance))
        
        return results
    
    def verify_user(self, probe_embedding: torch.Tensor, top_k: int = 10) -> Dict:
        """사용자 인증"""
        self.total_verifications += 1
        
        # 가장 가까운 사용자 찾기
        candidates = self.find_nearest_users(probe_embedding, k=top_k)
        
        if not candidates:
            return {
                'is_match': False,
                'matched_user': None,
                'distance': float('inf'),
                'confidence': 0.0,
                'top_k_results': []
            }
        
        # 최상위 매치
        best_user_id, best_distance = candidates[0]
        
        # L2 거리를 코사인 거리로 변환 (근사)
        cosine_distance = best_distance / 2.0  # 정규화된 벡터 가정
        
        # 임계값 확인
        is_match = cosine_distance <= self.distance_threshold
        
        if is_match:
            self.successful_verifications += 1
        
        # 신뢰도 계산
        confidence = max(0.0, 1.0 - (cosine_distance / self.distance_threshold))
        
        return {
            'is_match': is_match,
            'matched_user': best_user_id if is_match else None,
            'distance': cosine_distance,
            'confidence': confidence,
            'top_k_results': candidates[:5],
            'threshold': self.distance_threshold
        }
    
    def get_node(self, user_id: int) -> Optional[UserNode]:
        """특정 사용자 노드 반환"""
        return self.nodes.get(user_id)
    
    def save_nodes(self):
        """모든 노드 저장"""
        # 메타데이터 저장
        metadata = {
            'total_users': len(self.nodes),
            'feature_dimension': self.feature_dimension,
            'distance_threshold': self.distance_threshold,
            'total_verifications': self.total_verifications,
            'successful_verifications': self.successful_verifications,
            'save_time': datetime.now().isoformat()
        }
        
        metadata_path = self.storage_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 각 노드 저장
        for user_id, node in self.nodes.items():
            node_path = self.storage_path / f'node_{user_id}.pkl'
            node_dict = node.to_dict()
            
            # 정규화된 텐서는 별도 저장
            tensors_path = self.storage_path / f'tensors_{user_id}.pt'
            if node.normalized_tensors:
                torch.save(node.normalized_tensors, tensors_path)
            
            # 원본 이미지는 별도 저장
            if node.registration_image is not None:
                img_path = self.storage_path / f'img_{user_id}.npy'
                np.save(img_path, node.registration_image)
            
            with open(node_path, 'wb') as f:
                pickle.dump(node_dict, f)
        
        print(f"[NodeManager] 💾 Saved {len(self.nodes)} nodes")
    
    def load_nodes(self):
        """저장된 노드 로드"""
        metadata_path = self.storage_path / 'metadata.json'
        
        if not metadata_path.exists():
            print("[NodeManager] No saved nodes found")
            return
        
        # 메타데이터 로드
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.total_verifications = metadata.get('total_verifications', 0)
        self.successful_verifications = metadata.get('successful_verifications', 0)
        
        # 각 노드 로드
        node_files = list(self.storage_path.glob('node_*.pkl'))
        
        for node_file in node_files:
            user_id = int(node_file.stem.split('_')[1])
            
            with open(node_file, 'rb') as f:
                node_dict = pickle.load(f)
            
            node = UserNode.from_dict(node_dict, self.feature_dimension)
            
            # 정규화된 텐서 로드
            tensors_path = self.storage_path / f'tensors_{user_id}.pt'
            if tensors_path.exists():
                node.normalized_tensors = torch.load(tensors_path)
            
            # 원본 이미지 로드
            img_path = self.storage_path / f'img_{user_id}.npy'
            if img_path.exists():
                node.registration_image = np.load(img_path)
            
            self.nodes[user_id] = node
        
        # Faiss 인덱스 재구성
        self._update_faiss_index()
        
        print(f"[NodeManager] 📂 Loaded {len(self.nodes)} nodes")
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        total_embeddings = sum(node.sample_count for node in self.nodes.values())
        total_tensors = sum(len(node.normalized_tensors) for node in self.nodes.values())
        
        return {
            'total_users': len(self.nodes),
            'total_embeddings': total_embeddings,
            'total_stored_tensors': total_tensors,
            'avg_embeddings_per_user': total_embeddings / len(self.nodes) if self.nodes else 0,
            'total_verifications': self.total_verifications,
            'successful_verifications': self.successful_verifications,
            'success_rate': self.successful_verifications / self.total_verifications if self.total_verifications > 0 else 0,
            'feature_dimension': self.feature_dimension,
            'distance_threshold': self.distance_threshold
        }