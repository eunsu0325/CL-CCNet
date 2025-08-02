# framework/user_node.py - 정리된 버전
"""
User Node System with Faiss for Fast Search

사용자별 정보를 저장하고 관리하는 노드 시스템
- 사용자 ID
- 등록 시 사용한 원본 이미지
- 평균 임베딩 벡터
- Faiss를 이용한 빠른 검색
- 코사인 유사도 기반 인증
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import base64
import io
from PIL import Image
import cv2

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[UserNode] ⚠️ Faiss not found - using brute force search")

class UserNode:
    """
    사용자 노드
    
    저장 정보:
    - 사용자 ID
    - 등록 시 사용한 원본 이미지
    - 평균 임베딩 벡터
    """
    
    def __init__(self, user_id: int, registration_image: np.ndarray = None, embeddings: torch.Tensor = None):
        """
        Args:
            user_id: 사용자 ID
            registration_image: 등록용 원본 이미지 (numpy array)
            embeddings: [N, feature_dim] 형태의 임베딩 텐서
        """
        self.user_id = user_id
        self.registration_image = registration_image  # 원본 이미지 저장
        self.mean_embedding = None  # 평균 임베딩
        self.sample_count = 0
        
        # 메타데이터
        self.creation_time = datetime.now().isoformat()
        self.last_update = self.creation_time
        
        if embeddings is not None:
            self.update_embedding(embeddings)
    
    def update_embedding(self, embeddings: torch.Tensor):
        """평균 임베딩 업데이트"""
        if embeddings is None or embeddings.numel() == 0:
            return
            
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # 새로운 평균 계산
        new_mean = embeddings.mean(dim=0)
        
        if self.mean_embedding is None:
            # 첫 업데이트
            self.mean_embedding = new_mean
            self.sample_count = embeddings.size(0)
        else:
            # 증분 업데이트 (running average)
            total_count = self.sample_count + embeddings.size(0)
            self.mean_embedding = (self.mean_embedding * self.sample_count + new_mean * embeddings.size(0)) / total_count
            self.sample_count = total_count
        
        self.last_update = datetime.now().isoformat()
    
    def compute_similarity(self, query: torch.Tensor) -> float:
        """
        코사인 유사도 계산
        
        Args:
            query: 쿼리 임베딩
        Returns:
            similarity: 코사인 유사도 (0~1)
        """
        if self.mean_embedding is None:
            return 0.0
            
        # 코사인 유사도
        similarity = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0) if len(query.shape) == 1 else query,
            self.mean_embedding.unsqueeze(0) if len(self.mean_embedding.shape) == 1 else self.mean_embedding,
            dim=1
        )
        return similarity.item()
    
    def image_to_base64(self) -> Optional[str]:
        """이미지를 base64 문자열로 변환 (안전한 버전)"""
        if self.registration_image is None:
            return None
        
        try:
            # 이미지 데이터 정규화 및 타입 변환
            image_array = self.registration_image
            
            # 텐서인 경우 numpy로 변환
            if hasattr(image_array, 'cpu'):
                image_array = image_array.cpu().numpy()
            
            # 형태 확인 및 수정
            if len(image_array.shape) == 3:
                # (C, H, W) -> (H, W, C) 변환
                if image_array.shape[0] in [1, 3]:  # 채널이 첫 번째 차원
                    image_array = image_array.transpose(1, 2, 0)
            elif len(image_array.shape) == 4:
                # (1, C, H, W) -> (H, W, C) 변환
                image_array = image_array.squeeze(0).transpose(1, 2, 0)
            
            # 값 범위 정규화 (0-1 -> 0-255)
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # 그레이스케일 처리
            if len(image_array.shape) == 3 and image_array.shape[2] == 1:
                image_array = image_array.squeeze(2)  # (H, W, 1) -> (H, W)
            
            # 그레이스케일인 경우 L 모드로, 컬러인 경우 RGB 모드로
            if len(image_array.shape) == 2:
                pil_image = Image.fromarray(image_array, mode='L')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # BGR to RGB 변환 (OpenCV 이미지인 경우)
                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image, mode='RGB')
            else:
                # 예상치 못한 형태인 경우 기본 처리
                print(f"[UserNode] Warning: Unexpected image shape {image_array.shape}, creating dummy image")
                pil_image = Image.new('L', (64, 64), color=128)  # 64x64 회색 이미지
            
            # Base64 인코딩
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"[UserNode] Error converting image to base64: {e}")
            print(f"[UserNode] Image shape: {getattr(self.registration_image, 'shape', 'unknown')}")
            print(f"[UserNode] Image type: {type(self.registration_image)}")
            return None
    
    def base64_to_image(self, base64_str: str):
        """base64 문자열을 이미지로 변환"""
        if not base64_str:
            return
            
        # Base64 디코딩
        img_data = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Numpy array로 변환
        rgb_array = np.array(pil_image)
        
        # RGB to BGR 변환 (OpenCV 형식)
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            self.registration_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        else:
            self.registration_image = rgb_array
    
    def to_dict(self) -> Dict:
        """저장용 딕셔너리 변환"""
        data = {
            'user_id': self.user_id,
            'mean_embedding': self.mean_embedding.cpu().numpy().tolist() if self.mean_embedding is not None else None,
            'sample_count': self.sample_count,
            'creation_time': self.creation_time,
            'last_update': self.last_update,
            'registration_image': self.image_to_base64()  # 이미지를 base64로 저장
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict, device='cpu'):
        """딕셔너리에서 복원"""
        node = cls(data['user_id'])
        
        if data.get('mean_embedding') is not None:
            node.mean_embedding = torch.tensor(data['mean_embedding'], device=device)
            
        node.sample_count = data.get('sample_count', 0)
        node.creation_time = data.get('creation_time', '')
        node.last_update = data.get('last_update', '')
        
        # 이미지 복원
        if data.get('registration_image'):
            node.base64_to_image(data['registration_image'])
            
        return node
    
    def __repr__(self):
        return (f"UserNode(id={self.user_id}, samples={self.sample_count}, "
                f"has_image={self.registration_image is not None})")


class UserNodeManager:
    """
    사용자 노드 매니저 with Faiss
    
    Features:
    - 사용자 노드 CRUD 작업
    - Faiss를 이용한 빠른 Top-K 검색
    - 코사인 유사도 기반 인증
    - 간단한 충돌 감지
    """
    
    def __init__(self, config: Dict, device='cpu'):
        """
        Args:
            config: 설정 딕셔너리
            device: 연산 디바이스
        """
        self.config = config
        self.device = device
        
        # 경로 설정
        self.save_dir = Path(config.get('node_save_path', './results/user_nodes/'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 노드 저장소
        self.nodes: Dict[int, UserNode] = {}
        
        # 설정값
        self.similarity_threshold = config.get('similarity_threshold', 0.7)  # 코사인 유사도 임계값
        self.use_faiss = config.get('use_faiss_index', True) and FAISS_AVAILABLE
        self.feature_dim = config.get('feature_dimension', 128)
        
        # Faiss 인덱스
        self.faiss_index = None
        self.user_id_map = []  # Faiss 인덱스 위치 -> user_id 매핑
        
        # 로드
        self.load_nodes()
        
        # Faiss 인덱스 초기화
        if self.use_faiss and len(self.nodes) > 0:
            self._rebuild_faiss_index()
        
        print(f"[NodeManager] ✅ System initialized")
        print(f"[NodeManager] Loaded {len(self.nodes)} users")
        print(f"[NodeManager] Similarity threshold: {self.similarity_threshold}")
        print(f"[NodeManager] Faiss index: {'ENABLED' if self.use_faiss else 'DISABLED'}")
    
    def _initialize_faiss_index(self):
        """Faiss 인덱스 초기화"""
        if not self.use_faiss:
            return
            
        # 코사인 유사도를 위한 정규화된 내적 인덱스
        self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
        print(f"[NodeManager] Faiss index initialized (dim={self.feature_dim})")
    
    def _rebuild_faiss_index(self):
        """전체 Faiss 인덱스 재구성"""
        if not self.use_faiss:
            return
            
        # 인덱스 초기화
        self._initialize_faiss_index()
        self.user_id_map = []
        
        # 모든 노드의 평균 임베딩 추가
        embeddings = []
        
        for user_id, node in self.nodes.items():
            if node.mean_embedding is not None:
                embedding = node.mean_embedding.cpu().numpy().astype('float32')
                # L2 정규화 (코사인 유사도를 위해)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
                self.user_id_map.append(user_id)
        
        if embeddings:
            embeddings = np.array(embeddings)
            self.faiss_index.add(embeddings)
            print(f"[NodeManager] Faiss index rebuilt with {len(embeddings)} users")
    
    def _add_to_faiss_index(self, user_id: int, embedding: torch.Tensor):
        """단일 임베딩을 Faiss 인덱스에 추가"""
        if not self.use_faiss or self.faiss_index is None:
            return
            
        # numpy로 변환 및 정규화
        embedding_np = embedding.cpu().numpy().astype('float32')
        embedding_np = embedding_np / np.linalg.norm(embedding_np)
        
        # 인덱스에 추가
        self.faiss_index.add(embedding_np.reshape(1, -1))
        self.user_id_map.append(user_id)
    
    def add_user(self, user_id: int, embeddings: torch.Tensor, 
                 registration_image: np.ndarray = None) -> Optional[UserNode]:
        """새 사용자 추가"""
        # 기존 사용자 확인
        if user_id in self.nodes:
            print(f"[NodeManager] User {user_id} already exists, updating...")
            return self.update_user(user_id, embeddings, registration_image)
            
        # 새 노드 생성
        node = UserNode(user_id, registration_image, embeddings)
        self.nodes[user_id] = node
        
        # Faiss 인덱스에 추가
        if self.use_faiss:
            if self.faiss_index is None:
                self._initialize_faiss_index()
            self._add_to_faiss_index(user_id, node.mean_embedding)
        
        self.save_nodes()
        print(f"[NodeManager] Added new user {user_id}")
        return node
    
    def update_user(self, user_id: int, embeddings: torch.Tensor, 
                   registration_image: np.ndarray = None) -> Optional[UserNode]:
        """기존 사용자 업데이트"""
        if user_id not in self.nodes:
            return self.add_user(user_id, embeddings, registration_image)
            
        node = self.nodes[user_id]
        node.update_embedding(embeddings)
        
        # 이미지 업데이트 (제공된 경우)
        if registration_image is not None:
            node.registration_image = registration_image
        
        # Faiss 인덱스 재구성 (평균이 변경되었으므로)
        if self.use_faiss:
            self._rebuild_faiss_index()
        
        self.save_nodes()
        print(f"[NodeManager] Updated user {user_id}")
        return node
    
    def find_top_k_faiss(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
        """Faiss를 사용한 빠른 Top-K 검색"""
        if not self.use_faiss or self.faiss_index is None or self.faiss_index.ntotal == 0:
            return self.find_nearest_users_bruteforce(query_embedding, k)
        
        # Query 준비
        query_np = query_embedding.cpu().numpy().astype('float32')
        if len(query_np.shape) == 1:
            query_np = query_np.reshape(1, -1)
        
        # L2 정규화
        query_np = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
        
        # 검색
        k_search = min(k, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_np, k_search)
        
        # 결과 변환
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx < len(self.user_id_map):  # 유효한 인덱스
                user_id = self.user_id_map[idx]
                results.append((user_id, float(sim)))
        
        return results
    
    def find_nearest_users_bruteforce(self, query: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
        """브루트포스 방식의 최근접 사용자 검색 (Faiss 없을 때)"""
        if not self.nodes:
            return []
            
        similarities = []
        
        for user_id, node in self.nodes.items():
            sim = node.compute_similarity(query)
            similarities.append((user_id, sim))
            
        # 유사도 내림차순 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def find_nearest_users(self, query: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
        """가장 가까운 k명의 사용자 찾기 (Faiss 우선 사용)"""
        if self.use_faiss and self.faiss_index is not None:
            return self.find_top_k_faiss(query, k)
        else:
            return self.find_nearest_users_bruteforce(query, k)
    
    def verify_user(self, query_embedding: torch.Tensor, top_k: int = 5) -> Dict:
        """
        사용자 인증 (Faiss 기반 빠른 검색)
        
        Returns:
            {
                'is_match': bool,
                'matched_user': int or None,
                'similarity': float,
                'confidence': float,
                'top_k_results': list
            }
        """
        # Top-K 검색
        top_results = self.find_nearest_users(query_embedding, k=top_k)
        
        if not top_results:
            return {
                'is_match': False,
                'matched_user': None,
                'similarity': 0.0,
                'confidence': 0.0,
                'top_k_results': [],
                'threshold': self.similarity_threshold
            }
        
        # 최고 매칭
        best_user_id, best_similarity = top_results[0]
        
        # 인증 판정
        is_match = best_similarity >= self.similarity_threshold
        
        # 신뢰도 계산
        if is_match:
            confidence = (best_similarity - self.similarity_threshold) / (1.0 - self.similarity_threshold)
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = 0.0
        
        return {
            'is_match': is_match,
            'matched_user': best_user_id if is_match else None,
            'similarity': best_similarity,
            'confidence': confidence,
            'top_k_results': top_results,
            'threshold': self.similarity_threshold,
            'search_method': 'faiss' if self.use_faiss else 'bruteforce'
        }
    
    def check_collision(self, query_embedding: torch.Tensor, 
                       exclude_user: Optional[int] = None) -> Optional[Tuple[int, float]]:
        """
        임베딩 충돌 검사 (너무 유사한 사용자가 있는지)
        
        Returns:
            (user_id, similarity) if collision detected, None otherwise
        """
        collision_threshold = 0.95  # 매우 높은 유사도
        
        # Top-K 검색으로 가장 유사한 사용자들 찾기
        top_results = self.find_nearest_users(query_embedding, k=10)
        
        for user_id, similarity in top_results:
            if user_id == exclude_user:
                continue
                
            if similarity >= collision_threshold:
                print(f"[NodeManager] ⚠️ Collision detected! User {user_id} "
                      f"(similarity: {similarity:.4f})")
                return (user_id, similarity)
                
        return None
    
    def get_node(self, user_id: int) -> Optional[UserNode]:
        """특정 사용자 노드 반환"""
        return self.nodes.get(user_id)
    
    def remove_user(self, user_id: int) -> bool:
        """사용자 제거"""
        if user_id not in self.nodes:
            return False
            
        del self.nodes[user_id]
        
        # Faiss 인덱스 재구성
        if self.use_faiss:
            self._rebuild_faiss_index()
        
        self.save_nodes()
        print(f"[NodeManager] Removed user {user_id}")
        return True
    
    def save_nodes(self):
        """모든 노드 저장"""
        save_data = {
            'nodes': {str(uid): node.to_dict() for uid, node in self.nodes.items()},
            'total_users': len(self.nodes),
            'last_save': datetime.now().isoformat(),
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'use_faiss': self.use_faiss
            }
        }
        
        # JSON 저장
        json_path = self.save_dir / 'user_nodes.json'
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        # 바이너리 백업
        pkl_path = self.save_dir / 'user_nodes.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"[NodeManager] Saved {len(self.nodes)} nodes")
    
    def load_nodes(self):
        """저장된 노드 로드"""
        pkl_path = self.save_dir / 'user_nodes.pkl'
        json_path = self.save_dir / 'user_nodes.json'
        
        loaded = False
        
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                loaded = True
            except:
                print("[NodeManager] Failed to load pkl, trying json...")
                
        if not loaded and json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                loaded = True
            except Exception as e:
                print(f"[NodeManager] Failed to load nodes: {e}")
                return
                
        if loaded:
            for uid_str, node_data in data['nodes'].items():
                uid = int(uid_str)
                self.nodes[uid] = UserNode.from_dict(node_data, self.device)
                
            print(f"[NodeManager] Loaded {len(self.nodes)} nodes")
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        stats = {
            'total_users': len(self.nodes),
            'total_samples': sum(node.sample_count for node in self.nodes.values()),
            'avg_samples_per_user': sum(node.sample_count for node in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'similarity_threshold': self.similarity_threshold,
            'users_with_images': sum(1 for node in self.nodes.values() if node.registration_image is not None),
            'faiss_enabled': self.use_faiss,
            'search_method': 'faiss' if self.use_faiss else 'bruteforce'
        }
        
        if self.use_faiss and self.faiss_index is not None:
            stats['faiss_index_size'] = self.faiss_index.ntotal
            
        return stats
    
    def __repr__(self):
        return f"UserNodeManager(users={len(self.nodes)}, faiss={self.use_faiss})"