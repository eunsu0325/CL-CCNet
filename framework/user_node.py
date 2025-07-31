# framework/user_node.py
"""
COCONUT User Node System

사용자별 통계 정보를 저장하고 관리하는 노드 시스템
- Diagonal Mahalanobis 거리 기반 인증
- PQ 압축 준비 (추후 구현)
- 루프 클로저 지원

Design Philosophy:
- ON/OFF 가능한 모듈형 설계
- 메모리 효율적인 Diagonal 저장
- 학습 시에만 Full Covariance 사용
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

class UserNode:
    """
    사용자별 통계 정보를 저장하는 노드
    
    Features:
    - 평균 벡터 (μ) 저장
    - Diagonal 공분산 (Σ_diag) 저장  
    - PQ 압축 샘플 저장 (추후 구현)
    - 루프 클로저를 위한 원본 샘플 접근
    """
    
    def __init__(self, user_id: int, embeddings: torch.Tensor = None):
        """
        Args:
            user_id: 사용자 ID
            embeddings: [N, feature_dim] 형태의 임베딩 텐서
        """
        self.user_id = user_id
        self.mean = None  # μ
        self.cov_diag = None  # Σ_diag
        self.cov_full = None  # 학습 시에만 사용
        self.sample_count = 0
        
        # 메타데이터
        self.creation_time = datetime.now().isoformat()
        self.last_update = self.creation_time
        self.update_count = 0
        
        # PQ 압축용 (추후 구현)
        self.compressed_samples = []
        self.compression_enabled = False
        
        if embeddings is not None:
            self.update_statistics(embeddings)
    
    def update_statistics(self, embeddings: torch.Tensor):
        """평균과 분산 계산 및 업데이트"""
        if embeddings is None or embeddings.numel() == 0:
            return
            
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # 새로운 통계 계산
        new_mean = embeddings.mean(dim=0)
        new_cov_diag = embeddings.var(dim=0, unbiased=True) + 1e-6  # 안정성을 위한 epsilon
        
        if self.mean is None:
            # 첫 업데이트
            self.mean = new_mean
            self.cov_diag = new_cov_diag
            self.sample_count = embeddings.size(0)
        else:
            # 증분 업데이트 (running average)
            total_count = self.sample_count + embeddings.size(0)
            self.mean = (self.mean * self.sample_count + new_mean * embeddings.size(0)) / total_count
            
            # Welford's algorithm for variance
            # 더 안정적인 분산 업데이트
            self.cov_diag = (self.cov_diag * (self.sample_count - 1) + 
                           new_cov_diag * (embeddings.size(0) - 1)) / (total_count - 1)
            self.sample_count = total_count
        
        self.last_update = datetime.now().isoformat()
        self.update_count += 1
        
    def compute_full_covariance(self, embeddings: torch.Tensor):
        """학습용 Full Covariance 계산 (메모리 집약적, 학습 시에만 사용)"""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
            
        if embeddings.size(0) < 2:
            print(f"[UserNode] Warning: Need at least 2 samples for full covariance")
            return
            
        centered = embeddings - self.mean
        self.cov_full = (centered.T @ centered) / (embeddings.size(0) - 1)
        # 정규화 추가
        self.cov_full += torch.eye(self.cov_full.size(0), device=self.cov_full.device) * 1e-6
        
    def diagonal_mahalanobis_distance(self, query: torch.Tensor) -> float:
        """
        Diagonal Mahalanobis 거리 계산 (빠른 인증용)
        
        d² = (x - μ)ᵀ Σ⁻¹ (x - μ) where Σ is diagonal
        """
        if self.mean is None or self.cov_diag is None:
            return float('inf')
            
        diff = query - self.mean
        # Diagonal이므로 간단히 계산 가능
        dist_squared = torch.sum(diff**2 / self.cov_diag)
        return torch.sqrt(dist_squared).item()
    
    def full_mahalanobis_distance(self, query: torch.Tensor) -> float:
        """Full Mahalanobis 거리 계산 (정밀 검증용)"""
        if self.mean is None or self.cov_full is None:
            return self.diagonal_mahalanobis_distance(query)
            
        diff = query - self.mean
        try:
            cov_inv = torch.inverse(self.cov_full)
            dist_squared = diff @ cov_inv @ diff
            return torch.sqrt(dist_squared).item()
        except:
            # 역행렬 계산 실패 시 diagonal 사용
            return self.diagonal_mahalanobis_distance(query)
    
    def add_compressed_samples(self, embeddings: torch.Tensor):
        """PQ 압축 샘플 추가 (추후 구현)"""
        # TODO: Product Quantization 구현
        pass
    
    def get_decompressed_samples(self) -> Optional[torch.Tensor]:
        """압축된 샘플 복원 (추후 구현)"""
        # TODO: PQ 복원 구현
        return None
    
    def to_dict(self) -> Dict:
        """저장용 딕셔너리 변환"""
        data = {
            'user_id': self.user_id,
            'mean': self.mean.cpu().numpy().tolist() if self.mean is not None else None,
            'cov_diag': self.cov_diag.cpu().numpy().tolist() if self.cov_diag is not None else None,
            'sample_count': self.sample_count,
            'creation_time': self.creation_time,
            'last_update': self.last_update,
            'update_count': self.update_count,
            'compression_enabled': self.compression_enabled
        }
        
        # PQ 압축 데이터 추가 (있는 경우)
        if self.compressed_samples:
            data['compressed_samples'] = self.compressed_samples
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict, device='cpu'):
        """딕셔너리에서 복원"""
        node = cls(data['user_id'])
        
        if data.get('mean') is not None:
            node.mean = torch.tensor(data['mean'], device=device)
        if data.get('cov_diag') is not None:
            node.cov_diag = torch.tensor(data['cov_diag'], device=device)
            
        node.sample_count = data.get('sample_count', 0)
        node.creation_time = data.get('creation_time', '')
        node.last_update = data.get('last_update', '')
        node.update_count = data.get('update_count', 0)
        node.compression_enabled = data.get('compression_enabled', False)
        
        if 'compressed_samples' in data:
            node.compressed_samples = data['compressed_samples']
            
        return node
    
    def __repr__(self):
        return (f"UserNode(id={self.user_id}, samples={self.sample_count}, "
                f"updates={self.update_count})")


class UserNodeManager:
    """
    모든 사용자 노드를 관리하는 매니저
    
    Features:
    - 사용자 노드 CRUD 작업
    - 루프 클로저 충돌 감지
    - Faiss 인덱스 연동 (옵션)
    - 노드 영속성 관리
    """
    
    def __init__(self, config: Dict, device='cpu'):
        """
        Args:
            config: 설정 딕셔너리
            device: 연산 디바이스
        """
        self.config = config
        self.device = device
        
        # 사용자 노드 모드 on/off
        self.enabled = config.get('enable_user_nodes', True)
        if not self.enabled:
            print("[NodeManager] ⚠️ User node system is DISABLED")
            return
            
        # 경로 설정
        self.save_dir = Path(config.get('node_save_path', './results/user_nodes/'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 노드 저장소
        self.nodes: Dict[int, UserNode] = {}
        
        # 설정값
        self.collision_threshold = config.get('collision_threshold', 0.5)
        self.use_faiss = config.get('use_faiss_index', True)
        self.faiss_index = None
        
        # 로드
        self.load_nodes()
        
        print(f"[NodeManager] ✅ Initialized with {len(self.nodes)} nodes")
        print(f"[NodeManager] Mode: {'ENABLED' if self.enabled else 'DISABLED'}")
        print(f"[NodeManager] Collision threshold: {self.collision_threshold}")
    
    def is_enabled(self) -> bool:
        """사용자 노드 시스템 활성화 여부"""
        return self.enabled
    
    def add_user(self, user_id: int, embeddings: torch.Tensor) -> Optional[UserNode]:
        """새 사용자 추가"""
        if not self.enabled:
            return None
            
        # 기존 사용자 확인
        if user_id in self.nodes:
            print(f"[NodeManager] User {user_id} already exists, updating...")
            return self.update_user(user_id, embeddings)
            
        # 새 노드 생성
        node = UserNode(user_id, embeddings)
        self.nodes[user_id] = node
        
        # Faiss 인덱스 업데이트
        if self.use_faiss:
            self._update_faiss_index()
            
        self.save_nodes()
        print(f"[NodeManager] Added new user {user_id} with {embeddings.size(0)} samples")
        return node
    
    def update_user(self, user_id: int, embeddings: torch.Tensor) -> Optional[UserNode]:
        """기존 사용자 업데이트"""
        if not self.enabled:
            return None
            
        if user_id not in self.nodes:
            return self.add_user(user_id, embeddings)
            
        node = self.nodes[user_id]
        node.update_statistics(embeddings)
        
        # Faiss 인덱스 업데이트
        if self.use_faiss:
            self._update_faiss_index()
            
        self.save_nodes()
        print(f"[NodeManager] Updated user {user_id}")
        return node
    
    def check_collision(self, query_embedding: torch.Tensor, 
                       exclude_user: Optional[int] = None) -> Optional[Tuple[int, float]]:
        """
        임베딩 충돌 검사 (루프 클로저용)
        
        Returns:
            (user_id, distance) if collision detected, None otherwise
        """
        if not self.enabled or not self.nodes:
            return None
            
        min_distance = float('inf')
        collision_user = None
        
        for user_id, node in self.nodes.items():
            if user_id == exclude_user:
                continue
                
            distance = node.diagonal_mahalanobis_distance(query_embedding)
            
            if distance < min_distance:
                min_distance = distance
                collision_user = user_id
                
        if min_distance < self.collision_threshold:
            print(f"[NodeManager] ⚠️ Collision detected! User {collision_user} "
                  f"(distance: {min_distance:.4f} < {self.collision_threshold})")
            return (collision_user, min_distance)
            
        return None
    
    def find_nearest_users(self, query: torch.Tensor, k: int = 5) -> List[Tuple[int, float]]:
        """가장 가까운 k명의 사용자 찾기"""
        if not self.enabled or not self.nodes:
            return []
            
        distances = []
        
        for user_id, node in self.nodes.items():
            dist = node.diagonal_mahalanobis_distance(query)
            distances.append((user_id, dist))
            
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_node(self, user_id: int) -> Optional[UserNode]:
        """특정 사용자 노드 반환"""
        if not self.enabled:
            return None
        return self.nodes.get(user_id)
    
    def remove_user(self, user_id: int) -> bool:
        """사용자 제거"""
        if not self.enabled or user_id not in self.nodes:
            return False
            
        del self.nodes[user_id]
        
        if self.use_faiss:
            self._update_faiss_index()
            
        self.save_nodes()
        print(f"[NodeManager] Removed user {user_id}")
        return True
    
    def reconstruct_user_node(self, user_id: int, new_embeddings: torch.Tensor):
        """루프 클로저 후 노드 재구성"""
        if not self.enabled:
            return
            
        print(f"[NodeManager] Reconstructing node for user {user_id}")
        
        # 기존 노드 제거
        if user_id in self.nodes:
            del self.nodes[user_id]
            
        # 새로 생성
        self.add_user(user_id, new_embeddings)
        
    def _update_faiss_index(self):
        """Faiss 인덱스 업데이트 (추후 구현)"""
        # TODO: Faiss 인덱스 구현
        pass
    
    def save_nodes(self):
        """모든 노드 저장"""
        if not self.enabled:
            return
            
        save_data = {
            'nodes': {str(uid): node.to_dict() for uid, node in self.nodes.items()},
            'total_users': len(self.nodes),
            'last_save': datetime.now().isoformat(),
            'config': self.config
        }
        
        # JSON 저장
        json_path = self.save_dir / 'user_nodes.json'
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        # 바이너리 백업 (더 빠른 로딩용)
        pkl_path = self.save_dir / 'user_nodes.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"[NodeManager] Saved {len(self.nodes)} nodes")
    
    def load_nodes(self):
        """저장된 노드 로드"""
        if not self.enabled:
            return
            
        # 먼저 pkl 시도 (더 빠름)
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
        if not self.enabled:
            return {'enabled': False}
            
        stats = {
            'enabled': True,
            'total_users': len(self.nodes),
            'total_samples': sum(node.sample_count for node in self.nodes.values()),
            'avg_samples_per_user': sum(node.sample_count for node in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'total_updates': sum(node.update_count for node in self.nodes.values()),
            'collision_threshold': self.collision_threshold
        }
        
        return stats
    
    def __repr__(self):
        if not self.enabled:
            return "UserNodeManager(DISABLED)"
        return f"UserNodeManager(users={len(self.nodes)}, enabled=True)"


# 테스트용 헬퍼 함수
def test_user_node_system():
    """간단한 테스트"""
    print("\n=== Testing User Node System ===")
    
    # 테스트 설정
    config = {
        'enable_user_nodes': True,
        'node_save_path': './test_nodes/',
        'collision_threshold': 0.5,
        'use_faiss_index': False
    }
    
    # 매니저 생성
    manager = UserNodeManager(config)
    
    # 테스트 데이터
    embeddings1 = torch.randn(5, 128)  # 5개 샘플
    embeddings2 = torch.randn(3, 128)  # 3개 샘플
    
    # 사용자 추가
    node1 = manager.add_user(1, embeddings1)
    node2 = manager.add_user(2, embeddings2)
    
    # 충돌 테스트
    test_query = embeddings1[0] + torch.randn(128) * 0.1  # 약간의 노이즈
    collision = manager.check_collision(test_query)
    
    print(f"Collision test: {collision}")
    
    # 최근접 사용자
    nearest = manager.find_nearest_users(test_query, k=2)
    print(f"Nearest users: {nearest}")
    
    # 통계
    print(f"Statistics: {manager.get_statistics()}")
    
    print("=== Test Complete ===\n")


if __name__ == "__main__":
    test_user_node_system()