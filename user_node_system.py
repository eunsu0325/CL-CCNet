# user_node_system.py - 사용자 노드 시스템 + Mahalanobis 거리 인증

"""
🥥 CoCoNut 사용자 노드 시스템 구현

핵심 기능:
1. 사용자 노드 (μ, Σ_diag) 관리
2. Mahalanobis 거리 기반 인증
3. 온라인 노드 업데이트
4. 루프 클로저 감지 (추후 구현)

메모리 효율성:
- 사용자당 O(256) 메모리 (128D × 2)
- Diagonal covariance만 저장
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import json
from pathlib import Path
from collections import defaultdict
import faiss

class UserNode:
    """개별 사용자 노드 클래스"""
    
    def __init__(self, user_id: int, feature_dim: int = 128):
        self.user_id = user_id
        self.feature_dim = feature_dim
        
        # 통계 정보 (핵심 데이터)
        self.centroid = None  # μ: 평균 벡터 [128]
        self.diag_covariance = None  # Σ_diag: 대각 공분산 [128]
        
        # 온라인 업데이트용 임시 저장소
        self.embeddings = []  # 임베딩 벡터들 임시 저장
        self.is_finalized = False  # 노드 완성 여부
        
        print(f"[Node] 사용자 {user_id} 노드 생성 (feature_dim: {feature_dim})")
    
    def add_embedding(self, embedding: torch.Tensor):
        """새로운 임베딩 추가"""
        if self.is_finalized:
            print(f"[Node] Warning: 완성된 노드 {self.user_id}에 임베딩 추가 시도")
            return
        
        # L2 정규화된 임베딩인지 확인
        embedding = F.normalize(embedding.flatten(), dim=0)
        self.embeddings.append(embedding.cpu().numpy())
        
        print(f"[Node] 사용자 {self.user_id}: {len(self.embeddings)}개 임베딩 수집")
    
    def finalize_node(self, min_samples: int = 3):
        """노드 통계 계산 및 완성"""
        if len(self.embeddings) < min_samples:
            print(f"[Node] Warning: 사용자 {self.user_id} 샘플 부족 ({len(self.embeddings)} < {min_samples})")
            return False
        
        embeddings_array = np.array(self.embeddings)  # [N, 128]
        
        # 평균 벡터 계산
        self.centroid = np.mean(embeddings_array, axis=0)  # [128]
        
        # 대각 공분산 계산 (각 차원의 분산)
        self.diag_covariance = np.var(embeddings_array, axis=0)  # [128]
        
        # 수치적 안정성을 위한 최소값 설정
        min_variance = 1e-6
        self.diag_covariance = np.maximum(self.diag_covariance, min_variance)
        
        # 메모리 절약: 임베딩 리스트 삭제
        self.embeddings = []
        self.is_finalized = True
        
        print(f"[Node] ✅ 사용자 {self.user_id} 노드 완성:")
        print(f"   Centroid norm: {np.linalg.norm(self.centroid):.4f}")
        print(f"   Variance range: [{self.diag_covariance.min():.6f}, {self.diag_covariance.max():.6f}]")
        
        return True
    
    def mahalanobis_distance(self, query_embedding: torch.Tensor) -> float:
        """개선된 Diagonal Mahalanobis 거리 계산"""
        if not self.is_finalized:
            raise ValueError(f"사용자 {self.user_id} 노드가 아직 완성되지 않음")
        
        # 쿼리 임베딩 정규화
        query = F.normalize(query_embedding.flatten(), dim=0).cpu().numpy()  # [128]
        
        # 차이 벡터 계산
        diff = query - self.centroid  # [128]
        
        # 🔥 개선된 Mahalanobis 거리 계산
        # 1. 표준화된 거리 (각 차원을 표준편차로 나눔)
        std_devs = np.sqrt(self.diag_covariance)  # 표준편차
        standardized_diff = diff / std_devs  # 표준화
        
        # 2. 차원으로 정규화된 거리 (고차원 보정)
        raw_distance = np.linalg.norm(standardized_diff)
        normalized_distance = raw_distance / np.sqrt(self.feature_dim)  # 128차원 보정
        
        return normalized_distance
    
    def get_memory_usage(self) -> int:
        """메모리 사용량 반환 (bytes)"""
        if not self.is_finalized:
            return len(self.embeddings) * self.feature_dim * 4  # float32
        else:
            return self.feature_dim * 2 * 4  # centroid + diag_covariance
    
    def to_dict(self) -> Dict:
        """직렬화용 딕셔너리 변환"""
        return {
            'user_id': self.user_id,
            'feature_dim': self.feature_dim,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'diag_covariance': self.diag_covariance.tolist() if self.diag_covariance is not None else None,
            'is_finalized': self.is_finalized,
            'num_embeddings': len(self.embeddings)
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """딕셔너리에서 노드 복원"""
        node = cls(data['user_id'], data['feature_dim'])
        
        if data['centroid'] is not None:
            node.centroid = np.array(data['centroid'])
        if data['diag_covariance'] is not None:
            node.diag_covariance = np.array(data['diag_covariance'])
        
        node.is_finalized = data['is_finalized']
        return node

class CoconutNodeSystem:
    """CoCoNut 사용자 노드 관리 시스템"""
    
    def __init__(self, feature_dim: int = 128, save_path: str = "./user_nodes.pkl"):
        self.feature_dim = feature_dim
        self.save_path = Path(save_path)
        
        # 사용자 노드들
        self.nodes: Dict[int, UserNode] = {}  # {user_id: UserNode}
        self.temp_nodes: Dict[int, UserNode] = {}  # 임시 노드 (완성되지 않은)
        
        # Faiss 인덱스 (빠른 검색용)
        self.faiss_index = None
        self.node_ids = []  # Faiss 인덱스와 user_id 매핑
        
        # 루프 클로저 관련
        self.loop_closure_threshold = 2.0  # Mahalanobis 거리 임계값
        
        print(f"[NodeSystem] 시스템 초기화 (feature_dim: {feature_dim})")
        
        # 기존 노드 로드
        self._load_nodes()
    
    def register_embedding(self, user_id: int, embedding: torch.Tensor, 
                          finalize_threshold: int = 5) -> Optional[str]:
        """
        임베딩 등록 및 필요시 노드 완성
        
        Args:
            user_id: 사용자 ID
            embedding: 임베딩 벡터 [128]
            finalize_threshold: 노드 완성 임계값
            
        Returns:
            상태 메시지
        """
        # 1. 기존 완성된 노드가 있는지 확인
        if user_id in self.nodes:
            print(f"[NodeSystem] 사용자 {user_id}는 이미 등록됨")
            return "already_registered"
        
        # 2. 임시 노드에 추가
        if user_id not in self.temp_nodes:
            self.temp_nodes[user_id] = UserNode(user_id, self.feature_dim)
        
        self.temp_nodes[user_id].add_embedding(embedding)
        
        # 3. 완성 조건 확인
        if len(self.temp_nodes[user_id].embeddings) >= finalize_threshold:
            return self._finalize_user_node(user_id)
        
        return f"collecting_{len(self.temp_nodes[user_id].embeddings)}"
    
    def _finalize_user_node(self, user_id: int) -> str:
        """사용자 노드 완성 및 등록"""
        temp_node = self.temp_nodes[user_id]
        
        # 노드 통계 계산
        if temp_node.finalize_node():
            # 루프 클로저 검사 (추후 구현)
            loop_closure_detected = self._check_loop_closure(temp_node)
            
            if loop_closure_detected:
                return "loop_closure_detected"
            
            # 정상 등록
            self.nodes[user_id] = temp_node
            del self.temp_nodes[user_id]
            
            # Faiss 인덱스 업데이트
            self._update_faiss_index()
            
            # 저장
            self._save_nodes()
            
            print(f"[NodeSystem] ✅ 사용자 {user_id} 노드 등록 완료")
            return "registered"
        else:
            print(f"[NodeSystem] ❌ 사용자 {user_id} 노드 완성 실패")
            return "finalization_failed"
    
    def _check_loop_closure(self, new_node: UserNode) -> bool:
        """루프 클로저 감지 (임시 구현)"""
        if len(self.nodes) == 0:
            return False
        
        # 새 노드와 기존 노드들 간 거리 계산
        new_centroid = torch.tensor(new_node.centroid)
        
        for existing_id, existing_node in self.nodes.items():
            distance = existing_node.mahalanobis_distance(new_centroid)
            
            if distance < self.loop_closure_threshold:
                print(f"[LoopClosure] 🔄 감지: 사용자 {new_node.user_id} vs {existing_id} (거리: {distance:.4f})")
                # TODO: 루프 클로저 처리 로직
                return True
        
        return False
    
    def authenticate(self, query_embedding: torch.Tensor, 
                    auth_threshold: float = 0.8, top_k: int = 5) -> Dict:
        """
        개선된 Mahalanobis 거리 기반 인증
        
        Args:
            query_embedding: 쿼리 임베딩 [128]
            auth_threshold: 정규화된 인증 임계값 (0.8 추천)
            top_k: 상위 k개 후보 검사
            
        Returns:
            인증 결과 딕셔너리
        """
        if len(self.nodes) == 0:
            return {
                'authenticated': False,
                'user_id': None,
                'distance': float('inf'),
                'reason': 'no_registered_users'
            }
        
        # 1. Faiss로 빠른 후보 검색 (코사인 유사도 기반)
        top_candidates = self._faiss_search(query_embedding, top_k)
        
        # 2. 각 후보에 대해 Mahalanobis 거리 계산
        best_user_id = None
        best_distance = float('inf')
        
        for candidate_id in top_candidates:
            if candidate_id in self.nodes:
                distance = self.nodes[candidate_id].mahalanobis_distance(query_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_user_id = candidate_id
        
        # 3. 임계값 기반 인증 결정
        authenticated = best_distance < auth_threshold
        
        result = {
            'authenticated': authenticated,
            'user_id': best_user_id if authenticated else None,
            'distance': best_distance,
            'threshold': auth_threshold,
            'candidates_checked': len(top_candidates)
        }
        
        if authenticated:
            print(f"[Auth] ✅ 인증 성공: 사용자 {best_user_id} (거리: {best_distance:.4f})")
        else:
            print(f"[Auth] ❌ 인증 실패: 최소 거리 {best_distance:.4f} > 임계값 {auth_threshold}")
        
        return result
    
    def _update_faiss_index(self):
        """Faiss 인덱스 업데이트"""
        if len(self.nodes) == 0:
            return
        
        # 모든 노드의 centroid 수집
        centroids = []
        node_ids = []
        
        for user_id, node in self.nodes.items():
            centroids.append(node.centroid)
            node_ids.append(user_id)
        
        centroids_array = np.array(centroids).astype('float32')  # [N, 128]
        
        # Faiss 인덱스 생성 (코사인 유사도)
        self.faiss_index = faiss.IndexFlatIP(self.feature_dim)
        
        # L2 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(centroids_array)
        self.faiss_index.add(centroids_array)
        
        self.node_ids = node_ids
        print(f"[Faiss] 인덱스 업데이트: {len(node_ids)} 노드")
    
    def _faiss_search(self, query_embedding: torch.Tensor, k: int) -> List[int]:
        """Faiss로 빠른 후보 검색"""
        if self.faiss_index is None:
            return list(self.nodes.keys())
        
        # 쿼리 정규화
        query = F.normalize(query_embedding.flatten(), dim=0).cpu().numpy().astype('float32')
        query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        
        # 검색
        k = min(k, len(self.node_ids))
        similarities, indices = self.faiss_index.search(query, k)
        
        # user_id로 변환
        candidate_ids = [self.node_ids[idx] for idx in indices[0] if idx < len(self.node_ids)]
        
        return candidate_ids
    
    def get_system_stats(self) -> Dict:
        """시스템 통계 반환"""
        total_memory = sum(node.get_memory_usage() for node in self.nodes.values())
        temp_memory = sum(node.get_memory_usage() for node in self.temp_nodes.values())
        
        return {
            'registered_users': len(self.nodes),
            'temp_users': len(self.temp_nodes),
            'total_memory_bytes': total_memory,
            'temp_memory_bytes': temp_memory,
            'memory_per_user_bytes': total_memory // len(self.nodes) if self.nodes else 0,
            'faiss_index_size': len(self.node_ids)
        }
    
    def _save_nodes(self):
        """노드 저장"""
        save_data = {
            'nodes': {uid: node.to_dict() for uid, node in self.nodes.items()},
            'temp_nodes': {uid: node.to_dict() for uid, node in self.temp_nodes.items()},
            'feature_dim': self.feature_dim
        }
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[NodeSystem] 💾 노드 저장: {len(self.nodes)} 완성, {len(self.temp_nodes)} 임시")
    
    def _load_nodes(self):
        """노드 로드"""
        if not self.save_path.exists():
            print(f"[NodeSystem] 저장된 노드 없음")
            return
        
        try:
            with open(self.save_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # 완성된 노드 복원
            for uid, node_data in save_data.get('nodes', {}).items():
                self.nodes[int(uid)] = UserNode.from_dict(node_data)
            
            # 임시 노드 복원
            for uid, node_data in save_data.get('temp_nodes', {}).items():
                self.temp_nodes[int(uid)] = UserNode.from_dict(node_data)
            
            # Faiss 인덱스 재구성
            if self.nodes:
                self._update_faiss_index()
            
            print(f"[NodeSystem] 📂 노드 로드: {len(self.nodes)} 완성, {len(self.temp_nodes)} 임시")
            
        except Exception as e:
            print(f"[NodeSystem] ❌ 노드 로드 실패: {e}")

# 테스트 함수
def test_node_system():
    """노드 시스템 테스트"""
    print("🧪 사용자 노드 시스템 테스트")
    print("="*50)
    
    # 시스템 초기화
    node_system = CoconutNodeSystem(feature_dim=128)
    
    # 가짜 임베딩 생성
    def generate_user_embeddings(user_id: int, num_samples: int = 6):
        """특정 사용자의 유사한 임베딩들 생성"""
        base_vector = torch.randn(128) * 0.1
        embeddings = []
        
        for i in range(num_samples):
            noise = torch.randn(128) * 0.05  # 작은 노이즈
            embedding = F.normalize(base_vector + noise, dim=0)
            embeddings.append(embedding)
        
        return embeddings
    
    # 사용자 등록 시뮬레이션
    for user_id in [1, 2, 3]:
        print(f"\n👤 사용자 {user_id} 등록 중...")
        embeddings = generate_user_embeddings(user_id)
        
        for i, embedding in enumerate(embeddings):
            result = node_system.register_embedding(user_id, embedding)
            print(f"  임베딩 {i+1}: {result}")
    
    # 시스템 통계
    stats = node_system.get_system_stats()
    print(f"\n📊 시스템 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 인증 테스트
    print(f"\n🔐 인증 테스트")
    
    # 등록된 사용자 (사용자 1과 유사한 임베딩)
    test_embedding_1 = generate_user_embeddings(1, 1)[0]
    result = node_system.authenticate(test_embedding_1, auth_threshold=0.8)
    print(f"사용자 1 유사 임베딩: {result}")
    
    # 미등록 사용자 (완전히 다른 임베딩)
    test_embedding_unknown = F.normalize(torch.randn(128) * 2.0, dim=0)
    result = node_system.authenticate(test_embedding_unknown, auth_threshold=0.8)
    print(f"미등록 사용자: {result}")
    
    # 🔥 추가: 거리 분포 분석
    print(f"\n📊 거리 분포 분석")
    
    # 같은 사용자들의 거리
    genuine_distances = []
    for user_id in [1, 2, 3]:
        for _ in range(10):
            test_emb = generate_user_embeddings(user_id, 1)[0]
            distance = node_system.nodes[user_id].mahalanobis_distance(test_emb)
            genuine_distances.append(distance)
    
    # 다른 사용자들의 거리
    imposter_distances = []
    for _ in range(30):
        unknown_emb = F.normalize(torch.randn(128) * 2.0, dim=0)
        for user_id in [1, 2, 3]:
            distance = node_system.nodes[user_id].mahalanobis_distance(unknown_emb)
            imposter_distances.append(distance)
    
    print(f"Genuine 거리: 평균 {np.mean(genuine_distances):.4f}, 범위 [{np.min(genuine_distances):.4f}, {np.max(genuine_distances):.4f}]")
    print(f"Imposter 거리: 평균 {np.mean(imposter_distances):.4f}, 범위 [{np.min(imposter_distances):.4f}, {np.max(imposter_distances):.4f}]")
    
    # 권장 임계값 계산
    recommended_threshold = (np.max(genuine_distances) + np.min(imposter_distances)) / 2
    print(f"권장 임계값: {recommended_threshold:.4f}")
    
    # 권장 임계값으로 재테스트
    print(f"\n🎯 권장 임계값({recommended_threshold:.2f})으로 재테스트")
    result1 = node_system.authenticate(test_embedding_1, auth_threshold=recommended_threshold)
    result2 = node_system.authenticate(test_embedding_unknown, auth_threshold=recommended_threshold)
    print(f"사용자 1 유사: {result1['authenticated']} (거리: {result1['distance']:.4f})")
    print(f"미등록 사용자: {result2['authenticated']} (거리: {result2['distance']:.4f})")

if __name__ == "__main__":
    test_node_system()