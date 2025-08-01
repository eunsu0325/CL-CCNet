import torch
import numpy as np
from typing import List, Optional, Tuple
import torch.nn.functional as F


class UserNode:
    """
    사용자별 생체 정보를 저장하는 노드
    - 원본 이미지 저장 (최대 3개)
    - 평균 임베딩 계산 및 저장
    - PQ 압축 제거, 마할라노비스 제거
    """
    
    def __init__(self, user_id: int, max_images: int = 3):
        self.user_id = user_id
        self.max_images = max_images
        
        # 등록 시 사용한 원본 이미지 저장
        self.registered_images = []
        
        # 임베딩 저장 (평균 계산용)
        self.embeddings = []
        
        # 평균 임베딩 (Faiss 검색용)
        self.mean_embedding = None
        
        # Faiss 인덱스 내 위치
        self.faiss_idx = -1
        
        # 통계 정보
        self.update_count = 0
        self.last_update_time = None
    
    def add_sample(self, image: torch.Tensor, embedding: torch.Tensor) -> bool:
        """
        새로운 샘플 추가
        Args:
            image: 원본 이미지 텐서
            embedding: 특징 벡터 (L2 정규화됨)
        Returns:
            bool: 추가 성공 여부
        """
        # 임베딩 정규화 확인
        embedding = F.normalize(embedding, p=2, dim=0)
        
        # 이미지와 임베딩 저장
        if len(self.registered_images) < self.max_images:
            self.registered_images.append(image.cpu().clone())
            self.embeddings.append(embedding.cpu().clone())
        else:
            # 가장 오래된 것 교체 (FIFO)
            self.registered_images.pop(0)
            self.embeddings.pop(0)
            self.registered_images.append(image.cpu().clone())
            self.embeddings.append(embedding.cpu().clone())
        
        # 평균 임베딩 업데이트
        self._update_mean_embedding()
        
        self.update_count += 1
        return True
    
    def _update_mean_embedding(self):
        """평균 임베딩 계산 및 업데이트"""
        if len(self.embeddings) > 0:
            # 평균 계산
            mean = torch.stack(self.embeddings).mean(dim=0)
            # L2 정규화
            self.mean_embedding = F.normalize(mean, p=2, dim=0).numpy()
    
    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """평균 임베딩 반환 (Faiss 검색용)"""
        return self.mean_embedding
    
    def get_registered_images(self) -> List[torch.Tensor]:
        """등록된 원본 이미지 반환"""
        return self.registered_images
    
    def get_embeddings(self) -> List[torch.Tensor]:
        """저장된 임베딩 반환"""
        return self.embeddings
    
    def compute_distance(self, query_embedding: torch.Tensor) -> float:
        """
        쿼리 임베딩과의 최소 거리 계산 (각도 거리)
        Args:
            query_embedding: 쿼리 특징 벡터
        Returns:
            float: 최소 각도 거리 (0~1)
        """
        query_embedding = F.normalize(query_embedding, p=2, dim=0)
        
        min_distance = float('inf')
        for stored_embedding in self.embeddings:
            # 코사인 유사도
            cos_sim = torch.dot(query_embedding.flatten(), stored_embedding.flatten())
            # 각도 거리
            angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
            distance = (angle / np.pi).item()
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def to_dict(self) -> dict:
        """직렬화를 위한 딕셔너리 변환"""
        return {
            'user_id': self.user_id,
            'registered_images': [img.cpu().numpy() for img in self.registered_images],
            'embeddings': [emb.cpu().numpy() for emb in self.embeddings],
            'mean_embedding': self.mean_embedding,
            'update_count': self.update_count,
            'faiss_idx': self.faiss_idx
        }
    
    @classmethod
    def from_dict(cls, data: dict, max_images: int = 3) -> 'UserNode':
        """딕셔너리로부터 UserNode 생성"""
        node = cls(data['user_id'], max_images)
        
        # 이미지와 임베딩 복원
        for img_np in data.get('registered_images', []):
            node.registered_images.append(torch.from_numpy(img_np))
        
        for emb_np in data.get('embeddings', []):
            node.embeddings.append(torch.from_numpy(emb_np))
        
        node.mean_embedding = data.get('mean_embedding')
        node.update_count = data.get('update_count', 0)
        node.faiss_idx = data.get('faiss_idx', -1)
        
        return node
    
    def __repr__(self):
        return (f"UserNode(id={self.user_id}, "
                f"images={len(self.registered_images)}, "
                f"updates={self.update_count})")