# framework/ccnet_style_authentication.py
"""
CCNet 스타일 인증 시스템
- User Node + Faiss를 활용한 빠른 검색
- CCNet과 동일한 거리 메트릭 사용
- 128차원 L2 정규화 벡터
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
from datetime import datetime
import json

class CCNetStyleAuthenticator:
    """
    CCNet 스타일의 인증 시스템
    
    Features:
    - 128차원 L2 정규화 벡터
    - 코사인 거리 (arccos(cosine_similarity) / π)
    - Faiss를 통한 빠른 Top-K 검색
    - EER 기반 임계값 자동 결정
    """
    
    def __init__(self, node_manager, feature_extractor, device='cuda'):
        """
        Args:
            node_manager: SimplifiedUserNodeManager 인스턴스
            feature_extractor: CCNet 모델 (특징 추출용)
            device: 연산 디바이스
        """
        self.node_manager = node_manager
        self.feature_extractor = feature_extractor
        self.device = device
        
        # 임계값 (EER에서 결정됨)
        self.distance_threshold = 0.5  # 초기값, calibrate()에서 업데이트
        
        # 통계
        self.stats = {
            'total_verifications': 0,
            'correct_verifications': 0,
            'false_accepts': 0,
            'false_rejects': 0
        }
        
        print(f"[CCNetAuth] ✅ Initialized")
        print(f"[CCNetAuth] Feature dim: 128")
        print(f"[CCNetAuth] Distance metric: Cosine distance (arccos/π)")
        print(f"[CCNetAuth] Initial threshold: {self.distance_threshold}")
    
    def extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 128차원 L2 정규화 특징 추출
        
        Args:
            image: [1, C, H, W] or [C, H, W] 형태의 이미지
        Returns:
            128차원 L2 정규화된 특징 벡터
        """
        self.feature_extractor.eval()
        
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            
            # 특징 추출 (128차원)
            features = self.feature_extractor.getFeatureCode(image)
            
            # L2 정규화 (CCNet 스타일)
            features = features / torch.norm(features, p=2, dim=1, keepdim=True)
            
        return features.squeeze(0)  # [128]
    
    def compute_cosine_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        CCNet 스타일 코사인 거리 계산
        
        거리 = arccos(cosine_similarity) / π (범위: 0~1)
        """
        # 코사인 유사도 (내적, 이미 L2 정규화됨)
        cosine_sim = torch.dot(feat1, feat2).item()
        
        # 안전한 범위로 클리핑
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # 각도 거리 변환
        distance = np.arccos(cosine_sim) / np.pi
        
        return distance
    
    def verify_user(self, probe_image: torch.Tensor, top_k: int = 10) -> Dict:
        """
        사용자 인증 (CCNet 스타일)
        
        1. 프로브 이미지의 특징 추출
        2. Faiss로 Top-K 후보 빠르게 검색
        3. 후보들의 원본 이미지에서 특징 추출
        4. CCNet 거리 계산으로 정밀 매칭
        
        Returns:
            {
                'is_match': bool,
                'matched_user': int or None,
                'distance': float,
                'confidence': float,
                'top_k_results': [(user_id, distance), ...],
                'computation_time': float
            }
        """
        start_time = time.time()
        
        # 1. 프로브 이미지 특징 추출
        probe_feature = self.extract_feature(probe_image)
        
        # 2. Faiss로 Top-K 후보 검색 (빠른 사전 필터링)
        top_candidates = self.node_manager.find_nearest_users(probe_feature, k=top_k)
        
        if not top_candidates:
            return {
                'is_match': False,
                'matched_user': None,
                'distance': 1.0,
                'confidence': 0.0,
                'top_k_results': [],
                'computation_time': time.time() - start_time
            }
        
        # 3. 정밀 거리 계산 (CCNet 스타일)
        precise_results = []
        
        for user_id, _ in top_candidates:
            node = self.node_manager.get_node(user_id)
            
            if node and node.mean_embedding is not None:
                # CCNet 거리 계산
                distance = self.compute_cosine_distance(probe_feature, node.mean_embedding)
                precise_results.append((user_id, distance))
        
        # 거리 기준 정렬 (오름차순)
        precise_results.sort(key=lambda x: x[1])
        
        # 4. 최종 매칭 결정
        if precise_results:
            best_user_id, best_distance = precise_results[0]
            is_match = best_distance <= self.distance_threshold
            
            # 신뢰도 계산 (거리가 작을수록 높음)
            if is_match:
                confidence = 1.0 - (best_distance / self.distance_threshold)
            else:
                confidence = 0.0
        else:
            is_match = False
            best_user_id = None
            best_distance = 1.0
            confidence = 0.0
        
        # 통계 업데이트
        self.stats['total_verifications'] += 1
        
        computation_time = time.time() - start_time
        
        return {
            'is_match': is_match,
            'matched_user': best_user_id if is_match else None,
            'distance': best_distance,
            'confidence': confidence,
            'threshold': self.distance_threshold,
            'top_k_results': precise_results[:5],  # 상위 5개
            'computation_time': computation_time,
            'method': 'ccnet_style'
        }
    
    def batch_verify(self, probe_images: List[torch.Tensor], 
                    true_labels: List[int]) -> Dict:
        """
        배치 검증 (성능 평가용)
        
        Returns:
            평가 메트릭 (정확도, FAR, FRR 등)
        """
        results = {
            'correct': 0,
            'total': len(probe_images),
            'false_accepts': 0,
            'false_rejects': 0,
            'genuine_distances': [],
            'imposter_distances': [],
            'time_per_verification': []
        }
        
        registered_users = set(self.node_manager.nodes.keys())
        
        for probe_img, true_label in zip(probe_images, true_labels):
            # 인증 수행
            auth_result = self.verify_user(probe_img)
            results['time_per_verification'].append(auth_result['computation_time'])
            
            # 결과 분석
            if auth_result['is_match']:
                if auth_result['matched_user'] == true_label:
                    # True Accept
                    results['correct'] += 1
                    results['genuine_distances'].append(auth_result['distance'])
                else:
                    # False Accept
                    results['false_accepts'] += 1
                    results['imposter_distances'].append(auth_result['distance'])
            else:
                if true_label in registered_users:
                    # False Reject
                    results['false_rejects'] += 1
                    results['genuine_distances'].append(auth_result['distance'])
                else:
                    # True Reject
                    results['correct'] += 1
                    results['imposter_distances'].append(auth_result['distance'])
        
        # 메트릭 계산
        accuracy = results['correct'] / results['total'] * 100
        far = results['false_accepts'] / results['total'] * 100
        frr = results['false_rejects'] / results['total'] * 100
        avg_time = np.mean(results['time_per_verification']) * 1000  # ms
        
        return {
            **results,
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'avg_verification_time_ms': avg_time
        }
    
    def calibrate_threshold(self, calibration_data: List[Tuple[torch.Tensor, int]],
                          target_far: float = 0.01):
        """
        임계값 자동 조정 (EER 또는 목표 FAR 기준)
        
        Args:
            calibration_data: [(image, true_label), ...] 형태의 캘리브레이션 데이터
            target_far: 목표 FAR (예: 0.01 = 1%)
        """
        print(f"\n[Calibration] Starting threshold calibration...")
        print(f"[Calibration] Target FAR: {target_far*100:.2f}%")
        
        all_distances = []
        all_labels = []  # 1: genuine, 0: imposter
        
        # 모든 쌍에 대해 거리 계산
        for i, (probe_img, probe_label) in enumerate(calibration_data):
            probe_feature = self.extract_feature(probe_img)
            
            # 모든 등록된 사용자와 비교
            for user_id, node in self.node_manager.nodes.items():
                if node.mean_embedding is not None:
                    distance = self.compute_cosine_distance(probe_feature, node.mean_embedding)
                    all_distances.append(distance)
                    all_labels.append(1 if user_id == probe_label else 0)
        
        # NumPy 배열로 변환
        distances = np.array(all_distances)
        labels = np.array(all_labels)
        
        # EER 계산
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(labels, -distances, pos_label=1)
        
        # EER 지점 찾기
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # 목표 FAR에 해당하는 임계값 찾기
        far_idx = np.argmax(fpr >= target_far)
        far_threshold = thresholds[far_idx] if far_idx > 0 else thresholds[0]
        
        print(f"\n[Calibration] Results:")
        print(f"  EER: {eer*100:.2f}% at threshold {-eer_threshold:.4f}")
        print(f"  FAR {target_far*100:.1f}% at threshold {-far_threshold:.4f}")
        
        # 임계값 업데이트 (EER 사용)
        self.distance_threshold = -eer_threshold
        
        print(f"  ✅ Threshold updated to: {self.distance_threshold:.4f}")
        
        return {
            'eer': eer,
            'eer_threshold': -eer_threshold,
            'target_far_threshold': -far_threshold,
            'calibration_samples': len(calibration_data),
            'total_comparisons': len(all_distances)
        }
    
    def calculate_rank_accuracy(self, test_data: List[Tuple[torch.Tensor, int]], 
                               max_rank: int = 5) -> Dict:
        """
        Rank-1, Rank-5 등의 정확도 계산 (CCNet 스타일)
        """
        rank_correct = {r: 0 for r in range(1, max_rank + 1)}
        total = len(test_data)
        
        for probe_img, true_label in test_data:
            # Top-K 검색
            auth_result = self.verify_user(probe_img, top_k=max_rank)
            
            # Rank 확인
            for rank, (user_id, _) in enumerate(auth_result['top_k_results'], 1):
                if user_id == true_label:
                    for r in range(rank, max_rank + 1):
                        rank_correct[r] += 1
                    break
        
        # 정확도 계산
        rank_accuracies = {}
        for rank in range(1, max_rank + 1):
            rank_accuracies[f'rank_{rank}'] = (rank_correct[rank] / total) * 100
        
        return rank_accuracies
    
    def save_hard_pairs(self, test_data: List[Tuple[torch.Tensor, int]], 
                       output_dir: str = "./hard_pairs"):
        """
        CCNet처럼 어려운 매칭 쌍 저장
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hard_pairs = []
        
        for probe_img, true_label in test_data:
            auth_result = self.verify_user(probe_img)
            
            # 잘못된 매칭인 경우
            if auth_result['is_match'] and auth_result['matched_user'] != true_label:
                hard_pairs.append({
                    'true_user': true_label,
                    'matched_user': auth_result['matched_user'],
                    'distance': auth_result['distance'],
                    'type': 'false_accept'
                })
            elif not auth_result['is_match'] and true_label in self.node_manager.nodes:
                hard_pairs.append({
                    'true_user': true_label,
                    'matched_user': auth_result['top_k_results'][0][0] if auth_result['top_k_results'] else None,
                    'distance': auth_result['distance'],
                    'type': 'false_reject'
                })
        
        # JSON으로 저장
        with open(output_path / 'hard_pairs.json', 'w') as f:
            json.dump(hard_pairs, f, indent=2)
        
        print(f"[HardPairs] Saved {len(hard_pairs)} hard pairs to {output_path}")
        
        return hard_pairs


def create_ccnet_verifier(model, node_manager, device='cuda'):
    """CCNet 스타일 검증기 생성"""
    return CCNetStyleAuthenticator(node_manager, model, device)


# 사용 예시
if __name__ == "__main__":
    print("\n=== CCNet Style Authentication Demo ===")
    
    # 가상의 설정
    class DummyModel:
        def eval(self):
            pass
        
        def getFeatureCode(self, x):
            # 128차원 특징 반환
            return torch.randn(x.size(0), 128)
    
    class DummyNodeManager:
        def __init__(self):
            self.nodes = {
                1: type('obj', (object,), {'mean_embedding': torch.randn(128) / 2})(),
                2: type('obj', (object,), {'mean_embedding': torch.randn(128) / 2})(),
            }
        
        def find_nearest_users(self, query, k=5):
            return [(1, 0.3), (2, 0.7)]
        
        def get_node(self, user_id):
            return self.nodes.get(user_id)
    
    # 테스트
    model = DummyModel()
    node_manager = DummyNodeManager()
    
    authenticator = CCNetStyleAuthenticator(node_manager, model)
    
    # 인증 테스트
    test_image = torch.randn(1, 128, 128)
    result = authenticator.verify_user(test_image)
    
    print(f"\n인증 결과:")
    print(f"  매칭 여부: {result['is_match']}")
    print(f"  매칭 사용자: {result['matched_user']}")
    print(f"  거리: {result['distance']:.4f}")
    print(f"  신뢰도: {result['confidence']:.2%}")
    print(f"  계산 시간: {result['computation_time']*1000:.2f}ms")
    
    print("\n=== Demo Complete ===")