# evaluation/eval_utils.py - CCNet 스타일로 수정된 버전
"""
CoCoNut Unified Evaluation System

모든 평가 관련 기능을 하나로 통합:
- 기본 성능 평가 (Rank-1, EER)
- CCNet 스타일 인증 (두 이미지 활용)
- End-to-End 평가
- 시각화 및 리포트 생성
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader


class CoconutEvaluator:
    """
    CoCoNut 통합 평가 시스템 - CCNet 스타일
    
    Features:
    - 기본 성능 평가 (Rank-1, EER)
    - CCNet 스타일 인증 (두 이미지 페어 활용)
    - End-to-End 평가
    - 시각화 및 리포트 생성
    """
    
    def __init__(self, model, node_manager=None, device='cuda'):
        """
        Args:
            model: 학습된 CCNet 모델
            node_manager: User Node Manager (optional)
            device: 연산 디바이스
        """
        self.model = model
        self.node_manager = node_manager
        self.device = device
        
        # CCNet 스타일 인증 설정
        self.distance_threshold = 0.5  # 초기값
        self.feature_dim = 128 if hasattr(model, 'headless_mode') and model.headless_mode else 2048
        
        # 통계
        self.stats = {
            'total_verifications': 0,
            'correct_verifications': 0,
            'false_accepts': 0,
            'false_rejects': 0
        }
        
        print(f"[Evaluator] ✅ Initialized (CCNet Style)")
        print(f"[Evaluator] Model type: {'Headless' if hasattr(model, 'headless_mode') and model.headless_mode else 'Classification'}")
        print(f"[Evaluator] Feature dim: {self.feature_dim}")
        if node_manager:
            print(f"[Evaluator] Registered users: {len(node_manager.nodes)}")
    
    # ==================== 기본 평가 함수들 ====================
    
    def extract_features(self, dataloader):
        """데이터로더에서 모든 특징 벡터와 라벨을 추출 - CCNet 스타일"""
        self.model.eval()
        
        features_list = []
        labels_list = []

        with torch.no_grad():
            for datas, target in tqdm(dataloader, desc="Extracting features"):
                # CCNet 스타일: 두 이미지 활용
                data1 = datas[0].to(self.device)
                data2 = datas[1].to(self.device) if len(datas) > 1 else data1
                
                # 두 이미지의 특징 추출
                codes1 = self.model.getFeatureCode(data1)
                codes2 = self.model.getFeatureCode(data2)
                
                # 평균 특징 사용 (더 robust)
                codes = (codes1 + codes2) / 2
                
                features_list.append(codes.cpu().numpy())
                labels_list.append(target.cpu().numpy())
                
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        print(f"  Extracted {len(features)} features (using CCNet style averaging)")
        return features, labels

    def extract_features_separate(self, dataloader):
        """두 이미지를 각각 분리하여 특징 추출 (더 정확한 평가용)"""
        self.model.eval()
        
        features1_list = []
        features2_list = []
        labels_list = []

        with torch.no_grad():
            for datas, target in tqdm(dataloader, desc="Extracting paired features"):
                data1 = datas[0].to(self.device)
                data2 = datas[1].to(self.device) if len(datas) > 1 else data1
                
                codes1 = self.model.getFeatureCode(data1)
                codes2 = self.model.getFeatureCode(data2)
                
                features1_list.append(codes1.cpu().numpy())
                features2_list.append(codes2.cpu().numpy())
                labels_list.append(target.cpu().numpy())
                
        features1 = np.concatenate(features1_list, axis=0)
        features2 = np.concatenate(features2_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        print(f"  Extracted {len(features1)} feature pairs")
        return features1, features2, labels

    def calculate_scores(self, probe_features, gallery_features):
        """Probe와 Gallery 간의 모든 매칭 점수 계산"""
        # L2 정규화
        probe_features = probe_features / np.linalg.norm(probe_features, axis=1, keepdims=True)
        gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
        
        # 코사인 유사도
        cosine_similarity = np.dot(probe_features, gallery_features.T)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # 각도 거리로 변환
        distances = np.arccos(cosine_similarity) / np.pi
        
        return distances

    def calculate_rank_accuracy(self, distances, probe_labels, gallery_labels, max_rank=5):
        """Rank-N 정확도 계산"""
        num_probes = len(probe_labels)
        rank_correct = {r: 0 for r in range(1, max_rank + 1)}
        
        for i in range(num_probes):
            # 가장 가까운 순서대로 정렬
            sorted_indices = np.argsort(distances[i])
            
            for rank in range(1, max_rank + 1):
                # Top-K 안에 정답이 있는지 확인
                top_k_labels = gallery_labels[sorted_indices[:rank]]
                if probe_labels[i] in top_k_labels:
                    rank_correct[rank] += 1
        
        rank_accuracies = {}
        for rank in range(1, max_rank + 1):
            rank_accuracies[f'rank_{rank}'] = (rank_correct[rank] / num_probes) * 100
            
        return rank_accuracies

    def calculate_eer(self, genuine_scores, imposter_scores):
        """EER (Equal Error Rate) 계산"""
        if len(genuine_scores) == 0 or len(imposter_scores) == 0:
            return 0.0, 0.0
        
        # 음수 점수 처리
        all_scores = np.concatenate([genuine_scores, imposter_scores])
        min_score = np.min(all_scores)
        
        if min_score < 0:
            genuine_scores = genuine_scores - min_score
            imposter_scores = imposter_scores - min_score
        
        # 라벨 생성 (1: genuine, 0: imposter)
        labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
        scores = np.concatenate([genuine_scores, imposter_scores])

        try:
            # ROC 커브
            fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
            
            # EER 계산
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
            
        except Exception:
            # 대안 방법
            fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
            fnr = 1 - tpr
            
            diff = np.abs(fpr - fnr)
            eer_idx = np.argmin(diff)
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
            thresh = thresholds[eer_idx]

        return eer * 100, thresh

    def perform_basic_evaluation(self, train_loader, test_loader):
        """기본 성능 평가 (Rank-1, EER) - CCNet 스타일"""
        print("\n[Basic Evaluation] Starting (CCNet Style)...")
        
        # 1. 특징 추출 (두 이미지 평균 사용)
        gallery_features, gallery_labels = self.extract_features(train_loader)
        probe_features, probe_labels = self.extract_features(test_loader)
        
        # 2. 매칭 점수 계산
        print("  Calculating matching scores...")
        distances = self.calculate_scores(probe_features, gallery_features)
        
        # 3. Rank 정확도 계산
        rank_accuracies = self.calculate_rank_accuracy(distances, probe_labels, gallery_labels)
        print(f"  Rank-1 Accuracy: {rank_accuracies['rank_1']:.3f}%")
        
        # 4. EER 계산
        genuine_scores = []
        imposter_scores = []
        for i in range(len(probe_labels)):
            for j in range(len(gallery_labels)):
                score = distances[i, j]
                if probe_labels[i] == gallery_labels[j]:
                    genuine_scores.append(score)
                else:
                    imposter_scores.append(score)

        eer, threshold = self.calculate_eer(np.array(genuine_scores), np.array(imposter_scores))
        print(f"  EER: {eer:.4f}% at Threshold: {threshold:.4f}")
        
        results = {
            **rank_accuracies,
            'eer': eer,
            'eer_threshold': threshold,
            'num_gallery': len(gallery_labels),
            'num_probe': len(probe_labels)
        }
        
        return results
    
    # ==================== CCNet 스타일 인증 ====================
    
    def compute_ccnet_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """CCNet 스타일 코사인 거리 계산"""
        # L2 정규화
        feat1 = F.normalize(feat1, dim=0)
        feat2 = F.normalize(feat2, dim=0)
        
        # 코사인 유사도
        cosine_sim = torch.dot(feat1, feat2).item()
        
        # 안전한 범위로 클리핑
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # 각도 거리 변환
        distance = np.arccos(cosine_sim) / np.pi
        
        return distance
    
    def verify_user(self, probe_image: torch.Tensor, gallery_image: Optional[torch.Tensor] = None, 
                   top_k: int = 10) -> Dict:
        """
        사용자 인증 (CCNet 스타일)
        
        Args:
            probe_image: 프로브 이미지
            gallery_image: 갤러리 이미지 (옵션, CCNet 스타일)
            top_k: 상위 K개 결과
            
        Returns:
            인증 결과 딕셔너리
        """
        if not self.node_manager:
            return {
                'is_match': False,
                'error': 'No node manager available'
            }
        
        start_time = time.time()
        
        # 1. 특징 추출
        self.model.eval()
        with torch.no_grad():
            # 프로브 이미지
            if len(probe_image.shape) == 3:
                probe_image = probe_image.unsqueeze(0)
            probe_image = probe_image.to(self.device)
            probe_feature = self.model.getFeatureCode(probe_image).squeeze(0)
            
            # CCNet 스타일: 갤러리 이미지도 있으면 평균 사용
            if gallery_image is not None:
                if len(gallery_image.shape) == 3:
                    gallery_image = gallery_image.unsqueeze(0)
                gallery_image = gallery_image.to(self.device)
                gallery_feature = self.model.getFeatureCode(gallery_image).squeeze(0)
                
                # 두 특징의 평균 (더 robust)
                combined_feature = (probe_feature + gallery_feature) / 2
            else:
                combined_feature = probe_feature
        
        # 2. 가장 가까운 사용자 찾기
        top_candidates = self.node_manager.find_nearest_users(combined_feature, k=top_k)
        
        if not top_candidates:
            return {
                'is_match': False,
                'matched_user': None,
                'distance': 1.0,
                'confidence': 0.0,
                'computation_time': time.time() - start_time
            }
        
        # 3. 정밀 거리 계산
        precise_results = []
        
        for user_id, _ in top_candidates:
            node = self.node_manager.get_node(user_id)
            
            if node and node.mean_embedding is not None:
                distance = self.compute_ccnet_distance(combined_feature, node.mean_embedding)
                precise_results.append((user_id, distance))
        
        # 거리 기준 정렬
        precise_results.sort(key=lambda x: x[1])
        
        # 4. 최종 매칭 결정
        if precise_results:
            best_user_id, best_distance = precise_results[0]
            is_match = best_distance <= self.distance_threshold
            
            confidence = 1.0 - (best_distance / self.distance_threshold) if is_match else 0.0
            confidence = min(1.0, max(0.0, confidence))
        else:
            is_match = False
            best_user_id = None
            best_distance = 1.0
            confidence = 0.0
        
        # 통계 업데이트
        self.stats['total_verifications'] += 1
        
        return {
            'is_match': is_match,
            'matched_user': best_user_id if is_match else None,
            'distance': best_distance,
            'confidence': confidence,
            'threshold': self.distance_threshold,
            'top_k_results': precise_results[:5],
            'computation_time': time.time() - start_time,
            'used_gallery': gallery_image is not None
        }
    
    def calibrate_threshold(self, calibration_data: List[Union[Tuple[torch.Tensor, int], 
                                                               Tuple[Tuple[torch.Tensor, torch.Tensor], int]]],
                          target_far: float = 0.01):
        """임계값 자동 조정 - CCNet 스타일 지원"""
        print(f"\n[Calibration] Starting threshold calibration...")
        print(f"  Target FAR: {target_far*100:.2f}%")
        
        all_distances = []
        all_labels = []  # 1: genuine, 0: imposter
        
        # 모든 쌍에 대해 거리 계산
        for data_item, true_label in tqdm(calibration_data, desc="Calibrating"):
            # CCNet 스타일: 튜플이면 두 이미지
            if isinstance(data_item, tuple):
                probe_img, gallery_img = data_item
                result = self.verify_user(probe_img, gallery_img)
            else:
                result = self.verify_user(data_item)
            
            if 'top_k_results' in result:
                for user_id, distance in result['top_k_results']:
                    all_distances.append(distance)
                    all_labels.append(1 if user_id == true_label else 0)
        
        if not all_distances:
            print("  ⚠️ No distances calculated")
            return None
        
        # NumPy 배열로 변환
        distances = np.array(all_distances)
        labels = np.array(all_labels)
        
        # EER 계산
        fpr, tpr, thresholds = metrics.roc_curve(labels, -distances, pos_label=1)
        
        # EER 지점 찾기
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = -thresholds[eer_idx]
        
        # 목표 FAR에 해당하는 임계값
        far_idx = np.argmax(fpr >= target_far)
        far_threshold = -thresholds[far_idx] if far_idx > 0 else -thresholds[0]
        
        print(f"\n[Calibration] Results:")
        print(f"  EER: {eer*100:.2f}% at threshold {eer_threshold:.4f}")
        print(f"  FAR {target_far*100:.1f}% at threshold {far_threshold:.4f}")
        
        # 임계값 업데이트
        self.distance_threshold = eer_threshold
        
        print(f"  ✅ Threshold updated to: {self.distance_threshold:.4f}")
        
        return {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'target_far_threshold': far_threshold,
            'calibration_samples': len(calibration_data)
        }
    
    # ==================== End-to-End 평가 ====================
    
    def run_end_to_end_evaluation(self, test_file_path: str, 
                                 batch_size: int = 32,
                                 save_results: bool = True,
                                 output_dir: str = "./evaluation_results",
                                 use_ccnet_style: bool = True) -> Dict:
        """
        End-to-End 인증 평가
        
        Args:
            test_file_path: 테스트 파일 경로
            batch_size: 배치 크기
            save_results: 결과 저장 여부
            output_dir: 결과 저장 경로
            use_ccnet_style: CCNet 스타일 (두 이미지) 사용 여부
            
        Returns:
            종합 평가 결과
        """
        print("\n" + "="*80)
        print("🔍 END-TO-END AUTHENTICATION EVALUATION")
        print(f"   Mode: {'CCNet Style (2 images)' if use_ccnet_style else 'Single Image'}")
        print("="*80)
        
        start_time = time.time()
        
        # 1. 테스트 데이터 로드
        print("\n[Step 1/5] Loading test data...")
        test_dataset = MyDataset(txt=test_file_path, train=False)
        test_samples, test_labels = self._prepare_test_data(test_dataset, use_ccnet_style)
        
        # 2. 임계값 캘리브레이션
        if len(test_samples) > 500:
            print("\n[Step 2/5] Calibrating threshold...")
            calibration_data = list(zip(test_samples[:500], test_labels[:500]))
            calibration_result = self.calibrate_threshold(calibration_data)
        else:
            print("\n[Step 2/5] Skipping calibration (insufficient data)")
            calibration_result = None
        
        # 3. 전체 테스트셋 평가
        print("\n[Step 3/5] Evaluating full test set...")
        eval_results = self._evaluate_test_set(test_samples, test_labels, use_ccnet_style)
        
        # 4. 상세 분석
        print("\n[Step 4/5] Analyzing results...")
        analysis_results = self._analyze_results(eval_results)
        
        # 5. 결과 저장 및 시각화
        if save_results:
            print("\n[Step 5/5] Saving results and visualizations...")
            self._save_results(eval_results, analysis_results, output_dir)
        
        total_time = time.time() - start_time
        
        # 종합 결과
        summary = {
            'test_samples': len(test_samples),
            'registered_users': len(self.node_manager.nodes) if self.node_manager else 0,
            'accuracy': analysis_results['accuracy'],
            'eer': analysis_results['eer'],
            'rank1_accuracy': analysis_results.get('rank_accuracies', {}).get('rank_1', 0),
            'far': analysis_results['far'],
            'frr': analysis_results['frr'],
            'avg_verification_time_ms': analysis_results['avg_time_ms'],
            'total_evaluation_time': total_time,
            'calibration_result': calibration_result,
            'threshold_used': self.distance_threshold,
            'evaluation_mode': 'CCNet Style' if use_ccnet_style else 'Single Image'
        }
        
        # 결과 출력
        self._print_summary(summary)
        
        return summary
    
    def _prepare_test_data(self, test_dataset, use_ccnet_style: bool = True) -> Tuple[List, List[int]]:
        """테스트 데이터 준비 - CCNet 스타일 옵션"""
        test_samples = []
        test_labels = []
        
        for idx in tqdm(range(len(test_dataset)), desc="Loading test data"):
            data, label = test_dataset[idx]
            
            if use_ccnet_style and len(data) >= 2:
                # CCNet 스타일: 두 이미지를 튜플로
                test_samples.append((data[0], data[1]))
            else:
                # 단일 이미지
                test_samples.append(data[0])
                
            test_labels.append(label if isinstance(label, int) else label.item())
        
        print(f"  Loaded {len(test_samples)} test samples")
        print(f"  Using {'image pairs' if use_ccnet_style else 'single images'}")
        print(f"  Unique users in test set: {len(set(test_labels))}")
        
        return test_samples, test_labels
    
    def _evaluate_test_set(self, test_samples: List, test_labels: List[int], 
                          use_ccnet_style: bool = True) -> List[Dict]:
        """전체 테스트셋 평가 - CCNet 스타일 지원"""
        all_results = []
        registered_users = set(self.node_manager.nodes.keys()) if self.node_manager else set()
        
        for sample, true_label in tqdm(zip(test_samples, test_labels), 
                                     total=len(test_samples),
                                     desc="Evaluating"):
            # 인증 수행
            if use_ccnet_style and isinstance(sample, tuple):
                # CCNet 스타일: 두 이미지 사용
                auth_result = self.verify_user(sample[0], sample[1])
            else:
                # 단일 이미지
                auth_result = self.verify_user(sample)
            
            # 결과 저장
            result_entry = {
                'true_label': true_label,
                'is_registered': true_label in registered_users,
                'prediction': auth_result['matched_user'] if auth_result['is_match'] else None,
                'is_match': auth_result['is_match'],
                'distance': auth_result.get('distance', 1.0),
                'confidence': auth_result.get('confidence', 0.0),
                'top_5_results': auth_result.get('top_k_results', [])[:5],
                'computation_time': auth_result.get('computation_time', 0),
                'used_ccnet_style': auth_result.get('used_gallery', False)
            }
            all_results.append(result_entry)
        
        return all_results
    
    def _analyze_results(self, eval_results: List[Dict]) -> Dict:
        """결과 상세 분석"""
        total = len(eval_results)
        correct = 0
        false_accepts = 0
        false_rejects = 0
        true_rejects = 0
        
        genuine_distances = []
        imposter_distances = []
        
        rank_correct = {r: 0 for r in range(1, 6)}
        computation_times = []
        ccnet_style_count = 0
        
        for result in eval_results:
            computation_times.append(result['computation_time'])
            
            if result.get('used_ccnet_style', False):
                ccnet_style_count += 1
            
            if result['is_registered']:
                # 등록된 사용자
                if result['is_match'] and result['prediction'] == result['true_label']:
                    correct += 1
                    genuine_distances.append(result['distance'])
                elif result['is_match'] and result['prediction'] != result['true_label']:
                    false_accepts += 1
                    imposter_distances.append(result['distance'])
                else:  # not is_match
                    false_rejects += 1
                    genuine_distances.append(result['distance'])
                
                # Rank 정확도
                for rank, (user_id, _) in enumerate(result['top_5_results'], 1):
                    if user_id == result['true_label']:
                        for r in range(rank, 6):
                            rank_correct[r] += 1
                        break
            else:
                # 미등록 사용자
                if not result['is_match']:
                    correct += 1
                    true_rejects += 1
                    imposter_distances.append(result['distance'])
                else:
                    false_accepts += 1
                    imposter_distances.append(result['distance'])
        
        # 메트릭 계산
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        registered_count = sum(1 for r in eval_results if r['is_registered'])
        if registered_count > 0:
            far = (false_accepts / total) * 100
            frr = (false_rejects / registered_count) * 100
        else:
            far = frr = 0
        
        # EER 계산
        if genuine_distances and imposter_distances:
            eer, eer_threshold = self.calculate_eer(
                np.array(genuine_distances), 
                np.array(imposter_distances)
            )
        else:
            eer, eer_threshold = 0, 0
        
        # Rank 정확도
        rank_accuracies = {}
        if registered_count > 0:
            for rank in range(1, 6):
                rank_accuracies[f'rank_{rank}'] = (rank_correct[rank] / registered_count) * 100
        
        # 시간 통계
        avg_time_ms = np.mean(computation_times) * 1000 if computation_times else 0
        
        return {
            'total_samples': total,
            'registered_samples': registered_count,
            'correct': correct,
            'accuracy': accuracy,
            'false_accepts': false_accepts,
            'false_rejects': false_rejects,
            'true_rejects': true_rejects,
            'far': far,
            'frr': frr,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'genuine_distances': genuine_distances,
            'imposter_distances': imposter_distances,
            'rank_accuracies': rank_accuracies,
            'avg_time_ms': avg_time_ms,
            'min_time_ms': np.min(computation_times) * 1000 if computation_times else 0,
            'max_time_ms': np.max(computation_times) * 1000 if computation_times else 0,
            'ccnet_style_usage': (ccnet_style_count / total * 100) if total > 0 else 0
        }
    
    # ==================== 시각화 및 리포트 ====================
    
    def _save_results(self, eval_results: List[Dict], 
                     analysis: Dict, output_dir: str):
        """결과 저장 및 시각화"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON 결과 저장
        summary_data = {
            'timestamp': timestamp,
            'analysis': analysis,
            'threshold_used': self.distance_threshold,
            'node_manager_stats': self.node_manager.get_statistics() if self.node_manager else None,
            'evaluation_mode': 'CCNet Style' if analysis.get('ccnet_style_usage', 0) > 0 else 'Single Image'
        }
        
        with open(output_path / f'evaluation_summary_{timestamp}.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # 2. 시각화
        self._create_visualizations(analysis, output_path, timestamp)
        
        print(f"  ✅ Results saved to: {output_path}")
    
    def _create_visualizations(self, analysis: Dict, 
                              output_path: Path, timestamp: str):
        """결과 시각화"""
        plt.style.use('default')  # seaborn-v0_8-darkgrid 대신 default 사용
        
        # 1. 거리 분포 히스토그램
        plt.figure(figsize=(12, 6))
        
        if analysis['genuine_distances']:
            plt.hist(analysis['genuine_distances'], bins=50, alpha=0.7, 
                    label=f'Genuine (n={len(analysis["genuine_distances"])})', 
                    color='green', density=True)
        
        if analysis['imposter_distances']:
            plt.hist(analysis['imposter_distances'], bins=50, alpha=0.7,
                    label=f'Imposter (n={len(analysis["imposter_distances"])})', 
                    color='red', density=True)
        
        plt.axvline(self.distance_threshold, color='blue', 
                   linestyle='--', linewidth=2,
                   label=f'Threshold = {self.distance_threshold:.3f}')
        
        plt.xlabel('Distance')
        plt.ylabel('Density')
        plt.title('Distance Distribution (CCNet Style)' if analysis.get('ccnet_style_usage', 0) > 50 
                  else 'Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'distance_distribution_{timestamp}.png', dpi=300)
        plt.close()
        
        # 2. ROC 커브
        if analysis['genuine_distances'] and analysis['imposter_distances']:
            plt.figure(figsize=(8, 8))
            
            labels = ([1] * len(analysis['genuine_distances']) + 
                     [0] * len(analysis['imposter_distances']))
            scores = ([-d for d in analysis['genuine_distances']] + 
                     [-d for d in analysis['imposter_distances']])
            
            fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)
            
            plt.plot(fpr * 100, tpr * 100, 'b-', linewidth=2, 
                    label=f'ROC (AUC = {auc_score:.4f})')
            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
            
            eer_point = analysis['eer']
            plt.plot(eer_point, 100 - eer_point, 'ro', markersize=10, 
                    label=f'EER = {eer_point:.2f}%')
            
            plt.xlabel('False Acceptance Rate (%)')
            plt.ylabel('Genuine Acceptance Rate (%)')
            plt.title('ROC Curve (CCNet Style)' if analysis.get('ccnet_style_usage', 0) > 50 
                      else 'ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 20])
            plt.ylim([80, 100])
            plt.tight_layout()
            plt.savefig(output_path / f'roc_curve_{timestamp}.png', dpi=300)
            plt.close()
        
        # 3. Rank 정확도
        if analysis['rank_accuracies']:
            plt.figure(figsize=(10, 6))
            
            ranks = list(range(1, 6))
            accuracies = [analysis['rank_accuracies'].get(f'rank_{r}', 0) for r in ranks]
            
            bars = plt.bar(ranks, accuracies, color='skyblue', edgecolor='navy')
            
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            plt.xlabel('Rank')
            plt.ylabel('Accuracy (%)')
            plt.title('Rank-N Accuracy (CCNet Style)' if analysis.get('ccnet_style_usage', 0) > 50 
                      else 'Rank-N Accuracy')
            plt.ylim([0, 105])
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / f'rank_accuracy_{timestamp}.png', dpi=300)
            plt.close()
    
    def _print_summary(self, summary: Dict):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\n🔍 Dataset Information:")
        print(f"  - Test samples: {summary['test_samples']}")
        print(f"  - Registered users: {summary['registered_users']}")
        print(f"  - Distance threshold: {summary['threshold_used']:.4f}")
        print(f"  - Evaluation mode: {summary.get('evaluation_mode', 'Unknown')}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  - Overall Accuracy: {summary['accuracy']:.2f}%")
        print(f"  - Rank-1 Accuracy: {summary['rank1_accuracy']:.2f}%")
        print(f"  - EER: {summary['eer']:.2f}%")
        print(f"  - FAR: {summary['far']:.2f}%")
        print(f"  - FRR: {summary['frr']:.2f}%")
        
        print(f"\n⚡ Speed Performance:")
        print(f"  - Avg verification time: {summary['avg_verification_time_ms']:.2f} ms")
        print(f"  - Total evaluation time: {summary['total_evaluation_time']:.1f} seconds")
        
        print("\n" + "="*80)


# 편의 함수들
def perform_evaluation(model, train_loader, test_loader, device):
    """기본 성능 평가 실행 (기존 코드와의 호환성)"""
    evaluator = CoconutEvaluator(model, device=device)
    return evaluator.perform_basic_evaluation(train_loader, test_loader)

def run_end_to_end_evaluation(model, node_manager, config):
    """End-to-End 평가 실행 (기존 코드와의 호환성)"""
    test_file = getattr(config.dataset, 'test_set_file', None)
    if not test_file:
        print("⚠️ No test file specified in config")
        return None
    
    evaluator = CoconutEvaluator(model, node_manager)
    return evaluator.run_end_to_end_evaluation(
        test_file_path=test_file,
        save_results=True,
        output_dir="./evaluation_results",
        use_ccnet_style=True  # CCNet 스타일 사용
    )

# 테스트 코드
if __name__ == "__main__":
    print("CCNet Style Evaluation Module Test")
    print("This module supports both single image and CCNet-style (2 images) evaluation")