# framework/evaluate_authentication.py
"""
End-to-End 인증 평가 시스템
온라인 학습 후 테스트셋으로 전체 성능 평가
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from datasets.palm_dataset import MyDataset
from torch.utils.data import DataLoader
from framework.ccnet_style_authentication import CCNetStyleAuthenticator

class EndToEndEvaluator:
    """
    온라인 학습 후 End-to-End 평가
    
    1. 테스트셋 로드
    2. CCNet 스타일 인증 수행
    3. 상세한 성능 메트릭 계산
    4. 시각화 및 리포트 생성
    """
    
    def __init__(self, model, node_manager, test_file_path: str, device='cuda'):
        """
        Args:
            model: 학습된 CCNet 모델
            node_manager: User Node Manager
            test_file_path: 테스트 파일 경로 (txt)
            device: 연산 디바이스
        """
        self.model = model
        self.node_manager = node_manager
        self.test_file_path = test_file_path
        self.device = device
        
        # CCNet 스타일 인증기
        self.authenticator = CCNetStyleAuthenticator(node_manager, model, device)
        
        # 테스트 데이터셋 로드
        self.test_dataset = MyDataset(txt=test_file_path, train=False)
        
        print(f"[Evaluator] ✅ Initialized")
        print(f"[Evaluator] Test samples: {len(self.test_dataset)}")
        print(f"[Evaluator] Registered users: {len(node_manager.nodes)}")
        
    def run_full_evaluation(self, batch_size: int = 32, 
                          save_results: bool = True,
                          output_dir: str = "./evaluation_results") -> Dict:
        """
        전체 평가 실행
        
        Returns:
            종합 평가 결과
        """
        print("\n" + "="*80)
        print("🔍 END-TO-END AUTHENTICATION EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # 1. 데이터 준비
        print("\n[Step 1/5] Preparing test data...")
        test_samples, test_labels = self._prepare_test_data()
        
        # 2. 임계값 캘리브레이션 (선택사항)
        print("\n[Step 2/5] Calibrating threshold...")
        calibration_result = self._calibrate_threshold(test_samples[:500], test_labels[:500])
        
        # 3. 전체 테스트셋 평가
        print("\n[Step 3/5] Evaluating full test set...")
        eval_results = self._evaluate_test_set(test_samples, test_labels)
        
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
            'registered_users': len(self.node_manager.nodes),
            'accuracy': analysis_results['accuracy'],
            'eer': analysis_results['eer'],
            'rank1_accuracy': analysis_results['rank_accuracies']['rank_1'],
            'far': analysis_results['far'],
            'frr': analysis_results['frr'],
            'avg_verification_time_ms': analysis_results['avg_time_ms'],
            'total_evaluation_time': total_time,
            'calibration_result': calibration_result,
            'threshold_used': self.authenticator.distance_threshold
        }
        
        # 결과 출력
        self._print_summary(summary)
        
        return summary
    
    def _prepare_test_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """테스트 데이터 준비"""
        test_samples = []
        test_labels = []
        
        # 데이터셋에서 샘플 추출
        for idx in tqdm(range(len(self.test_dataset)), desc="Loading test data"):
            data, label = self.test_dataset[idx]
            # 첫 번째 이미지만 사용 (CCNet은 pair로 로드하므로)
            test_samples.append(data[0])
            test_labels.append(label if isinstance(label, int) else label.item())
        
        print(f"  Loaded {len(test_samples)} test samples")
        print(f"  Unique users in test set: {len(set(test_labels))}")
        
        return test_samples, test_labels
    
    def _calibrate_threshold(self, samples: List[torch.Tensor], 
                           labels: List[int]) -> Dict:
        """임계값 자동 조정"""
        calibration_data = list(zip(samples, labels))
        result = self.authenticator.calibrate_threshold(
            calibration_data, 
            target_far=0.01  # 1% FAR
        )
        return result
    
    def _evaluate_test_set(self, test_samples: List[torch.Tensor], 
                          test_labels: List[int]) -> Dict:
        """전체 테스트셋 평가"""
        all_results = []
        registered_users = set(self.node_manager.nodes.keys())
        
        # 진행률 표시하며 평가
        for sample, true_label in tqdm(zip(test_samples, test_labels), 
                                     total=len(test_samples),
                                     desc="Evaluating"):
            # 인증 수행
            auth_result = self.authenticator.verify_user(sample, top_k=10)
            
            # 결과 저장
            result_entry = {
                'true_label': true_label,
                'is_registered': true_label in registered_users,
                'prediction': auth_result['matched_user'] if auth_result['is_match'] else None,
                'is_match': auth_result['is_match'],
                'distance': auth_result['distance'],
                'confidence': auth_result['confidence'],
                'top_5_results': auth_result['top_k_results'][:5],
                'computation_time': auth_result['computation_time']
            }
            all_results.append(result_entry)
        
        return all_results
    
    def _analyze_results(self, eval_results: List[Dict]) -> Dict:
        """결과 상세 분석"""
        # 기본 통계
        total = len(eval_results)
        correct = 0
        false_accepts = 0
        false_rejects = 0
        true_rejects = 0
        
        # 거리 분포
        genuine_distances = []
        imposter_distances = []
        
        # Rank 정확도 계산용
        rank_correct = {r: 0 for r in range(1, 6)}
        
        # 시간 통계
        computation_times = []
        
        for result in eval_results:
            computation_times.append(result['computation_time'])
            
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
        accuracy = (correct / total) * 100
        
        # 등록된 사용자만 고려한 메트릭
        registered_count = sum(1 for r in eval_results if r['is_registered'])
        if registered_count > 0:
            far = (false_accepts / total) * 100
            frr = (false_rejects / registered_count) * 100
        else:
            far = frr = 0
        
        # EER 계산
        if genuine_distances and imposter_distances:
            eer, eer_threshold = self._calculate_eer(genuine_distances, imposter_distances)
        else:
            eer, eer_threshold = 0, 0
        
        # Rank 정확도
        rank_accuracies = {}
        if registered_count > 0:
            for rank in range(1, 6):
                rank_accuracies[f'rank_{rank}'] = (rank_correct[rank] / registered_count) * 100
        
        # 시간 통계
        avg_time_ms = np.mean(computation_times) * 1000
        
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
            'min_time_ms': np.min(computation_times) * 1000,
            'max_time_ms': np.max(computation_times) * 1000
        }
    
    def _calculate_eer(self, genuine_distances: List[float], 
                      imposter_distances: List[float]) -> Tuple[float, float]:
        """EER 계산"""
        # 라벨 생성 (genuine=1, imposter=0)
        labels = [1] * len(genuine_distances) + [0] * len(imposter_distances)
        scores = genuine_distances + imposter_distances
        
        # 거리를 음수로 변환 (낮은 거리 = 높은 점수)
        scores = [-s for s in scores]
        
        # ROC 커브
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        
        # EER 지점 찾기
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = -thresholds[eer_idx]  # 다시 양수로
        
        return eer * 100, eer_threshold
    
    def _save_results(self, eval_results: List[Dict], 
                     analysis: Dict, output_dir: str):
        """결과 저장 및 시각화"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. JSON 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_data = {
            'timestamp': timestamp,
            'test_file': self.test_file_path,
            'analysis': analysis,
            'threshold_used': self.authenticator.distance_threshold,
            'node_manager_stats': self.node_manager.get_statistics()
        }
        
        with open(output_path / f'evaluation_summary_{timestamp}.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # 2. 상세 결과 저장 (옵션)
        # with open(output_path / f'detailed_results_{timestamp}.json', 'w') as f:
        #     json.dump(eval_results, f, indent=2)
        
        # 3. 시각화
        self._create_visualizations(analysis, output_path, timestamp)
        
        print(f"  ✅ Results saved to: {output_path}")
    
    def _create_visualizations(self, analysis: Dict, 
                              output_path: Path, timestamp: str):
        """결과 시각화"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
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
        
        plt.axvline(self.authenticator.distance_threshold, color='blue', 
                   linestyle='--', linewidth=2,
                   label=f'Threshold = {self.authenticator.distance_threshold:.3f}')
        
        plt.xlabel('Cosine Distance (arccos/π)')
        plt.ylabel('Density')
        plt.title('Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'distance_distribution_{timestamp}.png', dpi=300)
        plt.close()
        
        # 2. ROC 커브
        if analysis['genuine_distances'] and analysis['imposter_distances']:
            plt.figure(figsize=(8, 8))
            
            # ROC 계산
            labels = ([1] * len(analysis['genuine_distances']) + 
                     [0] * len(analysis['imposter_distances']))
            scores = ([-d for d in analysis['genuine_distances']] + 
                     [-d for d in analysis['imposter_distances']])
            
            fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
            auc_score = metrics.auc(fpr, tpr)
            
            plt.plot(fpr * 100, tpr * 100, 'b-', linewidth=2, 
                    label=f'ROC (AUC = {auc_score:.4f})')
            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
            
            # EER 점 표시
            eer_point = analysis['eer']
            plt.plot(eer_point, 100 - eer_point, 'ro', markersize=10, 
                    label=f'EER = {eer_point:.2f}%')
            
            plt.xlabel('False Acceptance Rate (%)')
            plt.ylabel('Genuine Acceptance Rate (%)')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 20])  # 관심 영역 확대
            plt.ylim([80, 100])
            plt.tight_layout()
            plt.savefig(output_path / f'roc_curve_{timestamp}.png', dpi=300)
            plt.close()
        
        # 3. Rank 정확도 막대 그래프
        if analysis['rank_accuracies']:
            plt.figure(figsize=(10, 6))
            
            ranks = list(range(1, 6))
            accuracies = [analysis['rank_accuracies'][f'rank_{r}'] for r in ranks]
            
            bars = plt.bar(ranks, accuracies, color='skyblue', edgecolor='navy')
            
            # 값 표시
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            plt.xlabel('Rank')
            plt.ylabel('Accuracy (%)')
            plt.title('Rank-N Accuracy')
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


def run_end_to_end_evaluation(model, node_manager, config):
    """
    온라인 학습 후 실행할 End-to-End 평가
    
    Args:
        model: 학습된 모델
        node_manager: User Node Manager
        config: 설정 객체
    """
    # 테스트 파일 경로
    test_file = getattr(config.dataset, 'test_set_file', None)
    if not test_file:
        print("⚠️ No test file specified in config")
        return None
    
    # 평가기 생성
    evaluator = EndToEndEvaluator(
        model=model,
        node_manager=node_manager,
        test_file_path=test_file,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 전체 평가 실행
    results = evaluator.run_full_evaluation(
        batch_size=32,
        save_results=True,
        output_dir="./evaluation_results"
    )
    
    return results


# 사용 예시
if __name__ == "__main__":
    print("End-to-End Evaluation Module")
    print("This module evaluates authentication performance after online learning")