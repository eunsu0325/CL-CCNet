# evaluation/eval_utils.py - 확장된 평가 유틸리티
"""
COCONUT Evaluation Utilities (Extended)

FEATURES:
- Feature extraction and similarity computation
- Rank-1 accuracy calculation  
- EER (Equal Error Rate) calculation
- ROC curve analysis
- 🔥 NEW: Detailed biometric metrics (FAR/FRR/AUC)
- 🔥 NEW: Visualization and report generation
- 🔥 NEW: COCONUT Headless mode specialized evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
from pathlib import Path
import json
from datetime import datetime

# 기존 함수들 유지
def extract_features(model, dataloader, device):
    """주어진 데이터로더에서 모든 특징 벡터와 라벨을 추출합니다."""
    model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []

    with torch.no_grad():
        for datas, target in tqdm(dataloader, desc="Extracting features"):
            data = datas[0].to(device)
            codes = model.getFeatureCode(data)
            
            features_list.append(codes.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"Extracted {len(features)} features.")
    return features, labels

def calculate_scores(probe_features, gallery_features):
    """Probe 세트와 Gallery 세트 간의 모든 매칭 점수를 계산합니다."""
    cosine_similarity = np.dot(probe_features, gallery_features.T)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    distances = np.arccos(cosine_similarity) / np.pi
    
    return distances

def calculate_eer(genuine_scores, imposter_scores):
    """정규 매칭 점수와 비정규 매칭 점수를 바탕으로 EER을 계산합니다. (음수 점수 지원)"""
    
    # 🔥 음수 점수 처리: 전체 점수를 양수로 shift
    all_scores = np.concatenate([genuine_scores, imposter_scores])
    min_score = np.min(all_scores)
    
    if min_score < 0:
        print(f"[EER] 음수 점수 감지: {min_score:.6f}, 양수로 shift 적용")
        genuine_scores = genuine_scores - min_score
        imposter_scores = imposter_scores - min_score
        print(f"[EER] Shift 후 범위: [{np.min(all_scores - min_score):.6f}, {np.max(all_scores - min_score):.6f}]")
    
    # 🔥 점수 통계 출력
    print(f"[EER Stats] Genuine: count={len(genuine_scores)}, min={np.min(genuine_scores):.4f}, max={np.max(genuine_scores):.4f}, mean={np.mean(genuine_scores):.4f}")
    print(f"[EER Stats] Imposter: count={len(imposter_scores)}, min={np.min(imposter_scores):.4f}, max={np.max(imposter_scores):.4f}, mean={np.mean(imposter_scores):.4f}")
    
    # 라벨 생성
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    scores = np.concatenate([genuine_scores, imposter_scores])

    try:
        # 🔥 원래 방식: -scores 사용 (거리가 아닌 유사도로 변환)
        fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
        
        # EER 계산 시도
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(-eer)
        
        print(f"[EER] 성공적으로 계산됨: {eer*100:.4f}% at threshold {thresh:.6f}")
        
    except Exception as e:
        print(f"[EER] 보간 계산 실패: {e}")
        print("[EER] 대안 방법 사용...")
        
        # 🔥 대안: 직접 EER 지점 찾기
        fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
        fnr = 1 - tpr
        
        # FPR과 FNR이 가장 가까운 지점 찾기
        diff = np.abs(fpr - fnr)
        eer_idx = np.argmin(diff)
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        thresh = thresholds[eer_idx]
        
        print(f"[EER] 대안 계산 결과: {eer*100:.4f}% at threshold {thresh:.6f}")

    return eer * 100, thresh

def calculate_rank1(distances, probe_labels, gallery_labels):
    """거리 행렬을 바탕으로 Rank-1 정확도를 계산합니다."""
    correct_predictions = 0
    num_probes = len(probe_labels)

    indices_of_closest = np.argmin(distances, axis=1)
    
    for i in range(num_probes):
        predicted_label = gallery_labels[indices_of_closest[i]]
        if probe_labels[i] == predicted_label:
            correct_predictions += 1
            
    rank1_accuracy = (correct_predictions / num_probes) * 100
    return rank1_accuracy

# 🔥 NEW: 확장된 평가 함수들

def calculate_detailed_biometric_metrics(genuine_scores, imposter_scores, system_info=None):
    """
    상세한 생체인식 메트릭 계산
    
    Args:
        genuine_scores: 정품 매칭 점수 배열
        imposter_scores: 위조 매칭 점수 배열  
        system_info: 시스템 정보 (COCONUT 관련)
        
    Returns:
        dict: 상세한 메트릭 결과
    """
    if len(genuine_scores) == 0 or len(imposter_scores) == 0:
        print("⚠️ Warning: Empty score arrays")
        return None
    
    # 라벨 생성
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    scores = np.concatenate([genuine_scores, imposter_scores])
    
    # ROC 커브 계산
    fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
    fnr = 1 - tpr  # False Negative Rate
    
    # EER 계산
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = interp1d(fpr, thresholds)(eer)
    except:
        # 보간 실패 시 가장 가까운 지점 찾기
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
    
    # AUC 계산
    auc_score = metrics.auc(fpr, tpr)
    
    # 점수 통계
    genuine_stats = {
        'mean': np.mean(genuine_scores),
        'std': np.std(genuine_scores),
        'min': np.min(genuine_scores),
        'max': np.max(genuine_scores),
        'count': len(genuine_scores)
    }
    
    imposter_stats = {
        'mean': np.mean(imposter_scores),
        'std': np.std(imposter_scores),
        'min': np.min(imposter_scores),
        'max': np.max(imposter_scores),
        'count': len(imposter_scores)
    }
    
    # d-prime 계산 (분리도 측정)
    d_prime = abs(genuine_stats['mean'] - imposter_stats['mean']) / \
              np.sqrt(0.5 * (genuine_stats['std']**2 + imposter_stats['std']**2))
    
    # 다양한 임계값에서의 성능
    threshold_analysis = []
    test_thresholds = np.linspace(scores.min(), scores.max(), 100)
    
    for thresh in test_thresholds:
        predictions = scores > thresh
        
        tp = np.sum((labels == 1) & predictions)
        tn = np.sum((labels == 0) & ~predictions)
        fp = np.sum((labels == 0) & predictions)
        fn = np.sum((labels == 1) & ~predictions)
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        threshold_analysis.append({
            'threshold': thresh,
            'far': far,
            'frr': frr,
            'accuracy': accuracy
        })
    
    results = {
        'eer': eer * 100,
        'eer_threshold': eer_threshold,
        'auc': auc_score,
        'd_prime': d_prime,
        'genuine_stats': genuine_stats,
        'imposter_stats': imposter_stats,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'thresholds': thresholds
        },
        'threshold_analysis': threshold_analysis,
        'raw_scores': {
            'genuine': genuine_scores,
            'imposter': imposter_scores
        }
    }
    
    # COCONUT 시스템 정보 추가
    if system_info:
        results['system_info'] = system_info
    
    return results

def save_biometric_evaluation_plots(results, output_dir="./results/evaluation", 
                                   title_prefix="COCONUT"):
    """
    생체인식 평가 그래프들을 저장
    
    Args:
        results: calculate_detailed_biometric_metrics의 결과
        output_dir: 저장 디렉토리
        title_prefix: 그래프 제목 접두사
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 스타일 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. ROC 커브
    plt.figure(figsize=(10, 8))
    plt.plot(results['roc_curve']['fpr'] * 100, 
             results['roc_curve']['tpr'] * 100, 
             'b-', linewidth=2, label=f'ROC Curve (AUC = {results["auc"]:.4f})')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random Classifier')
    
    # EER 포인트 표시
    eer_point_idx = np.argmin(np.abs(results['roc_curve']['fpr'] - results['roc_curve']['fnr']))
    plt.plot(results['roc_curve']['fpr'][eer_point_idx] * 100,
             results['roc_curve']['tpr'][eer_point_idx] * 100,
             'ro', markersize=8, label=f'EER = {results["eer"]:.3f}%')
    
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('Genuine Acceptance Rate (%)', fontsize=12)
    plt.title(f'{title_prefix} ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 10])  # 관심 영역에 집중
    plt.ylim([90, 100])
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. DET 커브 (Detection Error Tradeoff)
    plt.figure(figsize=(10, 8))
    plt.plot(results['roc_curve']['fpr'] * 100,
             results['roc_curve']['fnr'] * 100,
             'r-', linewidth=2, label='DET Curve')
    plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='EER Line')
    
    # EER 포인트 표시
    plt.plot(results['roc_curve']['fpr'][eer_point_idx] * 100,
             results['roc_curve']['fnr'][eer_point_idx] * 100,
             'go', markersize=8, label=f'EER = {results["eer"]:.3f}%')
    
    plt.xlabel('False Acceptance Rate (%)', fontsize=12)
    plt.ylabel('False Rejection Rate (%)', fontsize=12)
    plt.title(f'{title_prefix} DET Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.tight_layout()
    plt.savefig(output_path / 'det_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 점수 분포 히스토그램
    plt.figure(figsize=(12, 8))
    
    # 겹치는 히스토그램
    plt.hist(results['raw_scores']['genuine'], bins=50, alpha=0.7, 
             label=f'Genuine ({len(results["raw_scores"]["genuine"])} samples)', 
             color='blue', density=True)
    plt.hist(results['raw_scores']['imposter'], bins=50, alpha=0.7,
             label=f'Imposter ({len(results["raw_scores"]["imposter"])} samples)',
             color='red', density=True)
    
    # EER 임계값 표시
    plt.axvline(results['eer_threshold'], color='green', linestyle='--', linewidth=2,
                label=f'EER Threshold = {results["eer_threshold"]:.4f}')
    
    # 통계 정보 표시
    plt.axvline(results['genuine_stats']['mean'], color='blue', linestyle=':', alpha=0.8,
                label=f'Genuine Mean = {results["genuine_stats"]["mean"]:.4f}')
    plt.axvline(results['imposter_stats']['mean'], color='red', linestyle=':', alpha=0.8,
                label=f'Imposter Mean = {results["imposter_stats"]["mean"]:.4f}')
    
    plt.xlabel('Similarity Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'{title_prefix} Score Distribution (d\' = {results["d_prime"]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. FAR/FRR vs Threshold
    plt.figure(figsize=(12, 8))
    
    thresholds = [ta['threshold'] for ta in results['threshold_analysis']]
    fars = [ta['far'] * 100 for ta in results['threshold_analysis']]
    frrs = [ta['frr'] * 100 for ta in results['threshold_analysis']]
    
    plt.plot(thresholds, fars, 'r-', linewidth=2, label='FAR (%)')
    plt.plot(thresholds, frrs, 'b-', linewidth=2, label='FRR (%)')
    
    # EER 포인트 표시
    plt.axvline(results['eer_threshold'], color='green', linestyle='--', linewidth=2,
                label=f'EER Threshold = {results["eer_threshold"]:.4f}')
    plt.axhline(results['eer'], color='green', linestyle='--', linewidth=2,
                label=f'EER = {results["eer"]:.3f}%')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Error Rate (%)', fontsize=12)
    plt.title(f'{title_prefix} FAR/FRR vs Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_path / 'far_frr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 평가 그래프들이 저장되었습니다: {output_path}")

def generate_evaluation_report(results, output_dir="./results/evaluation", 
                              system_name="COCONUT"):
    """
    평가 결과 리포트 생성 (JSON + 텍스트)
    
    Args:
        results: calculate_detailed_biometric_metrics의 결과
        output_dir: 저장 디렉토리
        system_name: 시스템 이름
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON 리포트 (상세 데이터)
    report_data = {
        'system_name': system_name,
        'evaluation_date': datetime.now().isoformat(),
        'metrics': {
            'eer_percent': float(results['eer']),
            'eer_threshold': float(results['eer_threshold']),
            'auc': float(results['auc']),
            'd_prime': float(results['d_prime'])
        },
        'genuine_statistics': {k: float(v) for k, v in results['genuine_stats'].items()},
        'imposter_statistics': {k: float(v) for k, v in results['imposter_stats'].items()},
        'system_info': results.get('system_info', {})
    }
    
    # JSON 저장
    with open(output_path / 'evaluation_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # 텍스트 리포트 (요약)
    report_text = f"""
{'='*80}
{system_name} 생체인식 시스템 평가 리포트
{'='*80}
평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 핵심 성능 지표:
  • Equal Error Rate (EER): {results['eer']:.4f}%
  • Area Under Curve (AUC): {results['auc']:.6f}
  • d-prime (분리도): {results['d_prime']:.4f}
  • EER 임계값: {results['eer_threshold']:.6f}

📈 정품 매칭 통계:
  • 샘플 수: {results['genuine_stats']['count']:,}
  • 평균: {results['genuine_stats']['mean']:.6f}
  • 표준편차: {results['genuine_stats']['std']:.6f}
  • 범위: [{results['genuine_stats']['min']:.6f}, {results['genuine_stats']['max']:.6f}]

📉 위조 매칭 통계:
  • 샘플 수: {results['imposter_stats']['count']:,}
  • 평균: {results['imposter_stats']['mean']:.6f}
  • 표준편차: {results['imposter_stats']['std']:.6f}
  • 범위: [{results['imposter_stats']['min']:.6f}, {results['imposter_stats']['max']:.6f}]

🎯 성능 평가:
"""
    
    # 성능 등급 평가
    if results['eer'] < 0.1:
        grade = "Excellent (< 0.1%)"
    elif results['eer'] < 1.0:
        grade = "Very Good (< 1.0%)"
    elif results['eer'] < 5.0:
        grade = "Good (< 5.0%)"
    elif results['eer'] < 10.0:
        grade = "Fair (< 10.0%)"
    else:
        grade = "Poor (≥ 10.0%)"
    
    report_text += f"  • EER 등급: {grade}\n"
    
    if results['auc'] > 0.99:
        auc_grade = "Excellent (> 0.99)"
    elif results['auc'] > 0.95:
        auc_grade = "Very Good (> 0.95)"
    elif results['auc'] > 0.90:
        auc_grade = "Good (> 0.90)"
    else:
        auc_grade = "Needs Improvement (≤ 0.90)"
    
    report_text += f"  • AUC 등급: {auc_grade}\n"
    
    # 시스템 정보 추가
    if 'system_info' in results and results['system_info']:
        report_text += f"\n🔧 시스템 정보:\n"
        for key, value in results['system_info'].items():
            report_text += f"  • {key}: {value}\n"
    
    report_text += f"\n{'='*80}\n"
    report_text += "📁 생성된 파일들:\n"
    report_text += "  • roc_curve.png - ROC 커브\n"
    report_text += "  • det_curve.png - DET 커브\n"
    report_text += "  • score_distribution.png - 점수 분포\n"
    report_text += "  • far_frr_curve.png - FAR/FRR 커브\n"
    report_text += "  • evaluation_report.json - 상세 데이터\n"
    report_text += "  • evaluation_summary.txt - 이 리포트\n"
    report_text += f"{'='*80}\n"
    
    # 텍스트 저장
    with open(output_path / 'evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ 평가 리포트가 생성되었습니다: {output_path}")
    print(f"📊 EER: {results['eer']:.4f}%, AUC: {results['auc']:.6f}")
    
    return report_data

def perform_coconut_evaluation(model, train_loader, test_loader, device, 
                              save_plots=True, save_report=True,
                              output_dir="./results/coconut_evaluation"):
    """
    COCONUT 시스템 전용 종합 평가
    
    Args:
        model: COCONUT 모델 (headless mode)
        train_loader: 갤러리 데이터로더
        test_loader: 프로브 데이터로더  
        device: 연산 디바이스
        save_plots: 그래프 저장 여부
        save_report: 리포트 저장 여부
        output_dir: 결과 저장 디렉토리
        
    Returns:
        dict: 상세한 평가 결과
    """
    print("\n" + "="*80)
    print("🥥 COCONUT 시스템 종합 평가 시작")
    print("="*80)
    
    # 1. 기본 평가 (기존 함수 활용)
    basic_results = perform_evaluation(model, train_loader, test_loader, device)
    
    # 2. 특징 추출
    print("\n📊 특징 추출 중...")
    gallery_features, gallery_labels = extract_features(model, train_loader, device)
    probe_features, probe_labels = extract_features(model, test_loader, device)
    
    # 3. 유사도 계산 (cosine similarity 사용)
    print("🔄 유사도 계산 중...")
    similarities = np.dot(probe_features, gallery_features.T)
    
    # 4. Genuine/Imposter 점수 분리
    genuine_scores = []
    imposter_scores = []
    
    for i, probe_label in enumerate(probe_labels):
        for j, gallery_label in enumerate(gallery_labels):
            score = similarities[i, j]
            
            if probe_label == gallery_label:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
    
    # 5. 시스템 정보 수집
    system_info = {}
    if hasattr(model, 'get_model_info'):
        system_info = model.get_model_info()
    
    system_info.update({
        'probe_samples': len(probe_features),
        'gallery_samples': len(gallery_features),
        'genuine_comparisons': len(genuine_scores),
        'imposter_comparisons': len(imposter_scores),
        'feature_dimension': probe_features.shape[1],
        'evaluation_type': 'COCONUT_Headless'
    })
    
    # 6. 상세 메트릭 계산
    print("📈 상세 메트릭 계산 중...")
    detailed_results = calculate_detailed_biometric_metrics(
        np.array(genuine_scores), 
        np.array(imposter_scores),
        system_info
    )
    
    if detailed_results is None:
        print("❌ 메트릭 계산 실패")
        return basic_results
    
    # 7. 그래프 생성
    if save_plots:
        print("📊 평가 그래프 생성 중...")
        save_biometric_evaluation_plots(
            detailed_results, 
            output_dir=output_dir,
            title_prefix="COCONUT"
        )
    
    # 8. 리포트 생성
    if save_report:
        print("📄 평가 리포트 생성 중...")
        generate_evaluation_report(
            detailed_results,
            output_dir=output_dir,
            system_name="COCONUT"
        )
    
    # 9. 결과 통합
    final_results = {
        **basic_results,
        'detailed_metrics': detailed_results,
        'output_directory': output_dir
    }
    
    print("\n" + "="*80)
    print("🎉 COCONUT 평가 완료!")
    print(f"📊 EER: {detailed_results['eer']:.4f}%")
    print(f"📈 AUC: {detailed_results['auc']:.6f}")
    print(f"📁 결과 저장: {output_dir}")
    print("="*80)
    
    return final_results

# 기존 perform_evaluation 함수는 그대로 유지
def perform_evaluation(model, train_loader, test_loader, device):
    """모델의 전체 성능 평가를 수행하는 메인 함수 (기존 버전)"""
    print("\n--- Starting Full Evaluation ---")
    
    # 1. 특징 추출
    gallery_features, gallery_labels = extract_features(model, train_loader, device)
    probe_features, probe_labels = extract_features(model, test_loader, device)
    
    # 2. 매칭 점수 계산
    print("Calculating matching scores...")
    distances = calculate_scores(probe_features, gallery_features)
    
    # 3. Rank-1 정확도 계산
    rank1_acc = calculate_rank1(distances, probe_labels, gallery_labels)
    print(f"Rank-1 Accuracy: {rank1_acc:.3f}%")
    
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

    eer, threshold = calculate_eer(np.array(genuine_scores), np.array(imposter_scores))
    print(f"Equal Error Rate (EER): {eer:.4f}% at Threshold: {threshold:.4f}")
    
    results = {
        'rank1_accuracy': rank1_acc,
        'eer': eer,
        'threshold': threshold
    }
    return results