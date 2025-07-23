# evaluation/eval_utils.py - 평가 유틸리티 (기존과 동일)
"""
COCONUT Evaluation Utilities

FEATURES:
- Feature extraction and similarity computation
- Rank-1 accuracy calculation  
- EER (Equal Error Rate) calculation
- ROC curve analysis
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

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
    """정규 매칭 점수와 비정규 매칭 점수를 바탕으로 EER을 계산합니다."""
    labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    scores = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, thresholds = metrics.roc_curve(labels, -scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(-eer)

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

def perform_evaluation(model, train_loader, test_loader, device):
    """모델의 전체 성능 평가를 수행하는 메인 함수"""
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

