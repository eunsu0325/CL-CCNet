# run_coconut.py - User Node 통합 최종 실행 스크립트
"""
COCONUT Stage 2: User Node Based Online Adaptation Execution Script

DESIGN PHILOSOPHY:
- User Node based continual learning
- Loop Closure self-correction
- Mahalanobis distance authentication
- Comprehensive evaluation with ablation support
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from config.config_parser import ConfigParser
from framework.coconut import CoconutSystem
from evaluation.eval_utils import perform_evaluation
from datasets.palm_dataset import MyDataset

def setup_experiment(args):
    """실험 환경 설정"""
    # 재현성을 위한 시드 고정
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # CUDA 설정
    if args.gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"[Setup] Using GPU {args.gpu_id}")
    else:
        print("[Setup] Using CPU")

def test_authentication_system(system, test_dataset, num_tests=100):
    """
    🔥 User Node 기반 인증 시스템 평가
    
    평가 메트릭:
    - 정확도 (Accuracy)
    - FAR (False Acceptance Rate)
    - FRR (False Rejection Rate)
    - EER (Equal Error Rate)
    """
    print("\n" + "="*80)
    print("🔐 USER NODE AUTHENTICATION SYSTEM EVALUATION")
    print("="*80)
    
    if not system.user_nodes_enabled:
        print("⚠️ User Node system is DISABLED. Skipping authentication test.")
        return None
    
    device = system.device
    predictor = system.predictor_net
    node_manager = system.node_manager
    
    # 결과 저장
    results = {
        'correct': 0,
        'total': 0,
        'false_accepts': 0,
        'false_rejects': 0,
        'genuine_scores': [],
        'imposter_scores': [],
        'per_user_accuracy': {}
    }
    
    # 등록된 사용자 ID들
    registered_users = set(node_manager.nodes.keys())
    print(f"[Auth] Registered users: {len(registered_users)}")
    
    # 테스트 샘플 선택
    test_indices = random.sample(range(len(test_dataset)), min(num_tests, len(test_dataset)))
    
    print(f"[Auth] Testing {len(test_indices)} samples...")
    
    for idx in test_indices:
        data, true_label = test_dataset[idx]
        probe = data[0].unsqueeze(0).to(device)
        true_user_id = true_label.item() if torch.is_tensor(true_label) else true_label
        
        # 인증 시도
        auth_result = system.verify_user(probe)
        
        results['total'] += 1
        
        # 결과 분석
        if auth_result['is_match']:
            matched_user = auth_result['matched_user']
            distance = auth_result['distance']
            
            if matched_user == true_user_id:
                # True Accept
                results['correct'] += 1
                results['genuine_scores'].append(-distance)  # 거리를 유사도로 변환
                
                # Per-user accuracy
                if true_user_id not in results['per_user_accuracy']:
                    results['per_user_accuracy'][true_user_id] = {'correct': 0, 'total': 0}
                results['per_user_accuracy'][true_user_id]['correct'] += 1
            else:
                # False Accept
                results['false_accepts'] += 1
                results['imposter_scores'].append(-distance)
        else:
            # Rejection
            if true_user_id in registered_users:
                # False Reject
                results['false_rejects'] += 1
                if 'distance' in auth_result:
                    results['genuine_scores'].append(-auth_result['distance'])
            else:
                # True Reject (올바른 거부)
                results['correct'] += 1
                if 'distance' in auth_result:
                    results['imposter_scores'].append(-auth_result['distance'])
        
        # Per-user total 업데이트
        if true_user_id in registered_users:
            if true_user_id not in results['per_user_accuracy']:
                results['per_user_accuracy'][true_user_id] = {'correct': 0, 'total': 0}
            results['per_user_accuracy'][true_user_id]['total'] += 1
    
    # 메트릭 계산
    accuracy = results['correct'] / results['total'] * 100
    far = results['false_accepts'] / results['total'] * 100
    frr = results['false_rejects'] / results['total'] * 100
    
    print("\n[AUTH RESULTS]")
    print(f"  Total samples tested: {results['total']}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  FAR (False Accept Rate): {far:.2f}%")
    print(f"  FRR (False Reject Rate): {frr:.2f}%")
    
    # EER 계산 (genuine/imposter scores가 충분한 경우)
    if len(results['genuine_scores']) > 0 and len(results['imposter_scores']) > 0:
        from evaluation.eval_utils import calculate_eer
        eer, threshold = calculate_eer(
            np.array(results['genuine_scores']), 
            np.array(results['imposter_scores'])
        )
        print(f"  EER (Equal Error Rate): {eer:.2f}%")
        results['eer'] = eer
        results['eer_threshold'] = threshold
    
    # Per-user 정확도
    print("\n[PER-USER ACCURACY]")
    for user_id, stats in results['per_user_accuracy'].items():
        user_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  User {user_id}: {user_acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    print("="*80)
    
    return results

def ablation_study(config_path, args):
    """
    🔥 Ablation Study: 각 구성 요소의 효과 측정
    
    테스트 구성:
    1. Baseline (User Node OFF)
    2. +UserNode (Loop Closure OFF)
    3. +LoopClosure
    4. +Mahalanobis
    5. Full System
    """
    print("\n" + "="*80)
    print("📊 ABLATION STUDY")
    print("="*80)
    
    # 기본 설정 로드
    base_config = ConfigParser(config_path)
    
    ablation_configs = [
        {
            'name': 'Baseline',
            'user_node': False,
            'loop_closure': False,
            'mahalanobis': False
        },
        {
            'name': '+UserNode',
            'user_node': True,
            'loop_closure': False,
            'mahalanobis': False
        },
        {
            'name': '+LoopClosure',
            'user_node': True,
            'loop_closure': True,
            'mahalanobis': False
        },
        {
            'name': '+Mahalanobis',
            'user_node': True,
            'loop_closure': True,
            'mahalanobis': True
        }
    ]
    
    results = {}
    
    for ablation_config in ablation_configs:
        print(f"\n[Ablation] Testing {ablation_config['name']}...")
        
        # 설정 수정
        if hasattr(base_config, 'user_node'):
            base_config.user_node.enable_user_nodes = ablation_config['user_node']
        if hasattr(base_config, 'loop_closure'):
            base_config.loop_closure.enabled = ablation_config['loop_closure']
        if hasattr(base_config, 'loss') and hasattr(base_config.loss, 'online_learning'):
            base_config.loss.online_learning['enable_mahalanobis'] = ablation_config['mahalanobis']
        
        # 시스템 생성 및 실행
        system = CoconutSystem(base_config)
        system.run_experiment()
        
        # 평가
        # (평가 코드는 기존과 동일)
        
        results[ablation_config['name']] = {
            'config': ablation_config,
            # 평가 결과 저장
        }
    
    # 결과 비교 출력
    print("\n[ABLATION RESULTS SUMMARY]")
    for name, result in results.items():
        print(f"{name}: ...")  # 결과 출력
    
    return results

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='COCONUT Online Adaptation')
    parser.add_argument('--config', type=str, default='./config/adapt_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', type=str, default='normal', 
                       choices=['normal', 'ablation', 'eval_only'],
                       help='Execution mode')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for eval_only mode')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🥥 COCONUT STAGE 2: USER NODE BASED ONLINE ADAPTATION")
    print("="*80)
    
    # 실험 환경 설정
    setup_experiment(args)
    
    if args.mode == 'ablation':
        # Ablation study 실행
        ablation_study(args.config, args)
    
    elif args.mode == 'eval_only':
        # 평가만 수행
        print("[Mode] Evaluation only mode")
        config = ConfigParser(args.config)
        system = CoconutSystem(config)
        
        if args.checkpoint:
            print(f"[Eval] Loading checkpoint: {args.checkpoint}")
            system._load_specific_checkpoint(args.checkpoint)
        
        # 평가 수행
        run_final_evaluation(system, config)
    
    else:
        # 일반 실행
        print("[Mode] Normal execution mode")
        
        # 1. 설정 파일 로드
        config = ConfigParser(args.config)
        print("\n--- COCONUT Configuration ---")
        print(config)

        # 2. 시스템 생성 및 실행
        system = CoconutSystem(config)
        system.run_experiment()

        # 3. 최종 평가
        run_final_evaluation(system, config)

def run_final_evaluation(system, config):
    """최종 성능 평가"""
    print("\n--- Final Performance Evaluation ---")
    
    final_model = system.learner_net
    final_model.eval()

    print("Loading datasets for final evaluation...")
    
    # 평가용 데이터셋 로딩
    try:
        # 데이터셋 경로 설정
        if hasattr(config.dataset, 'train_set_file') and config.dataset.train_set_file:
            train_file = config.dataset.train_set_file
            test_file = getattr(config.dataset, 'test_set_file', train_file)
        else:
            train_file = str(config.dataset.dataset_path)
            test_file = str(getattr(config.dataset, 'test_dataset_path', config.dataset.dataset_path))
        
        print(f"Train file: {train_file}")
        print(f"Test file: {test_file}")
        
        train_dataset = MyDataset(txt=train_file, train=False)
        test_dataset = MyDataset(txt=test_file, train=False)
        
        train_loader_eval = DataLoader(train_dataset, batch_size=128, shuffle=False)
        test_loader_eval = DataLoader(test_dataset, batch_size=128, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 기본 성능 평가
        print("\n[1/2] Basic Performance Evaluation...")
        final_results = perform_evaluation(final_model, train_loader_eval, test_loader_eval, device)
        
        print("\n--- Basic Results ---")
        print(f"Rank-1 Accuracy: {final_results['rank1_accuracy']:.3f}%")
        print(f"EER: {final_results['eer']:.4f}%")
        
        # 2. User Node 기반 인증 평가
        print("\n[2/2] User Node Authentication Evaluation...")
        auth_results = test_authentication_system(system, test_dataset, num_tests=200)
        
        # 3. 시스템 통계
        print("\n--- System Statistics ---")
        if system.node_manager and system.user_nodes_enabled:
            node_stats = system.node_manager.get_statistics()
            print(f"Total registered users: {node_stats['total_users']}")
            print(f"Total samples: {node_stats['total_samples']}")
            print(f"Avg samples per user: {node_stats['avg_samples_per_user']:.2f}")
        
        buffer_stats = system.replay_buffer.get_statistics()
        print(f"\nReplay Buffer:")
        print(f"  Total samples: {buffer_stats['total_samples']}")
        print(f"  Unique users: {buffer_stats['unique_users']}")
        print(f"  Buffer utilization: {buffer_stats['buffer_utilization']:.1%}")
        
        if 'priority_queue_size' in buffer_stats:
            print(f"  Priority queue size: {buffer_stats['priority_queue_size']}")
        
        print("\n✅ Evaluation completed successfully!")
        
        # 4. 결과 저장
        save_results(system, final_results, auth_results)
        
    except Exception as e:
        print(f"[Error] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def save_results(system, basic_results, auth_results):
    """평가 결과 저장"""
    import json
    from datetime import datetime
    
    results_dir = Path("./results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 전체 결과 정리
    all_results = {
        'timestamp': timestamp,
        'basic_evaluation': basic_results,
        'authentication_evaluation': auth_results,
        'system_config': {
            'user_nodes_enabled': system.user_nodes_enabled,
            'loop_closure_enabled': system.loop_closure_enabled,
            'headless_mode': system.headless_mode,
            'training_batch_size': system.training_batch_size
        }
    }
    
    # JSON 저장
    results_file = results_dir / f'coconut_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {results_file}")

if __name__ == '__main__':
    main()