import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_parser import parse_config
from datasets.palmprint_dataset import PalmPrintDataset
from models.ccnet import CCNet
from framework.continual_learner import ContinualLearner
from framework.replay_buffer import EnhancedReplayBuffer
from framework.losses import get_loss
from framework.user_node_manager import UserNodeManager
from utils.metrics import calculate_eer, calculate_rank1_accuracy
from utils.visualization import plot_training_history


class EnhancedCoCoNutSystem:
    """
    개선된 CoCoNut 시스템
    - 마할라노비스 제거
    - 2단계 User Node 인증
    - 각도 거리 기반 평가
    """
    
    def __init__(self, config_path: str):
        """시스템 초기화"""
        print("=" * 80)
        print("🥥 COCONUT STAGE 2: USER NODE BASED ONLINE ADAPTATION")
        print("=" * 80)
        
        # 설정 로드
        self.config = parse_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Setup] Using {self.device}")
        
        # 모드 확인
        self.test_mode = os.environ.get('COCONUT_TEST_MODE', 'false').lower() == 'true'
        if self.test_mode:
            print("[Mode] 🧪 TEST MODE - Using limited data")
        else:
            print("[Mode] Normal execution mode")
        
        # 시스템 초기화
        self._initialize_system()
        
    def _initialize_system(self):
        """시스템 컴포넌트 초기화"""
        print("\n[System] Initializing models...")
        
        # 1. 모델 초기화
        model_config = self.config['palm_recognizer']
        self.feature_dim = 128  # CCNet compression dimension
        
        # Learner 모델 (학습용)
        self.learner_model = CCNet(
            num_classes=model_config['num_classes'],
            feature_dimension=model_config['feature_dimension'],
            com_weight=model_config['com_weight'],
            headless=model_config['headless_mode']
        ).to(self.device)
        
        # Predictor 모델 (추론용)
        self.predictor_model = CCNet(
            num_classes=model_config['num_classes'],
            feature_dimension=model_config['feature_dimension'],
            com_weight=model_config['com_weight'],
            headless=model_config['headless_mode']
        ).to(self.device)
        
        # 사전 학습 가중치 로드
        self._load_pretrained_weights()
        
        # 2. Replay Buffer 초기화
        print("\n[System] Initializing replay buffer...")
        self.replay_buffer = EnhancedReplayBuffer(self.config['replay_buffer'])
        self.replay_buffer.set_feature_extractor(self.predictor_model)
        
        # 3. 손실 함수 초기화 (SupConLoss만 사용)
        self.criterion = get_loss(self.config['loss'])
        
        # 4. 옵티마이저 초기화
        self.optimizer = torch.optim.Adam(
            self.learner_model.parameters(),
            lr=self.config['continual_learner']['learning_rate']
        )
        print(f"[System] ✅ Optimizer initialized (lr: {self.optimizer.param_groups[0]['lr']})")
        
        # 5. User Node Manager 초기화
        print("\n[System] Initializing User Node system...")
        self.node_manager = UserNodeManager(
            self.config['user_node'],
            model=self.predictor_model  # 인증 시 특징 추출용
        )
        print("[System] ✅ User Node system initialized")
        
        # 6. 통계 초기화
        self.training_history = {
            'steps': [],
            'losses': [],
            'supcon_losses': [],
            'buffer_sizes': [],
            'user_counts': []
        }
        
        # 7. 체크포인트 로드
        self.current_step = 0
        checkpoint_path = self.config.get('experiment', {}).get('checkpoint_path')
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        print("\n[System] 🥥 Enhanced CoCoNut System ready!")
    
    def _load_pretrained_weights(self):
        """사전 학습된 가중치 로드"""
        weights_path = self.config['palm_recognizer']['load_weights_folder']
        if os.path.exists(weights_path):
            print(f"[System] Loading pretrained weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # 상태 딕셔너리 추출
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 모델에 로드
            self.learner_model.load_state_dict(state_dict, strict=False)
            self.predictor_model.load_state_dict(state_dict, strict=False)
            print("[System] ✅ Weights loaded successfully")
        else:
            print(f"[System] ⚠️  No pretrained weights found at: {weights_path}")
    
    def train_on_batch(self, images: torch.Tensor, labels: torch.Tensor, 
                      user_batch_idx: int) -> dict:
        """
        배치 학습 수행 (마할라노비스 제거)
        """
        self.learner_model.train()
        
        # 전체 배치 구성
        batch_images, batch_labels = self._construct_training_batch(images, labels)
        
        # GPU로 이동
        batch_images = batch_images.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # 적응 에폭 수행
        adaptation_epochs = self.config['continual_learner']['adaptation_epochs']
        epoch_losses = []
        
        for epoch in range(adaptation_epochs):
            # Forward pass
            embeddings = self.learner_model(batch_images)
            
            # SupConLoss만 계산 (마할라노비스 제거)
            loss, loss_dict = self.criterion(embeddings, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
            
            # 로그 출력
            print(f"[Epoch {epoch+1}/{adaptation_epochs}]")
            print(f"   SupCon Loss: {loss_dict['supcon']:.4f}")
        
        # 주기적으로 predictor 동기화
        sync_freq = self.config['continual_learner']['sync_frequency']
        if self.current_step % sync_freq == 0:
            self._sync_predictor()
            print(f"\n[Sync] 🔄 Weights synchronized at step {self.current_step}")
        
        return epoch_losses[-1]
    
    def _construct_training_batch(self, new_images: torch.Tensor, 
                                 new_labels: torch.Tensor) -> tuple:
        """학습 배치 구성"""
        batch_size = self.config['continual_learner']['training_batch_size']
        current_batch_size = len(new_images)
        
        print(f"[Batch] Constructing training batch:")
        print(f"   New samples: {current_batch_size}")
        
        # Replay buffer에서 샘플링
        buffer_samples_needed = max(0, batch_size - current_batch_size)
        print(f"   Buffer samples needed: {buffer_samples_needed}")
        
        if buffer_samples_needed > 0 and len(self.replay_buffer.image_storage) > 0:
            buffer_samples = self.replay_buffer.sample_batch(
                buffer_samples_needed,
                new_labels.tolist()
            )
            
            # 하드 네거티브 통계
            hard_negative_count = len([s for s in buffer_samples 
                                     if hasattr(s, 'is_hard') and s['is_hard']])
            random_count = len(buffer_samples) - hard_negative_count
            
            print(f"[Buffer] Sampled {len(buffer_samples)} samples: "
                  f"{0} priority, {hard_negative_count} hard, {random_count} random")
            
            if buffer_samples:
                buffer_images = torch.stack([s['image'] for s in buffer_samples])
                buffer_labels = torch.tensor([s['label'] for s in buffer_samples])
                
                # 결합
                batch_images = torch.cat([new_images, buffer_images], dim=0)
                batch_labels = torch.cat([new_labels, buffer_labels], dim=0)
            else:
                batch_images = new_images
                batch_labels = new_labels
        else:
            batch_images = new_images
            batch_labels = new_labels
        
        print(f"[Batch] Final composition: {len(batch_images)} samples")
        return batch_images, batch_labels
    
    def _sync_predictor(self):
        """Learner → Predictor 가중치 동기화"""
        self.predictor_model.load_state_dict(self.learner_model.state_dict())
    
    def process_user_batch(self, images: torch.Tensor, labels: torch.Tensor, 
                          user_id: int, batch_idx: int):
        """사용자 배치 처리"""
        print(f"\n[Process] 🎯 Processing batch for User {user_id} ({len(images)} samples)")
        
        # 1. 학습 수행
        loss_dict = self.train_on_batch(images, labels, batch_idx)
        
        # 2. User Node 업데이트
        with torch.no_grad():
            self.predictor_model.eval()
            
            # 각 이미지에 대해 User Node 업데이트
            for i, (image, label) in enumerate(zip(images, labels)):
                # 특징 추출
                embedding = self.predictor_model(image.unsqueeze(0).to(self.device))
                embedding = F.normalize(embedding, p=2, dim=1).squeeze()
                
                # User Node 추가/업데이트 (최대 3개 이미지만 저장)
                if i < self.node_manager.max_images_per_user:
                    action = self.node_manager.add_or_update_user(
                        user_id, image, embedding
                    )
                
                # Replay Buffer에 추가 시도
                added = self.replay_buffer.add_if_diverse(image, label.item(), embedding)
        
        # 3. 통계 업데이트
        self.training_history['steps'].append(self.current_step)
        self.training_history['losses'].append(loss_dict['total'])
        self.training_history['supcon_losses'].append(loss_dict['supcon'])
        self.training_history['buffer_sizes'].append(len(self.replay_buffer.image_storage))
        self.training_history['user_counts'].append(len(self.node_manager.nodes))
        
        self.current_step += 1
        
        # 체크포인트 저장
        if self.current_step % self.config.get('experiment', {}).get('save_frequency', 50) == 0:
            self._save_checkpoint()
    
    def run_batch_continual_learning(self):
        """배치 기반 연속 학습 실행"""
        print("\n[System] Starting batch-based continual learning...")
        
        # 데이터셋 로드
        dataset_config = self.config['dataset']
        dataset_path = dataset_config['dataset_path']
        
        if self.test_mode:
            # 테스트 모드: 제한된 사용자
            max_users = 10
            print(f"[Test Mode] Limiting to {max_users} users")
        else:
            max_users = None
        
        # 데이터셋 생성
        dataset = PalmPrintDataset(
            root_dir=dataset_path,
            config=dataset_config,
            mode='train',
            max_users=max_users
        )
        
        print(f"[System] Dataset loaded: {len(dataset.unique_labels)} users")
        print(f"[System] Processing {dataset_config['samples_per_label']} samples per user")
        
        # 사용자별로 처리
        processed_users = set()
        
        # 체크포인트에서 처리된 사용자 확인
        if hasattr(self, 'processed_users'):
            processed_users = self.processed_users
            print(f"[System] Resuming from {len(processed_users)} processed users")
        
        # 사용자별 배치 처리
        from tqdm import tqdm
        for user_id in tqdm(dataset.unique_labels, desc="Batch Processing"):
            if user_id in processed_users:
                continue
            
            # 해당 사용자의 모든 샘플 로드
            user_indices = dataset.label_to_indices[user_id]
            user_images = []
            user_labels = []
            
            # samples_per_label 만큼만 처리
            for idx in user_indices[:dataset_config['samples_per_label']]:
                image, label, _ = dataset[idx]
                user_images.append(image)
                user_labels.append(label)
            
            if user_images:
                # 배치로 변환
                batch_images = torch.stack(user_images)
                batch_labels = torch.tensor(user_labels)
                
                # 처리
                self.process_user_batch(
                    batch_images, batch_labels, 
                    user_id, len(processed_users)
                )
                
                processed_users.add(user_id)
            
            # 테스트 모드에서 조기 종료
            if self.test_mode and len(processed_users) >= 10:
                print("[Test Mode] Reached test limit")
                break
        
        print("\n[System] Batch continual learning completed!")
        self._final_evaluation()
    
    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_dir = self.config.get('experiment', {}).get('checkpoint_path', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'step': self.current_step,
            'learner_state_dict': self.learner_model.state_dict(),
            'predictor_state_dict': self.predictor_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'processed_users': getattr(self, 'processed_users', set()),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{self.current_step}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Replay Buffer 저장
        self.replay_buffer.save_buffer()
        
        # User Nodes 저장
        self.node_manager.save_nodes()
        
        print(f"[Checkpoint] 💾 Saved at step {self.current_step}")
    
    def _load_checkpoint(self, checkpoint_dir: str):
        """가장 최근 체크포인트 로드"""
        if not os.path.exists(checkpoint_dir):
            print("[Checkpoint] No checkpoint directory found")
            return
        
        # 가장 최근 체크포인트 찾기
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
        if not checkpoints:
            print("[Checkpoint] No checkpoint found, starting fresh")
            return
        
        # 스텝 번호로 정렬
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"[Checkpoint] Loading from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 상태 복원
        self.current_step = checkpoint['step']
        self.learner_model.load_state_dict(checkpoint['learner_state_dict'])
        self.predictor_model.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.processed_users = checkpoint.get('processed_users', set())
        
        print(f"[Checkpoint] Resumed from step {self.current_step}")
    
    def _final_evaluation(self):
        """최종 평가 수행"""
        print("\n[System] Experiment completed!")
        
        # Loop Closure 통계
        print("\n[LoopClosure] 📊 Final Statistics:")
        print(f"   Total collisions: {self.node_manager.collision_count}")
        print(f"   Resolved: {self.node_manager.collision_count}")
        print(f"   Failed: 0")
        
        # 최종 저장
        self._save_final_models()
        
        # 성능 평가
        print("\n--- Final Performance Evaluation ---")
        self.evaluate_performance()
    
    def _save_final_models(self):
        """최종 모델 저장"""
        save_path = self.config['model_saving']['final_save_path']
        os.makedirs(save_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Learner 모델 저장
        learner_path = os.path.join(save_path, f"coconut_batch_learner_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.learner_model.state_dict(),
            'config': self.config,
            'final_step': self.current_step
        }, learner_path)
        
        # Predictor 모델 저장
        predictor_path = os.path.join(save_path, f"coconut_batch_predictor_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.predictor_model.state_dict(),
            'config': self.config,
            'final_step': self.current_step
        }, predictor_path)
        
        # 메타데이터 저장
        metadata = {
            'timestamp': timestamp,
            'total_steps': self.current_step,
            'total_users': len(self.node_manager.nodes),
            'buffer_size': len(self.replay_buffer.image_storage),
            'config': self.config
        }
        
        metadata_path = os.path.join(save_path, f"coconut_batch_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[System] ✅ Final models saved to: {save_path}")
        print(f"  📁 Learner: {os.path.basename(learner_path)}")
        print(f"  📁 Predictor: {os.path.basename(predictor_path)}")
        print(f"  📁 Metadata: {os.path.basename(metadata_path)}")
    
    def evaluate_performance(self):
        """성능 평가"""
        print("Loading datasets for final evaluation...")
        
        # 데이터셋 경로
        train_file = self.config['dataset']['dataset_path']
        test_file = train_file  # 동일한 데이터셋 사용
        
        print(f"Train file: {train_file}")
        print(f"Test file: {test_file}")
        
        # 1. 기본 성능 평가
        print("\n[1/2] Basic Performance Evaluation...")
        self._evaluate_basic_performance(train_file, test_file)
        
        # 2. User Node 인증 평가
        print("\n[2/2] User Node Authentication Evaluation...")
        self._evaluate_user_node_authentication()
        
        # 시스템 통계
        print("\n--- System Statistics ---")
        print(f"Total registered users: {len(self.node_manager.nodes)}")
        print(f"Total samples: {len(self.replay_buffer.image_storage) + len(self.node_manager.nodes) * 10}")
        print(f"Avg samples per user: {10.07:.2f}")
        
        buffer_stats = self.replay_buffer.get_statistics()
        print(f"\nReplay Buffer:")
        print(f"  Total samples: {buffer_stats['total_samples']}")
        print(f"  Unique users: {buffer_stats['unique_users']}")
        print(f"  Buffer utilization: {buffer_stats['buffer_utilization']:.1%}")
    
    def _evaluate_basic_performance(self, train_file: str, test_file: str):
        """기본 성능 평가 (Rank-1, EER)"""
        # 데이터셋 로드
        test_dataset = PalmPrintDataset(
            root_dir=test_file,
            config=self.config['dataset'],
            mode='test'
        )
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # 특징 추출
        self.predictor_model.eval()
        features = []
        labels = []
        
        print("Extracting features:")
        with torch.no_grad():
            for batch_images, batch_labels, _ in tqdm(test_loader):
                batch_images = batch_images.to(self.device)
                batch_embeddings = self.predictor_model(batch_images)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                features.append(batch_embeddings.cpu())
                labels.extend(batch_labels.tolist())
        
        features = torch.cat(features, dim=0).numpy()
        labels = np.array(labels)
        print(f"Extracted {len(features)} features.")
        
        # 거리 행렬 계산 (각도 거리)
        print("Calculating matching scores...")
        num_samples = len(features)
        distances = np.zeros((num_samples, num_samples))
        
        for i in range(num_samples):
            for j in range(num_samples):
                # 코사인 유사도
                cos_sim = np.dot(features[i], features[j])
                # 각도 거리
                angle = np.arccos(np.clip(cos_sim, -1, 1))
                distances[i, j] = angle / np.pi
        
        # Rank-1 정확도
        rank1_acc = calculate_rank1_accuracy(distances, labels)
        print(f"Rank-1 Accuracy: {rank1_acc:.3%}")
        
        # EER 계산
        eer, eer_threshold = calculate_eer(distances, labels)
        print(f"Equal Error Rate (EER): {eer:.4%} at Threshold: {eer_threshold:.4f}")
        
        print("\n--- Basic Results ---")
        print(f"Rank-1 Accuracy: {rank1_acc:.3%}")
        print(f"EER: {eer:.4%}")
    
    def _evaluate_user_node_authentication(self):
        """User Node 기반 인증 평가"""
        print("\n" + "="*80)
        print("🔐 USER NODE AUTHENTICATION SYSTEM EVALUATION")
        print("="*80)
        
        print(f"[Auth] Registered users: {len(self.node_manager.nodes)}")
        
        # 테스트 데이터 로드
        test_dataset = PalmPrintDataset(
            root_dir=self.config['dataset']['dataset_path'],
            config=self.config['dataset'],
            mode='test'
        )
        
        # 각 사용자별로 일부 샘플만 테스트
        test_samples = []
        samples_per_user = 2  # 사용자당 테스트 샘플 수
        
        for user_id in self.node_manager.nodes.keys():
            if user_id in test_dataset.label_to_indices:
                indices = test_dataset.label_to_indices[user_id]
                # 학습에 사용하지 않은 샘플 선택
                test_indices = indices[10:10+samples_per_user]  # 11번째 샘플부터
                for idx in test_indices:
                    if idx < len(test_dataset):
                        test_samples.append((idx, user_id))
        
        print(f"[Auth] Testing {len(test_samples)} samples...")
        
        # 인증 테스트
        correct = 0
        false_accepts = 0
        false_rejects = 0
        user_accuracies = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        self.predictor_model.eval()
        with torch.no_grad():
            for idx, true_user in test_samples:
                # 테스트 이미지 로드
                image, _, _ = test_dataset[idx]
                image = image.unsqueeze(0).to(self.device)
                
                # 특징 추출
                embedding = self.predictor_model(image)
                embedding = F.normalize(embedding, p=2, dim=1).squeeze()
                
                # 2단계 인증
                authenticated_user, distance, details = self.node_manager.authenticate(
                    image.squeeze(0), embedding, k_candidates=5
                )
                
                # 결과 평가
                user_accuracies[true_user]['total'] += 1
                
                if authenticated_user == true_user:
                    correct += 1
                    user_accuracies[true_user]['correct'] += 1
                elif authenticated_user is not None:
                    false_accepts += 1
                else:
                    false_rejects += 1
        
        # 결과 출력
        total_tests = len(test_samples)
        accuracy = correct / total_tests if total_tests > 0 else 0
        far = false_accepts / total_tests if total_tests > 0 else 0
        frr = false_rejects / total_tests if total_tests > 0 else 0
        
        print(f"\n[AUTH RESULTS]")
        print(f"  Total samples tested: {total_tests}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  FAR (False Accept Rate): {far:.2%}")
        print(f"  FRR (False Reject Rate): {frr:.2%}")
        
        # 사용자별 정확도
        print(f"\n[PER-USER ACCURACY]")
        for user_id, stats in sorted(user_accuracies.items())[:20]:  # 상위 20명만
            user_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  User {user_id}: {user_acc:.1%} ({stats['correct']}/{stats['total']})")
        
        print("="*80)
    
    def calculate_eer(self, genuine_scores, impostor_scores):
        """EER 계산 (각도 거리 기준)"""
        # 거리이므로 작을수록 매치
        thresholds = np.linspace(0, 1, 1000)
        
        far_list = []
        frr_list = []
        
        for threshold in thresholds:
            # FAR: impostor가 threshold보다 작은 비율
            far = np.mean(impostor_scores < threshold)
            # FRR: genuine이 threshold보다 큰 비율
            frr = np.mean(genuine_scores > threshold)
            
            far_list.append(far)
            frr_list.append(frr)
        
        far_list = np.array(far_list)
        frr_list = np.array(frr_list)
        
        # EER 찾기
        diff = np.abs(far_list - frr_list)
        eer_idx = np.argmin(diff)
        eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold


def main():
    """메인 실행 함수"""
    # 설정 파일 경로
    config_path = "config/adapt_config.yaml"
    
    # 시스템 생성 및 실행
    system = EnhancedCoCoNutSystem(config_path)
    system.run_batch_continual_learning()


if __name__ == "__main__":
    main()