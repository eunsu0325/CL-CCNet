'''

=== COCONUT STAGE 1: PRETRAIN PHASE ===

DESIGN RATIONALE:
1. Hybrid Loss Strategy:
   - ArcFace (α=0.8): Creates large inter-class margins for robust separation
   - SupCon (β=0.2): Enhances intra-class cohesion and feature structuring
   
2. Why NOT W2ML in Pretrain:
   - Prevents overfitting to dataset-specific noise and artifacts
   - Ensures generalization capability to unseen environments
   - Builds stable foundation before adaptive learning
   
3. Mathematical Foundation:
   L_pretrain = α * L_ArcFace + β * L_SupCon
   where α + β = 1.0 for balanced optimization 
   
'''

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.ccnet_model import ccnet  # 수정: modles → models
from datasets.palm_dataset import MyDataset
from framework.losses import SupConLoss
from evaluation.eval_utils import perform_evaluation

class CCNetTrainer:
    def __init__(self, config):
        """
        🔥 하이브리드 손실 함수를 적용한 사전 훈련 시스템
        
        DESIGN PHILOSOPHY:
        - Stage 1 goal: Build robust, generalizable feature space
        - Method: Proven hybrid loss without premature specialization
        - Rationale: Stable foundation before adaptive learning
        """
        self.config = config
        print("="*80)
        print("🥥 COCONUT STAGE 1: ROBUST FEATURE SPACE CONSTRUCTION")
        print("="*80)
        print("📋 DESIGN STRATEGY:")
        print("   - Loss Function: Hybrid (ArcFace + SupCon)")
        print("   - ArcFace Weight: 0.8 (Inter-class separation)")
        print("   - SupCon Weight: 0.2 (Intra-class cohesion)")
        print("   - W2ML Status: NOT applied (prevents overfitting)")
        print("   - Goal: Generalizable feature space")
        print("="*80)

        # 설정값 객체 할당
        self.cfg_dataset = self.config.dataset
        self.cfg_model = self.config.palm_recognizer
        self.cfg_training = self.config.training
        self.cfg_loss = self.config.loss
        self.cfg_paths = self.config.paths

        self.device = torch.device(f"cuda:{self.cfg_training.gpu_id}" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg_training.gpu_id)
        print(f"[Trainer] Using device: {self.device}")

        # 결과 저장 경로 생성
        self.checkpoint_path = Path(self.cfg_paths.checkpoint_path)
        self.results_path = Path(self.cfg_paths.results_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

        # 초기화
        self._initialize_dataloaders()
        self._initialize_model()
        self._initialize_hybrid_loss_with_rationale()
        self._initialize_optimizer()

        # 로깅 변수
        self.train_losses, self.train_accuracy = [], []
        self.val_losses, self.val_accuracy = [], []
        self.best_acc = 0.0
        
        # 🔥 손실 분해 로깅
        self.loss_breakdown = {
            'arcface_losses': [],
            'supcon_losses': [], 
            'total_losses': []
        }

    def _initialize_dataloaders(self):
        """데이터셋과 데이터 로더를 생성합니다."""
        print("[Trainer] Loading datasets...")
        trainset = MyDataset(txt=self.cfg_dataset.train_set_file, train=True)
        valset = MyDataset(txt=self.cfg_dataset.test_set_file, train=False) 
        
        self.train_loader = DataLoader(
            dataset=trainset, 
            batch_size=self.cfg_training.batch_size, 
            num_workers=2, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=valset, 
            batch_size=self.cfg_training.batch_size,
            num_workers=2, 
            shuffle=False
        )
        print(f"[Trainer] Train samples: {len(trainset)}, Validation samples: {len(valset)}")

    def _initialize_model(self):
        """모델을 초기화합니다."""
        print('------[Trainer] Init Model------')
        self.model = ccnet(
            num_classes=self.cfg_model.num_classes,
            weight=self.cfg_model.com_weight
        ).to(self.device)
        self.best_model = copy.deepcopy(self.model)

    def _initialize_hybrid_loss_with_rationale(self):
        """
        🔥 하이브리드 손실 함수 초기화 - 설계 근거와 함께
        
        Mathematical Formulation:
        L_total = α * L_ArcFace + β * L_SupCon
        
        Where:
        - L_ArcFace: Angular margin loss for inter-class separation
        - L_SupCon: Supervised contrastive loss for intra-class cohesion
        - α = 0.8: Emphasizes classification accuracy and margin
        - β = 0.2: Provides feature space structuring
        
        Rationale:
        - 0.8:0.2 ratio empirically validated in CCNet paper
        - Ensures both strong classification and good feature geometry
        - Stable learning without W2ML premature specialization
        """
        print("[Trainer] 🎯 Initializing Hybrid Loss Functions...")
        
        # ArcFace 손실 (분류 + 마진 최대화)
        self.arcface_criterion = nn.CrossEntropyLoss()
        print("   ✓ ArcFace Loss: Inter-class margin maximization")
        
        # SupCon 손실 (특징 공간 구조화)  
        self.supcon_criterion = SupConLoss(temperature=self.cfg_loss.temp)
        print("   ✓ SupCon Loss: Intra-class cohesion enhancement")
        
        # 가중치 설정 (CCNet 검증된 비율)
        self.alpha = self.cfg_loss.weight1  # 0.8 (ArcFace)
        self.beta = self.cfg_loss.weight2   # 0.2 (SupCon)
        
        print(f"   ✓ Loss Weights: α={self.alpha} (ArcFace), β={self.beta} (SupCon)")
        print("   📖 Rationale: 0.8:0.2 ratio ensures both classification accuracy")
        print("      and feature space structuring (validated in CCNet)")
        print("   🚫 W2ML NOT used: Prevents overfitting to dataset-specific noise")
        
        # 가중치 합이 1.0인지 검증
        total_weight = self.alpha + self.beta
        assert abs(total_weight - 1.0) < 1e-6, f"Loss weights must sum to 1.0, got {total_weight}"
        print("   ✓ Weight validation passed")

    def _initialize_optimizer(self):
        """옵티마이저와 스케줄러를 정의합니다."""
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg_training.lr)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.cfg_training.redstep, 
            gamma=0.8
        )

    def train(self):
        """
        🔥 하이브리드 손실을 적용한 사전 훈련 메인 루프
        """
        print("\n[Trainer] Starting Hybrid Loss Pre-training...")
        print("="*60)
        print("STAGE 1: BUILDING ROBUST FEATURE SPACE")
        print("Strategy: ArcFace + SupCon Hybrid Loss")
        print("="*60)
        
        for epoch in range(self.cfg_training.epoch_num):
            start_time = time.time()
            
            # 훈련 및 검증 실행
            self.fit(epoch, phase='training')
            val_epoch_loss, val_epoch_accuracy = self.fit(epoch, phase='validation')
            
            self.scheduler.step()
            
            # 최고 모델 저장
            if val_epoch_accuracy >= self.best_acc:
                self.best_acc = val_epoch_accuracy
                print(f"[Trainer] 🏆 New best model! Validation Acc: {self.best_acc:.3f}%")
                torch.save(self.model.state_dict(), self.checkpoint_path / 'net_params_best.pth')
                self.best_model = copy.deepcopy(self.model)

            # 주기적 모델 저장
            if epoch % self.cfg_paths.save_interval == 0 and epoch != 0:
                torch.save(self.model.state_dict(), self.checkpoint_path / f'epoch_{epoch+1}_net_params.pth')
            
            # 주기적 상세 성능 평가
            if epoch % self.cfg_paths.test_interval == 0 and epoch != 0:
                print("\n[Trainer] Performing intermediate evaluation...")
                perform_evaluation(self.model, self.train_loader, self.val_loader, self.device)

            # 🔥 손실 분해 분석
            if epoch % 100 == 0:
                self._analyze_loss_breakdown(epoch)

            epoch_duration = time.time() - start_time
            print(f"--- Epoch {epoch+1}/{self.cfg_training.epoch_num} completed in {epoch_duration:.2f}s ---")

        print("\n[Trainer] Pre-training finished.")
        print("------ Final Evaluation with Best Model ------")
        perform_evaluation(self.best_model, self.train_loader, self.val_loader, self.device)

    def fit(self, epoch, phase='training'):
        """
        🔥 하이브리드 손실이 적용된 훈련/검증 루프
        
        Loss Computation Process:
        1. Forward pass through CCNet model
        2. ArcFace loss from classification output  
        3. SupCon loss from feature embeddings
        4. Weighted combination: L = α*L_ArcFace + β*L_SupCon
        """
        if phase == 'training':
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.val_loader

        running_loss = 0.0
        running_arcface_loss = 0.0
        running_supcon_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for datas, target in tqdm(data_loader, desc=f"Epoch {epoch+1} - {phase.capitalize()}"):
            data = datas[0].to(self.device)      # 첫 번째 이미지
            data_con = datas[1].to(self.device)  # 대조 학습용 두 번째 이미지
            target = target.to(self.device)
            
            batch_size = data.size(0)
            total_samples += batch_size

            if phase == 'training':
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'training'):
                # 🔥 1. CCNet Forward Pass (ArcFace 포함)
                output, fe1 = self.model(data, target if phase == 'training' else None)
                _, fe2 = self.model(data_con, target if phase == 'training' else None)
                
                # 🔥 2. ArcFace 손실 계산 (분류 성능 + 마진)
                arcface_loss = self.arcface_criterion(output, target)
                
                # 🔥 3. SupCon 손실 계산 (특징 공간 구조화)
                features_for_contrastive = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
                supcon_loss = self.supcon_criterion(features_for_contrastive, target)
                
                # 🔥 4. 하이브리드 손실 결합
                total_loss = self.alpha * arcface_loss + self.beta * supcon_loss
                
                if phase == 'training':
                    total_loss.backward()
                    self.optimizer.step()

            # 통계 기록
            running_loss += total_loss.item() * batch_size
            running_arcface_loss += arcface_loss.item() * batch_size
            running_supcon_loss += supcon_loss.item() * batch_size
            
            _, preds = torch.max(output, 1)
            running_corrects += torch.sum(preds == target.data)

        # 에포크 결과 계산
        epoch_loss = running_loss / total_samples
        epoch_arcface_loss = running_arcface_loss / total_samples
        epoch_supcon_loss = running_supcon_loss / total_samples
        epoch_acc = (running_corrects.double() / total_samples) * 100
        
        # 로그 기록
        if phase == 'training':
            self.train_losses.append(epoch_loss)
            self.train_accuracy.append(epoch_acc.item())
            self.loss_breakdown['arcface_losses'].append(epoch_arcface_loss)
            self.loss_breakdown['supcon_losses'].append(epoch_supcon_loss)
            self.loss_breakdown['total_losses'].append(epoch_loss)
        else:
            self.val_losses.append(epoch_loss)
            self.val_accuracy.append(epoch_acc.item())

        # 🔥 상세한 손실 분해 출력
        print(f'📊 Epoch {epoch+1} {phase.capitalize()} Loss Breakdown:')
        print(f'   ArcFace: {epoch_arcface_loss:.6f} (weight: {self.alpha}) → {self.alpha * epoch_arcface_loss:.6f}')
        print(f'   SupCon:  {epoch_supcon_loss:.6f} (weight: {self.beta}) → {self.beta * epoch_supcon_loss:.6f}')
        print(f'   Total:   {epoch_loss:.6f}')
        print(f'   Accuracy: {epoch_acc:.3f}%')
        print(f'   Verification: {self.alpha * epoch_arcface_loss + self.beta * epoch_supcon_loss:.6f}')
        
        return epoch_loss, epoch_acc.item()

    def _analyze_loss_breakdown(self, epoch):
        """🔥 손실 분해 분석"""
        print(f"\n[Loss Analysis] Epoch {epoch} Breakdown:")
        
        if len(self.loss_breakdown['arcface_losses']) > 0:
            recent_arcface = self.loss_breakdown['arcface_losses'][-1]
            recent_supcon = self.loss_breakdown['supcon_losses'][-1]
            recent_total = self.loss_breakdown['total_losses'][-1]
            
            arcface_contribution = self.alpha * recent_arcface
            supcon_contribution = self.beta * recent_supcon
            
            print(f"  ArcFace Loss: {recent_arcface:.5f} (weight: {self.alpha}) → {arcface_contribution:.5f}")
            print(f"  SupCon Loss:  {recent_supcon:.5f} (weight: {self.beta}) → {supcon_contribution:.5f}")
            print(f"  Total Loss:   {recent_total:.5f}")
            print(f"  Loss Ratio:   {recent_arcface/recent_supcon:.2f} (ArcFace/SupCon)")
        
        print()  # 줄바꿈