# pretrain.py - 사전 훈련 실행 스크립트
"""
COCONUT Stage 1: Pre-training Execution Script

DESIGN PHILOSOPHY:
- Build robust feature space using hybrid loss
- Prepare foundation for Stage 2 adaptation
- No W2ML to prevent dataset-specific overfitting
"""

from config.config_parser import ConfigParser
from models.trainer import CCNetTrainer

def main():
    """
    COCONUT 프로젝트의 사전 훈련을 위한 메인 실행 스크립트.
    
    EXECUTION FLOW:
    1. Load pretrain configuration
    2. Initialize CCNet trainer with hybrid loss
    3. Execute training with ArcFace + SupCon
    4. Save robust feature space foundation
    """
    print("="*80)
    print("🥥 COCONUT STAGE 1: PRE-TRAINING EXECUTION")
    print("="*80)
    
    # 1. 설정 파일 로드
    config = ConfigParser('./config/pretrain_config.yaml')
    print("--- COCONUT Pre-training Configuration ---")
    print(config)

    # 2. 훈련 전문가(Trainer) 객체 생성
    trainer = CCNetTrainer(config)

    # 3. 훈련 시작
    trainer.train()

    print("\n[INFO] ✅ Pre-training finished successfully.")
    print("[INFO] 🎯 Robust feature space foundation created")
    print("[INFO] 🔄 Ready for Stage 2 adaptation")


if __name__ == '__main__':
    main()