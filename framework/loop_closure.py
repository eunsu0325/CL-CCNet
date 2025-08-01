# loop_closure.py
"""
Loop Closure System
CCNet 스타일 충돌 감지 + 온라인 재학습을 통한 해결
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm

class LoopClosureSystem:
    """
    Loop Closure: 사용자 노드 충돌 감지 및 해결
    
    1. CCNet 스타일로 충돌 감지
    2. 충돌 시 원본 샘플들로 재학습
    3. 거리 보정 후 노드 업데이트
    """
    
    def __init__(self, node_manager, replay_buffer, learner_net, optimizer, criterion, device='cuda'):
        self.node_manager = node_manager
        self.replay_buffer = replay_buffer
        self.learner_net = learner_net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # CCNet 스타일 파라미터
        self.collision_threshold = 0.3  # 코사인 거리 임계값 (arccos/π)
        self.resolution_epochs = 5      # 충돌 해결 학습 에폭
        self.top_k = 5                  # Faiss Top-K 검색
        
        # 통계
        self.stats = {
            'total_registrations': 0,
            'collisions_detected': 0,
            'collisions_resolved': 0,
            'failed_resolutions': 0
        }
        
        print(f"[LoopClosure] ✅ System initialized")
        print(f"[LoopClosure] Collision threshold: {self.collision_threshold}")
        print(f"[LoopClosure] Resolution epochs: {self.resolution_epochs}")
    
    def check_collision(self, new_user_id: int, 
                       new_embeddings: torch.Tensor,
                       new_samples: List[torch.Tensor]) -> Optional[Dict]:
        """
        CCNet 스타일 충돌 검사
        
        Returns:
            충돌 정보 또는 None
        """
        self.stats['total_registrations'] += 1
        
        # 1. 새 사용자의 평균 임베딩
        new_mean_embedding = new_embeddings.mean(dim=0)
        
        # 2. Faiss로 Top-K 가장 가까운 사용자 찾기
        top_k_users = self.node_manager.find_nearest_users(new_mean_embedding, k=self.top_k)
        
        if not top_k_users:
            return None
        
        # 3. CCNet 스타일 정밀 거리 계산
        for candidate_user_id, _ in top_k_users:
            if candidate_user_id == new_user_id:
                continue
                
            # 기존 사용자 노드 가져오기
            existing_node = self.node_manager.get_node(candidate_user_id)
            if not existing_node or existing_node.mean_embedding is None:
                continue
            
            # CCNet 코사인 거리 계산
            distance = self._compute_ccnet_distance(new_mean_embedding, existing_node.mean_embedding)
            
            print(f"[LoopClosure] Distance to User {candidate_user_id}: {distance:.4f}")
            
            # 충돌 판정
            if distance < self.collision_threshold:
                self.stats['collisions_detected'] += 1
                
                collision_info = {
                    'new_user_id': new_user_id,
                    'existing_user_id': candidate_user_id,
                    'distance': distance,
                    'new_embeddings': new_embeddings,
                    'new_samples': new_samples
                }
                
                print(f"[LoopClosure] ⚠️ COLLISION DETECTED!")
                print(f"  New User {new_user_id} <-> Existing User {candidate_user_id}")
                print(f"  Distance: {distance:.4f} < {self.collision_threshold}")
                
                return collision_info
        
        return None
    
    def resolve_collision(self, collision_info: Dict) -> Dict:
        """
        충돌 해결: 온라인 재학습을 통한 거리 보정
        """
        print(f"\n[LoopClosure] 🔧 Starting collision resolution...")
        
        new_user_id = collision_info['new_user_id']
        existing_user_id = collision_info['existing_user_id']
        new_samples = collision_info['new_samples']
        original_distance = collision_info['distance']
        
        # 1. 기존 사용자의 원본 샘플 가져오기
        existing_node = self.node_manager.get_node(existing_user_id)
        
        # 기존 사용자의 등록 이미지를 버퍼에서 찾기
        existing_samples = self._get_user_samples_from_buffer(existing_user_id)
        
        if not existing_samples and existing_node.registration_image is not None:
            # 노드에 저장된 등록 이미지 사용
            existing_sample_tensor = self._image_to_tensor(existing_node.registration_image)
            existing_samples = [existing_sample_tensor]
        
        if not existing_samples:
            print(f"[LoopClosure] ⚠️ No samples found for existing user {existing_user_id}")
            return {'success': False, 'reason': 'no_existing_samples'}
        
        print(f"[LoopClosure] Found {len(existing_samples)} samples for existing user")
        
        # 2. 특별 학습 배치 구성
        training_batch = self._construct_collision_batch(
            new_samples, 
            existing_samples, 
            new_user_id, 
            existing_user_id
        )
        
        # 3. 집중 학습 수행
        print(f"[LoopClosure] Performing {self.resolution_epochs} epochs of focused training...")
        
        distance_history = []
        
        for epoch in range(self.resolution_epochs):
            # 학습
            loss = self._train_collision_batch(training_batch)
            
            # 중간 거리 확인
            with torch.no_grad():
                new_features = self._extract_features(new_samples)
                existing_features = self._extract_features(existing_samples)
                
                new_mean = new_features.mean(dim=0)
                existing_mean = existing_features.mean(dim=0)
                
                current_distance = self._compute_ccnet_distance(new_mean, existing_mean)
                distance_history.append(current_distance)
                
                print(f"  Epoch {epoch+1}/{self.resolution_epochs}: "
                      f"Loss={loss:.4f}, Distance={current_distance:.4f}")
        
        # 4. 최종 결과 평가
        final_distance = distance_history[-1]
        success = final_distance >= self.collision_threshold
        
        if success:
            # 5. 재학습 후 새로운 충돌 확인
            print(f"[LoopClosure] 🔍 Checking for new collisions after resolution...")
            
            new_features = self._extract_features(new_samples)
            existing_features = self._extract_features(existing_samples)
            
            # 새 사용자의 다른 충돌 확인
            new_mean = new_features.mean(dim=0)
            new_collisions = self._check_other_collisions(new_mean, new_user_id, [existing_user_id])
            
            # 기존 사용자의 다른 충돌 확인
            existing_mean = existing_features.mean(dim=0)
            existing_collisions = self._check_other_collisions(existing_mean, existing_user_id, [new_user_id])
            
            if new_collisions or existing_collisions:
                success = False
                print(f"[LoopClosure] ⚠️ New collisions detected after resolution!")
                if new_collisions:
                    print(f"  New user {new_user_id} collides with: {new_collisions}")
                if existing_collisions:
                    print(f"  Existing user {existing_user_id} collides with: {existing_collisions}")
            else:
                print(f"[LoopClosure] ✅ No new collisions detected")
        
        if success:
            self.stats['collisions_resolved'] += 1
            print(f"[LoopClosure] ✅ Collision successfully resolved!")
        else:
            self.stats['failed_resolutions'] += 1
            print(f"[LoopClosure] ❌ Failed to resolve collision")
        
        print(f"  Original distance: {original_distance:.4f}")
        print(f"  Final distance: {final_distance:.4f}")
        print(f"  Improvement: {final_distance - original_distance:.4f}")
        
        # 6. 노드 업데이트 (성공한 경우만)
        if success:
            # 새 사용자 노드 업데이트
            new_features = self._extract_features(new_samples)
            self.node_manager.update_user(new_user_id, new_features, new_samples[0].cpu().numpy())
            
            # 기존 사용자 노드 업데이트
            existing_features = self._extract_features(existing_samples)
            self.node_manager.update_user(existing_user_id, existing_features)
            
            print(f"[LoopClosure] 📝 Both user nodes updated successfully")
        
        return {
            'success': success,
            'original_distance': original_distance,
            'final_distance': final_distance,
            'improvement': final_distance - original_distance,
            'distance_history': distance_history,
            'epochs_trained': self.resolution_epochs
        }
    
    def _compute_ccnet_distance(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """CCNet 스타일 코사인 거리 계산"""
        # L2 정규화
        feat1_norm = F.normalize(feat1.unsqueeze(0), p=2, dim=1)
        feat2_norm = F.normalize(feat2.unsqueeze(0), p=2, dim=1)
        
        # 코사인 유사도
        cosine_sim = torch.dot(feat1_norm[0], feat2_norm[0]).item()
        
        # 안전한 범위로 클리핑
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # 각도 거리 변환 (CCNet 스타일)
        distance = np.arccos(cosine_sim) / np.pi
        
        return distance
    
    def _get_user_samples_from_buffer(self, user_id: int) -> List[torch.Tensor]:
        """리플레이 버퍼에서 사용자 샘플 가져오기"""
        user_samples = []
        
        for item in self.replay_buffer.image_storage:
            if item['user_id'] == user_id:
                user_samples.append(item['image'])
        
        return user_samples
    
    def _image_to_tensor(self, numpy_image: np.ndarray) -> torch.Tensor:
        """numpy 이미지를 텐서로 변환"""
        # numpy (H, W, C) -> tensor (C, H, W)
        if len(numpy_image.shape) == 3:
            tensor = torch.from_numpy(numpy_image).permute(2, 0, 1).float() / 255.0
        else:
            tensor = torch.from_numpy(numpy_image).unsqueeze(0).float() / 255.0
        return tensor
    
    def _construct_collision_batch(self, new_samples: List[torch.Tensor],
                                 existing_samples: List[torch.Tensor],
                                 new_user_id: int,
                                 existing_user_id: int) -> Dict:
        """충돌 해결용 특별 배치 구성"""
        # 전체 배치 구성
        all_images = []
        all_labels = []
        
        # 새 사용자 샘플
        all_images.extend(new_samples)
        all_labels.extend([new_user_id] * len(new_samples))
        
        # 기존 사용자 샘플
        all_images.extend(existing_samples)
        all_labels.extend([existing_user_id] * len(existing_samples))
        
        # 리플레이 버퍼에서 추가 샘플 (하드 네거티브)
        buffer_size = 32 - len(all_images)  # 전체 배치 크기 32
        if buffer_size > 0 and len(self.replay_buffer.image_storage) > 0:
            # 충돌한 두 사용자를 제외한 샘플들
            buffer_samples = []
            buffer_labels = []
            
            for item in self.replay_buffer.image_storage:
                if item['user_id'] not in [new_user_id, existing_user_id]:
                    buffer_samples.append(item['image'])
                    buffer_labels.append(item['user_id'])
                    
                    if len(buffer_samples) >= buffer_size:
                        break
            
            all_images.extend(buffer_samples[:buffer_size])
            all_labels.extend(buffer_labels[:buffer_size])
        
        print(f"[LoopClosure] Collision batch composition:")
        print(f"  New user samples: {len(new_samples)}")
        print(f"  Existing user samples: {len(existing_samples)}")
        print(f"  Buffer samples: {len(all_images) - len(new_samples) - len(existing_samples)}")
        print(f"  Total batch size: {len(all_images)}")
        
        return {
            'images': all_images,
            'labels': all_labels
        }
    
    def _train_collision_batch(self, batch_data: Dict) -> float:
        """충돌 해결을 위한 학습"""
        self.learner_net.train()
        self.optimizer.zero_grad()
        
        images = batch_data['images']
        labels = batch_data['labels']
        
        # 특징 추출
        embeddings = []
        for img in images:
            img_tensor = img.to(self.device)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            _, embedding = self.learner_net(img_tensor)
            embeddings.append(embedding)
        
        embeddings_tensor = torch.cat(embeddings, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # SupCon Loss
        loss_dict = self.criterion(embeddings_tensor, labels_tensor)
        
        # Backward
        loss_dict['total'].backward()
        self.optimizer.step()
        
        return loss_dict['total'].item()
    
    def _extract_features(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """특징 추출"""
        self.learner_net.eval()
        
        features = []
        with torch.no_grad():
            for sample in samples:
                if len(sample.shape) == 3:
                    sample = sample.unsqueeze(0)
                sample = sample.to(self.device)
                
                feature = self.learner_net.getFeatureCode(sample)
                features.append(feature)
        
        self.learner_net.train()
        return torch.cat(features, dim=0)
    
    def _check_other_collisions(self, embedding: torch.Tensor, 
                               current_user_id: int, 
                               exclude_users: List[int]) -> List[Tuple[int, float]]:
        """
        재학습 후 다른 사용자들과의 충돌 확인
        
        Returns:
            충돌하는 사용자들의 리스트 [(user_id, distance), ...]
        """
        collisions = []
        
        # Faiss로 Top-K 검색
        top_k_users = self.node_manager.find_nearest_users(embedding, k=self.top_k * 2)
        
        for candidate_user_id, _ in top_k_users:
            # 자기 자신과 이미 처리한 사용자 제외
            if candidate_user_id == current_user_id or candidate_user_id in exclude_users:
                continue
            
            # 해당 사용자 노드 가져오기
            candidate_node = self.node_manager.get_node(candidate_user_id)
            if not candidate_node or candidate_node.mean_embedding is None:
                continue
            
            # CCNet 거리 계산
            distance = self._compute_ccnet_distance(embedding, candidate_node.mean_embedding)
            
            # 충돌 확인
            if distance < self.collision_threshold:
                collisions.append((candidate_user_id, distance))
        
        return collisions
    
    def get_statistics(self) -> Dict:
        """Loop Closure 통계"""
        success_rate = 0
        if self.stats['collisions_detected'] > 0:
            success_rate = self.stats['collisions_resolved'] / self.stats['collisions_detected'] * 100
        
        return {
            'total_registrations': self.stats['total_registrations'],
            'collisions_detected': self.stats['collisions_detected'],
            'collisions_resolved': self.stats['collisions_resolved'],
            'failed_resolutions': self.stats['failed_resolutions'],
            'collision_rate': self.stats['collisions_detected'] / max(1, self.stats['total_registrations']) * 100,
            'resolution_success_rate': success_rate
        }
    
    def print_statistics(self):
        """통계 출력"""
        stats = self.get_statistics()
        
        print(f"\n[LoopClosure] 📊 Statistics:")
        print(f"  Total registrations: {stats['total_registrations']}")
        print(f"  Collisions detected: {stats['collisions_detected']} ({stats['collision_rate']:.1f}%)")
        print(f"  Collisions resolved: {stats['collisions_resolved']}")
        print(f"  Failed resolutions: {stats['failed_resolutions']}")
        print(f"  Resolution success rate: {stats['resolution_success_rate']:.1f}%")


# 사용 예시
def integrate_loop_closure(coconut_system):
    """CoCoNut 시스템에 Loop Closure 통합"""
    
    # Loop Closure 시스템 생성
    loop_closure = LoopClosureSystem(
        node_manager=coconut_system.node_manager,
        replay_buffer=coconut_system.replay_buffer,
        learner_net=coconut_system.learner_net,
        optimizer=coconut_system.optimizer,
        criterion=coconut_system.criterion,
        device=coconut_system.device
    )
    
    # process_label_batch 메서드 수정
    original_process = coconut_system.process_label_batch
    
    def process_with_loop_closure(samples, user_id):
        # 먼저 임베딩 추출
        embeddings = coconut_system._extract_batch_features(samples)
        
        # Loop Closure 체크
        collision_info = loop_closure.check_collision(user_id, embeddings, samples)
        
        if collision_info:
            # 충돌 해결
            resolution = loop_closure.resolve_collision(collision_info)
            
            if not resolution['success']:
                print(f"[LoopClosure] ⚠️ Failed to resolve collision, proceeding anyway...")
        
        # 원래 프로세스 진행
        return original_process(samples, user_id)
    
    # 메서드 교체
    coconut_system.process_label_batch = process_with_loop_closure
    coconut_system.loop_closure = loop_closure
    
    return coconut_system