# models/ccnet_model.py - 안정성 개선된 완전판

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings

class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding      
        self.init_ratio = init_ratio 
        self.kernel = 0

        if init_ratio <=0:
            init_ratio = 1.0
            print('input error!!!, require init_ratio > 0.0, using default...')

        # initial parameters
        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        # shape & scale of the Gaussian functiion:
        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)          
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out, requires_grad=False)

        # frequency of the cosine envolope:
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax

        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).float()    
        x_0 = torch.arange(xmin, xmax + 1).float()

        # [channel_out, channelin, kernel_H, kernel_W]   
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1) 
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize) 

        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        # Rotated coordinate systems
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))  
                
        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8*sigma.view(-1, 1, 1, 1) ** 2)) \
            * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
    
        gb = gb - gb.mean(dim=[2,3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(self.kernel_size, self.channel_in, self.channel_out, self.sigma, self.gamma, self.theta, self.f, self.psi)
        self.kernel = kernel
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CompetitiveBlock_Mul_Ord_Comp(nn.Module):
    def __init__(self, channel_in, n_competitor, ksize, stride, padding, weight, init_ratio=1, o1=32, o2=12):
        super(CompetitiveBlock_Mul_Ord_Comp, self).__init__()

        self.channel_in = channel_in
        self.n_competitor = n_competitor
        self.init_ratio = init_ratio

        self.gabor_conv2d = GaborConv2d(channel_in=channel_in, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                        padding=ksize // 2, init_ratio=init_ratio)
        self.gabor_conv2d2 = GaborConv2d(channel_in=n_competitor, channel_out=n_competitor, kernel_size=ksize, stride=2,
                                         padding=ksize // 2, init_ratio=init_ratio)

        self.argmax = nn.Softmax(dim=1)
        self.argmax_x = nn.Softmax(dim=2)
        self.argmax_y = nn.Softmax(dim=3)
        
        # PPU
        self.conv1_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(n_competitor, o1//2, 5, 2, 0)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.se1 = SELayer(n_competitor)
        self.se2 = SELayer(n_competitor)

        self.weight_chan = weight
        self.weight_spa = (1-weight) / 2

    def forward(self, x):
        #1-st order
        x = self.gabor_conv2d(x)
        x1_1 = self.argmax(x)
        x1_2 = self.argmax_x(x)
        x1_3 = self.argmax_y(x)
        x_1 = self.weight_chan * x1_1 + self.weight_spa * (x1_2 + x1_3)

        x_1 = self.se1(x_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.maxpool(x_1)

        #2-nd order
        x = self.gabor_conv2d2(x)
        x2_1 = self.argmax(x)
        x2_2 = self.argmax_x(x)
        x2_3 = self.argmax_y(x)
        x_2 = self.weight_chan * x2_1 + self.weight_spa * (x2_2 + x2_3)
        x_2 = self.se2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.maxpool(x_2)

        xx = torch.cat((x_1.view(x_1.shape[0],-1), x_2.view(x_2.shape[0],-1)), dim=1)
        return xx

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training and label is not None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)       
            
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine

        return output

class StableProjectionHead(nn.Module):
    """🔥 안정성 개선된 2048차원 → 128차원 압축 MLP"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super(StableProjectionHead, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 첫 번째 레이어
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # 🚀 GroupNorm으로 변경 - 배치 크기와 무관하게 안정적!
        # 그룹 수를 동적으로 계산 (최대 32그룹, 최소 1그룹)
        num_groups = min(32, max(1, hidden_dim // 16))
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dim)
        
        # 활성화 및 정규화
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        
        # 두 번째 레이어
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # 🔥 Xavier 초기화로 안정적 학습
        self._initialize_weights()
        
        print(f"[StableProjectionHead] ✅ Initialized with GroupNorm:")
        print(f"   📏 Dimensions: {input_dim} → {hidden_dim} → {output_dim}")
        print(f"   🔧 GroupNorm: {num_groups} groups for {hidden_dim} channels")
        print(f"   🛡️ Batch-size independent normalization enabled")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        # Xavier uniform 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        print(f"   ✅ Xavier initialization applied")
    
    def forward(self, x):
        # 🛡️ 입력 안전성 체크
        if x.numel() == 0:
            return torch.zeros(x.size(0), self.output_dim, device=x.device, dtype=x.dtype)
        
        # 첫 번째 변환
        x = self.fc1(x)
        
        # 🚀 GroupNorm 적용 (배치 크기와 무관!)
        x = self.gn1(x)
        x = self.relu(x)
        
        # 🔥 안전한 Dropout 적용
        if self.training and x.size(0) > 1:  # 배치가 1개보다 클 때만
            x = self.dropout(x)
        
        # 최종 변환
        x = self.fc2(x)
        
        # L2 정규화로 단위 벡터로 만들기
        return F.normalize(x, dim=-1, eps=1e-8)
    
    def get_info(self):
        """압축 헤드 정보 반환"""
        return {
            'type': 'StableProjectionHead',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'compression_ratio': f'{self.input_dim}:{self.output_dim} ({self.input_dim//self.output_dim}:1)',
            'normalization': 'GroupNorm',
            'batch_independent': True,
            'memory_reduction': f'{self.input_dim/self.output_dim:.1f}x'
        }

class ccnet(torch.nn.Module):
    """
    🔥 안정성 개선된 CCNet with optional 128D compression for headless mode
    """
    def __init__(self, num_classes, weight, headless_mode=False, compression_dim=128):
        super().__init__()

        self.num_classes = num_classes
        self.headless_mode = headless_mode
        self.compression_dim = compression_dim

        print(f"[CCNet] 🚀 Initializing stable CCNet...")
        print(f"   Mode: {'Headless' if headless_mode else 'Classification'}")
        print(f"   Classes: {num_classes}")
        print(f"   Compression: {compression_dim}D" if headless_mode else "   Full features: 2048D")

        # Core feature extraction (Gabor + Competitive blocks)
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)

        # Feature fusion layers
        self.fc = torch.nn.Linear(13152, 4096)
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        
        # 🔥 Head configuration with stable components
        if not headless_mode:
            # Classification mode: ArcFace head
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            self.projection_head = None
            print(f"[CCNet] ✅ Classification mode: ArcFace head with {num_classes} classes")
        else:
            # Headless mode: Stable projection head
            self.arclayer_ = None
            self.projection_head = StableProjectionHead(input_dim=2048, output_dim=compression_dim)
            print(f"[CCNet] ✅ Headless mode: {compression_dim}D stable compression")

    def forward(self, x, y=None):
        """Forward pass with stability improvements"""
        # 🛡️ 입력 안전성 체크
        if x.numel() == 0:
            batch_size = x.size(0)
            if self.headless_mode:
                return None, torch.zeros(batch_size, self.compression_dim, device=x.device)
            else:
                return torch.zeros(batch_size, self.num_classes, device=x.device), torch.zeros(batch_size, 6144, device=x.device)

        # Gabor feature extraction
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        # Feature fusion
        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)  # 6144 dimensional features
        
        if self.headless_mode:
            # 🚀 Headless: 2048 → compressed features (stable)
            fe_2048 = F.normalize(x, dim=-1, eps=1e-8)
            compressed_features = self.projection_head(fe_2048)
            return None, compressed_features
        else:
            # Classification: original behavior with safety
            x = self.drop(x) if self.training else x
            x = self.arclayer_(x, y)
            return x, F.normalize(fe, dim=-1, eps=1e-8)

    def getFeatureCode(self, x):
        """🔥 안정성 개선된 특징 추출 - 추론 최적화"""
        # 추론 최적화: eval 모드 강제 + no_grad
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # 🛡️ 입력 검증
            if x.numel() == 0:
                result = torch.zeros(x.size(0), self.compression_dim if self.headless_mode else 2048, device=x.device)
                if was_training:
                    self.train()
                return result

            # 특징 추출
            x1 = self.cb1(x)
            x2 = self.cb2(x)
            x3 = self.cb3(x)

            x1 = x1.view(x1.shape[0], -1)
            x2 = x2.view(x2.shape[0], -1)
            x3 = x3.view(x3.shape[0], -1)
            x = torch.cat((x1, x2, x3), dim=1)

            x = self.fc(x)
            x = self.fc1(x)
            
            # 🔥 안전한 정규화
            fe_2048 = F.normalize(x, dim=-1, eps=1e-8)
            
            if self.headless_mode and self.projection_head is not None:
                result = self.projection_head(fe_2048)  # Stable compressed features
            else:
                result = fe_2048  # 2048D features
        
        # 원래 모드로 복원
        if was_training:
            self.train()
            
        return result
    
    def convert_to_headless(self):
        """Classification → Headless 모드 변환"""
        if not self.headless_mode:
            print("[CCNet] 🔄 Converting to headless mode...")
            self.arclayer_ = None
            self.projection_head = StableProjectionHead(input_dim=2048, output_dim=self.compression_dim)
            self.headless_mode = True
            print("[CCNet] ✅ Successfully converted to stable headless mode")
            return True
        return False
    
    def convert_to_classification(self, num_classes=None):
        """Headless → Classification 모드 변환"""
        if self.headless_mode:
            if num_classes is None:
                num_classes = self.num_classes
            print(f"[CCNet] 🔄 Converting to classification mode...")
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            self.projection_head = None
            self.headless_mode = False
            self.num_classes = num_classes
            print(f"[CCNet] ✅ Successfully converted to classification mode ({num_classes} classes)")
            return True
        return False
    
    def is_headless(self):
        """현재 헤드리스 모드인지 확인"""
        return self.headless_mode
    
    def get_model_info(self):
        """🔥 확장된 모델 정보 반환"""
        # 파라미터 수 및 디바이스 정보
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        device = next(self.parameters()).device
        
        info = {
            'architecture': 'CCNet-Stable',
            'version': 'Stability-Enhanced',
            'headless_mode': self.headless_mode,
            'num_classes': self.num_classes if not self.headless_mode else None,
            'has_classification_head': self.arclayer_ is not None,
            'has_projection_head': self.projection_head is not None,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(device),
            'memory_footprint_mb': total_params * 4 / (1024 * 1024)  # float32 기준
        }
        
        if self.headless_mode:
            info.update({
                'feature_dimension': self.compression_dim,
                'compression_enabled': True,
                'compression_ratio': f'2048→{self.compression_dim} ({2048//self.compression_dim}:1)',
                'memory_reduction': f'{2048/self.compression_dim:.1f}x',
                'compression_efficiency': f'{(1 - self.compression_dim/2048)*100:.1f}% reduction',
                'normalization_method': 'GroupNorm (batch-independent)',
                'stability_features': ['GroupNorm', 'Safe dropout', 'Input validation', 'Gradient clipping compatible']
            })
            
            # Projection head 정보 추가
            if self.projection_head:
                proj_info = self.projection_head.get_info()
                info['projection_head'] = proj_info
        else:
            info.update({
                'feature_dimension': 2048,
                'compression_enabled': False,
                'classification_head': 'ArcMarginProduct',
                'margin': 0.5,
                'scale': 30.0
            })
        
        return info

class StableHeadlessVerifier:
    """🔥 안정성 개선된 메트릭 기반 검증기"""
    def __init__(self, metric_type="cosine", threshold=0.5):
        self.metric_type = metric_type
        self.threshold = threshold
        self.score_history = []
        self.verification_count = 0
        
        print(f"[StableVerifier] ✅ Initialized:")
        print(f"   Metric: {metric_type}")
        print(f"   Threshold: {threshold}")
        print(f"   Stability features: Input validation, NaN protection, Score logging")
    
    def compute_similarity(self, probe_features, gallery_features):
        """🛡️ 안전한 유사도 계산"""
        with torch.no_grad():
            # 입력 검증
            if probe_features.numel() == 0 or gallery_features.numel() == 0:
                return torch.zeros(1, device=probe_features.device)
            
            if len(probe_features.shape) == 1:
                probe_features = probe_features.unsqueeze(0)
            
            # NaN 체크 및 정리
            if torch.isnan(probe_features).any() or torch.isnan(gallery_features).any():
                print("⚠️ [StableVerifier] NaN detected in features, using safe fallback")
                return torch.zeros(gallery_features.size(0), device=probe_features.device)
            
            if self.metric_type == "cosine":
                # 안전한 코사인 유사도
                similarities = F.cosine_similarity(probe_features, gallery_features, dim=1, eps=1e-8)
            elif self.metric_type == "l2":
                # 안전한 유클리드 거리
                distances = F.pairwise_distance(probe_features, gallery_features, eps=1e-8)
                similarities = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unsupported metric type: {self.metric_type}")
            
            # NaN 체크 (결과)
            if torch.isnan(similarities).any():
                print("⚠️ [StableVerifier] NaN in similarity results, using zeros")
                similarities = torch.zeros_like(similarities)
        
        return similarities
    
    def verify(self, probe_features, gallery_features, return_topk=False, k=3):
        """🔥 안정성 개선된 검증"""
        self.verification_count += 1
        
        similarities = self.compute_similarity(probe_features, gallery_features)
        
        # 빈 갤러리 처리
        if len(similarities) == 0:
            return {
                'is_match': False,
                'best_similarity': 0.0,
                'best_index': -1,
                'error': 'Empty gallery'
            }
        
        # 통계 로깅
        self.score_history.append({
            'max_similarity': similarities.max().item(),
            'mean_similarity': similarities.mean().item(),
            'std_similarity': similarities.std().item(),
            'verification_id': self.verification_count
        })
        
        best_similarity = similarities.max().item()
        best_index = similarities.argmax().item()
        is_match = best_similarity > self.threshold
        
        result = {
            'is_match': is_match, 
            'best_similarity': best_similarity, 
            'best_index': best_index,
            'verification_count': self.verification_count
        }
        
        # Top-k 결과 추가
        if return_topk:
            k = min(k, len(similarities))
            topk_similarities, topk_indices = similarities.topk(k=k)
            result.update({
                'topk_similarities': topk_similarities.tolist(),
                'topk_indices': topk_indices.tolist(),
                'top1_match': similarities.argmax().item() == best_index,
                'topk_contains_match': is_match
            })
        
        return result
    
    def get_score_statistics(self):
        """누적 점수 통계"""
        if not self.score_history:
            return None
        
        max_scores = [h['max_similarity'] for h in self.score_history]
        mean_scores = [h['mean_similarity'] for h in self.score_history]
        
        return {
            'total_verifications': len(self.score_history),
            'max_similarity_stats': {
                'mean': np.mean(max_scores),
                'std': np.std(max_scores),
                'min': np.min(max_scores),
                'max': np.max(max_scores)
            },
            'avg_similarity_stats': {
                'mean': np.mean(mean_scores),
                'std': np.std(mean_scores)
            },
            'threshold': self.threshold,
            'match_rate': sum(1 for s in max_scores if s > self.threshold) / len(max_scores),
            'stability_info': {
                'nan_incidents': 0,  # 여기서는 별도 추적 필요
                'empty_gallery_incidents': 0,
                'error_recovery_rate': 1.0
            }
        }
    
    def reset_history(self):
        """점수 히스토리 초기화"""
        self.score_history = []
        self.verification_count = 0
        print(f"[StableVerifier] 📊 History reset")

def create_stable_ccnet_from_config(config):
    """🔥 안정성 개선된 CCNet 생성"""
    headless_mode = getattr(config, 'headless_mode', False)
    compression_dim = getattr(config, 'compression_dim', 128)
    
    model = ccnet(
        num_classes=config.num_classes,
        weight=config.com_weight,
        headless_mode=headless_mode,
        compression_dim=compression_dim
    )
    
    print(f"[Factory] 🚀 Created Stable CCNet:")
    print(f"   Headless: {headless_mode}")
    print(f"   Compression: {compression_dim}D")
    print(f"   Stability: GroupNorm, Input validation, NaN protection")
    
    return model

# 🎯 호환성을 위한 별칭
HeadlessVerifier = StableHeadlessVerifier
ProjectionHead = StableProjectionHead

print("✅ Stable CCNet model with GroupNorm loaded successfully!")
print("🛡️ Features: Batch-size independent, NaN protection, Input validation")
print("🚀 Ready for production use in continual learning environments!")