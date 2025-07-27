# models/ccnet_model.py - 기존과 완전 호환되는 Headless 지원 버전

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

class ProjectionHead(nn.Module):
    """2048차원 → 128차원 압축을 위한 MLP - 개선 버전"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Xavier 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        print(f"[ProjectionHead] Initialized: {input_dim} → {hidden_dim} → {output_dim}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # ✅ training 모드에서만 dropout 적용
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1)

class ccnet(torch.nn.Module):
    """
    CCNet with optional 128D compression for headless mode
    """
    def __init__(self, num_classes, weight, headless_mode=False, compression_dim=128):
        super().__init__()

        self.num_classes = num_classes
        self.headless_mode = headless_mode
        self.compression_dim = compression_dim

        # Core feature extraction
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)

        # Feature fusion
        self.fc = torch.nn.Linear(13152, 4096)
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        
        # Head configuration
        if not headless_mode:
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            self.projection_head = None
            print(f"[CCNet] Initialized with classification head (classes: {num_classes})")
        else:
            self.arclayer_ = None
            self.projection_head = ProjectionHead(input_dim=2048, output_dim=compression_dim)
            print(f"[CCNet] Initialized in HEADLESS mode with {compression_dim}D compression")

    def forward(self, x, y=None):
        # Feature extraction
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)  # 6144 dimensional features
        
        if self.headless_mode:
            # Headless: 2048 → 128 compression
            fe_2048 = F.normalize(x, dim=-1)
            compressed_features = self.projection_head(fe_2048)
            return None, compressed_features
        else:
            # Classification: original behavior
            x = self.drop(x)
            x = self.arclayer_(x, y)
            return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        """특징 추출 - 추론 최적화 적용"""
        # ✅ 추론 최적화: eval 모드 강제 + no_grad
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            x1 = self.cb1(x)
            x2 = self.cb2(x)
            x3 = self.cb3(x)

            x1 = x1.view(x1.shape[0], -1)
            x2 = x2.view(x2.shape[0], -1)
            x3 = x3.view(x3.shape[0], -1)
            x = torch.cat((x1, x2, x3), dim=1)

            x = self.fc(x)
            x = self.fc1(x)
            fe_2048 = x / torch.norm(x, p=2, dim=1, keepdim=True)
            
            if self.headless_mode and self.projection_head is not None:
                result = self.projection_head(fe_2048)  # 128D
            else:
                result = fe_2048  # 2048D
        
        # 원래 모드로 복원
        if was_training:
            self.train()
            
        return result
    
    def convert_to_headless(self):
        if not self.headless_mode:
            print("[CCNet] Converting to headless mode...")
            self.arclayer_ = None
            self.projection_head = ProjectionHead(input_dim=2048, output_dim=self.compression_dim)
            self.headless_mode = True
            return True
        return False
    
    def convert_to_classification(self, num_classes=None):
        if self.headless_mode:
            if num_classes is None:
                num_classes = self.num_classes
            print(f"[CCNet] Converting to classification mode...")
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            self.projection_head = None
            self.headless_mode = False
            self.num_classes = num_classes
            return True
        return False
    
    def is_headless(self):
        return self.headless_mode
    
    def get_model_info(self):
        """모델 정보 반환 - 확장 버전"""
        # ✅ 파라미터 수 및 디바이스 정보 추가
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        device = next(self.parameters()).device
        
        info = {
            'architecture': 'CCNet',
            'headless_mode': self.headless_mode,
            'num_classes': self.num_classes if not self.headless_mode else None,
            'has_classification_head': self.arclayer_ is not None,
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
                'compression_efficiency': f'{(1 - self.compression_dim/2048)*100:.1f}% reduction'
            })
        else:
            info.update({
                'feature_dimension': 2048,
                'compression_enabled': False
            })
        
        return info

class HeadlessVerifier:
    """메트릭 기반 검증기 - 확장 버전"""
    def __init__(self, metric_type="cosine", threshold=0.5):
        self.metric_type = metric_type
        self.threshold = threshold
        self.score_history = []  # ✅ score logging 추가
        print(f"[Verifier] Initialized: {metric_type}, threshold: {threshold}")
    
    def compute_similarity(self, probe_features, gallery_features):
        """✅ no_grad 최적화 적용"""
        with torch.no_grad():
            if len(probe_features.shape) == 1:
                probe_features = probe_features.unsqueeze(0)
            
            if self.metric_type == "cosine":
                similarities = F.cosine_similarity(probe_features, gallery_features, dim=1)
            elif self.metric_type == "l2":
                distances = F.pairwise_distance(probe_features, gallery_features)
                similarities = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unsupported metric type: {self.metric_type}")
        
        return similarities
    
    def verify(self, probe_features, gallery_features, return_topk=False, k=3):
        """✅ top-k 지원 추가"""
        similarities = self.compute_similarity(probe_features, gallery_features)
        
        # ✅ score logging
        self.score_history.append({
            'max_similarity': similarities.max().item(),
            'mean_similarity': similarities.mean().item(),
            'std_similarity': similarities.std().item()
        })
        
        best_similarity = similarities.max().item()
        best_index = similarities.argmax().item()
        is_match = best_similarity > self.threshold
        
        result = {
            'is_match': is_match, 
            'best_similarity': best_similarity, 
            'best_index': best_index
        }
        
        # ✅ top-k 결과 추가
        if return_topk:
            topk_similarities, topk_indices = similarities.topk(k=min(k, len(similarities)))
            result.update({
                'topk_similarities': topk_similarities.tolist(),
                'topk_indices': topk_indices.tolist(),
                'top1_match': similarities.argmax().item() == best_index,
                'topk_contains_match': is_match  # top-1이 매치면 top-k도 매치
            })
        
        return result
    
    def get_score_statistics(self):
        """✅ 누적 점수 통계"""
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
            'match_rate': sum(1 for s in max_scores if s > self.threshold) / len(max_scores)
        }
    
    def reset_history(self):
        """점수 히스토리 초기화"""
        self.score_history = []

def create_ccnet_from_config(config):
    """Config에서 CCNet 생성"""
    headless_mode = getattr(config, 'headless_mode', False)
    compression_dim = getattr(config, 'compression_dim', 128)
    
    model = ccnet(
        num_classes=config.num_classes,
        weight=config.com_weight,
        headless_mode=headless_mode,
        compression_dim=compression_dim
    )
    
    print(f"[Factory] Created CCNet: headless={headless_mode}, compression={compression_dim}")
    return model

