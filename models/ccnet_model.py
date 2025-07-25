# models/ccnet_model.py - 기존 CCNet과 완전 호환되는 Headless 지원 버전

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math
import warnings

# 기존 클래스들 그대로 유지
class GaborConv2d(nn.Module):
    '''
    DESCRIPTION: an implementation of the Learnable Gabor Convolution (LGC) layer \n
    INPUTS: \n
    channel_in: should be 1 \n
    channel_out: number of the output channels \n
    kernel_size, stride, padding: 2D convolution parameters \n
    init_ratio: scale factor of the initial parameters (receptive filed) \n
    '''
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
    '''
    DESCRIPTION: an implementation of the Competitive Block::
    [CB = LGC + argmax + PPU] \n
    '''
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
    r"""Implement of large margin arc distance"""
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
        if self.training:
            assert label is not None
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

# 🔥 Headless 지원이 추가된 CCNet
class ccnet(torch.nn.Module):
    '''
    CompNet = CB1//CB2//CB3 + FC + Dropout + (angular_margin) Output
    
    🔥 NEW: Headless Mode Support
    - headless_mode=False: 기존과 완전 동일 (100% 호환)
    - headless_mode=True: classification head 제거, metric verification
    '''

    def __init__(self, num_classes, weight, headless_mode=False):
        super(ccnet, self).__init__()

        self.num_classes = num_classes
        self.headless_mode = headless_mode

        # 🔥 Core Feature Extraction (항상 동일)
        self.cb1 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=35, stride=3, padding=17, init_ratio=1, weight=weight)
        self.cb2 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=36, ksize=17, stride=3, padding=8, init_ratio=0.5, o2=24, weight=weight)
        self.cb3 = CompetitiveBlock_Mul_Ord_Comp(channel_in=1, n_competitor=9, ksize=7, stride=3, padding=3, init_ratio=0.25, weight=weight)

        # Feature fusion layers (항상 동일)
        self.fc = torch.nn.Linear(13152, 4096)
        self.fc1 = torch.nn.Linear(4096, 2048)
        self.drop = torch.nn.Dropout(p=0.5)
        
        # 🔥 Classification Head (headless_mode에 따라 조건적 생성)
        if not headless_mode:
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            print(f"[CCNet] Initialized with classification head (classes: {num_classes})")
        else:
            self.arclayer_ = None
            print(f"[CCNet] Initialized in HEADLESS mode (no classification head)")

    def forward(self, x, y=None):
        """
        Forward pass with headless support
        
        🔥 Return format:
        - headless_mode=False: (logits, features) - 기존과 동일
        - headless_mode=True: (None, features) - logits 없음
        """
        # Feature extraction (기존과 완전 동일)
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x = torch.cat((x1, x2, x3), dim=1)

        x1 = self.fc(x)
        x = self.fc1(x1)
        fe = torch.cat((x1, x), dim=1)  # 6144 dimensional features
        
        # 🔥 Headless vs Normal mode
        if self.headless_mode:
            # Headless: classification head 없음, features만 반환
            return None, F.normalize(fe, dim=-1)
        else:
            # Normal: 기존과 완전 동일
            x = self.drop(x)
            x = self.arclayer_(x, y)
            return x, F.normalize(fe, dim=-1)

    def getFeatureCode(self, x):
        """
        특징 추출 전용 메서드 (기존과 완전 동일)
        headless/normal 모드 관계없이 동일하게 작동
        """
        x1 = self.cb1(x)
        x2 = self.cb2(x)
        x3 = self.cb3(x)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        x3 = x3.view(x3.shape[0], -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(x)
        x = self.fc1(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x
    
    def convert_to_headless(self):
        """
        런타임에 classification head를 제거하는 메서드
        온라인 학습 중에 동적으로 변경 가능
        """
        if not self.headless_mode:
            print("[CCNet] 🔪 Converting to headless mode...")
            self.arclayer_ = None
            self.headless_mode = True
            print("[CCNet] ✅ Classification head removed successfully")
            return True
        else:
            print("[CCNet] ⚠️ Already in headless mode")
            return False
    
    def convert_to_classification(self, num_classes=None):
        """
        런타임에 classification head를 추가하는 메서드
        """
        if self.headless_mode:
            if num_classes is None:
                num_classes = self.num_classes
            
            print(f"[CCNet] 🔧 Converting to classification mode (classes: {num_classes})...")
            self.arclayer_ = ArcMarginProduct(2048, num_classes, s=30, m=0.5, easy_margin=False)
            self.headless_mode = False
            self.num_classes = num_classes
            print("[CCNet] ✅ Classification head added successfully")
            return True
        else:
            print("[CCNet] ⚠️ Already in classification mode")
            return False
    
    def is_headless(self):
        """현재 headless 모드인지 확인"""
        return self.headless_mode
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'architecture': 'CCNet',
            'headless_mode': self.headless_mode,
            'num_classes': self.num_classes if not self.headless_mode else None,
            'feature_dimension': 2048,
            'has_classification_head': self.arclayer_ is not None,
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

# 🔥 Headless 모드를 위한 메트릭 기반 검증기
class HeadlessVerifier:
    """
    Headless 모드에서 사용하는 메트릭 기반 검증기
    Classification head 없이 특징 간 유사도로 인증 수행
    """
    def __init__(self, metric_type="cosine", threshold=0.5):
        self.metric_type = metric_type
        self.threshold = threshold
        print(f"[Verifier] Initialized metric-based verifier: {metric_type}, threshold: {threshold}")
    
    def compute_similarity(self, probe_features, gallery_features):
        """특징 간 유사도 계산"""
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
    
    def verify(self, probe_features, gallery_features):
        """메트릭 기반 인증 수행"""
        similarities = self.compute_similarity(probe_features, gallery_features)
        
        best_similarity = similarities.max().item()
        best_index = similarities.argmax().item()
        is_match = best_similarity > self.threshold
        
        return is_match, best_similarity, best_index

# 🔥 Config에 따른 모델 생성 팩토리 함수
def create_ccnet_from_config(config):
    """
    Config 설정에 따라 CCNet 모델 생성
    
    Args:
        config: PalmRecognizerConfig 객체
        
    Returns:
        ccnet 모델 인스턴스
    """
    headless_mode = getattr(config, 'headless_mode', False)
    
    model = ccnet(
        num_classes=config.num_classes,
        weight=config.com_weight,
        headless_mode=headless_mode
    )
    
    print(f"[Factory] Created CCNet with headless_mode={headless_mode}")
    return model

if __name__ == "__main__":
    print("🔧 Testing CCNet with Headless Support")
    
    # 기존 방식 테스트 (100% 호환)
    print("\n--- 기존 방식 테스트 (호환성 확인) ---")
    inp = torch.randn(2, 1, 128, 128)
    net_original = ccnet(600, weight=0.8)  # headless_mode 생략 = False
    out, features = net_original(inp)
    print(f"기존 방식 - Logits: {out.shape}, Features: {features.shape}")
    
    # Headless 방식 테스트
    print("\n--- Headless 방식 테스트 ---")
    net_headless = ccnet(600, weight=0.8, headless_mode=True)
    out_headless, features_headless = net_headless(inp)
    print(f"Headless 방식 - Logits: {out_headless}, Features: {features_headless.shape}")
    
    # 런타임 변환 테스트
    print("\n--- 런타임 변환 테스트 ---")
    net_convert = ccnet(600, weight=0.8, headless_mode=False)
    print(f"변환 전: headless={net_convert.is_headless()}")
    
    net_convert.convert_to_headless()
    print(f"변환 후: headless={net_convert.is_headless()}")
    
    out_converted, _ = net_convert(inp)
    print(f"변환 후 출력: {out_converted}")
    
    # Config 팩토리 테스트
    print("\n--- Config 팩토리 테스트 ---")
    class MockConfig:
        def __init__(self, headless_mode):
            self.num_classes = 600
            self.com_weight = 0.8
            self.headless_mode = headless_mode
    
    config_normal = MockConfig(headless_mode=False)
    config_headless = MockConfig(headless_mode=True)
    
    model_normal = create_ccnet_from_config(config_normal)
    model_headless = create_ccnet_from_config(config_headless)
    
    print(f"Config 생성 - Normal: {model_normal.is_headless()}")
    print(f"Config 생성 - Headless: {model_headless.is_headless()}")
