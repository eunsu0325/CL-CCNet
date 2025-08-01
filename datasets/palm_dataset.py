# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')


        c,h,w = tensor.size()
   
        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h*w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std() 
        t = t.sub_(m).div_(s+1e-6)
        tensor[idx] = t
        
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats = self.outchannels, dim = 0)
    
        return tensor



class MyDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None 
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image
    return_raw: whether to return raw images for visualization

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''
    
    def __init__(self, txt, transforms=None, train=True, imside=128, outchannels=1, return_raw=False):        

        self.train = train
        self.imside = imside # 128, 224
        self.chs = outchannels # 1, 3
        self.return_raw = return_raw  # 🔥 FIX: 매개변수로 받아서 초기화
        self.text_path = txt        

        self.transforms = transforms

        if transforms is None:
            if not train: 
                self.transforms = T.Compose([ 
                                                        
                    T.Resize(self.imside),                  
                    T.ToTensor(),   
                    NormSingleROI(outchannels=self.chs)
                    
                    ]) 
            else:
                self.transforms = T.Compose([  
                                
                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),# 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8,1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),# (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, resample=Image.BICUBIC, expand=False, center=(0.5*self.imside, 0.0)),
                            T.RandomRotation(degrees=10, resample=Image.BICUBIC, expand=False, center=(0.0, 0.5*self.imside)),
                        ]),
                    ]),     

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)                   
                    ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 빈 줄 스킵
                continue
                
            try:
                item = line.split(' ')
                if len(item) >= 2:  # 최소 2개 요소 확인
                    self.images_path.append(item[0])
                    self.images_label.append(item[1])
                else:
                    print(f"[Dataset] Skipping malformed line {line_num}: '{line}'")
            except Exception as e:
                print(f"[Dataset] Error on line {line_num}: {e}")
                continue

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]
        
        # 같은 사람의 다른 이미지 찾기
        idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
        if self.train == True:
            while(idx2 == index):
                idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
        else:
            idx2 = index
            
        img_path2 = self.images_path[idx2]
        
        # 원본 이미지 로드
        data_raw = Image.open(img_path).convert('L')
        data2_raw = Image.open(img_path2).convert('L')
        
        # 변환 적용 (학습용)
        data = self.transforms(data_raw)
        data2 = self.transforms(data2_raw)
        
        # 원본 반환 옵션
        if self.return_raw:
            # 원본을 numpy 배열로 (시각화용)
            raw_np1 = np.array(data_raw)
            raw_np2 = np.array(data2_raw)
            return [data, data2], int(label), [raw_np1, raw_np2]
        else:
            return [data, data2], int(label)
    
    def __len__(self):
        return len(self.images_path)