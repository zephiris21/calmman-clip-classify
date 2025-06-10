#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
from PIL import Image


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 기반 이진분류 모델 (학습 시와 동일한 구조)"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        
        # timm에서 EfficientNet-B0 로드 (학습 시와 동일)
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained,
            num_classes=0,  # 분류 헤드 제거
            drop_rate=dropout_rate
        )
        
        # 특징 차원 얻기
        self.feature_dim = self.backbone.num_features
        
        # 커스텀 분류 헤드 (학습 시와 동일)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x):
        # 백본을 통과하여 특징 추출
        features = self.backbone(x)
        # 분류 헤드를 통과
        outputs = self.classifier(features)
        return outputs


class TorchFacialClassifier:
    """PyTorch 기반 얼굴 표정 분류기"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device(config['classifier']['device'] 
                                  if torch.cuda.is_available() 
                                  else 'cpu')
        
        self.logger.info(f"디바이스: {self.device}")
        
        # 모델 로드
        self.model = self._load_model()
        
        # 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 배치 처리 설정
        self.batch_size = config['classifier']['batch_size']
        self.batch_timeout = config['classifier']['batch_timeout']
        
        self.logger.info(f"분류기 초기화 완료 (배치: {self.batch_size}, 타임아웃: {self.batch_timeout}초)")
    
    def _load_model(self) -> nn.Module:
        """모델 로드"""
        model_dir = self.config['classifier']['model_path']
        
        # 상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)
        if not os.path.isabs(model_dir):
            # 현재 스크립트 위치에서 상위로 올라가서 프로젝트 루트 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # src의 상위 = 프로젝트 루트
            model_dir = os.path.join(project_root, model_dir)
            model_dir = os.path.normpath(model_dir)
        
        self.logger.info(f"모델 디렉토리 경로: {model_dir}")
        
        # 모델 인스턴스 생성 (학습 시와 동일한 파라미터)
        model = EfficientNetClassifier(
            num_classes=2, 
            pretrained=False,  # 가중치는 저장된 것에서 로드
            dropout_rate=0.3   # 학습 시와 동일한 dropout
        )
        model = model.to(self.device)
        
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            
            if model_files:
                # stage2 모델 우선 선택
                stage2_models = [f for f in model_files if 'stage2' in f]
                if stage2_models:
                    latest_model = sorted(stage2_models)[-1]
                else:
                    latest_model = sorted(model_files)[-1]
                
                model_full_path = os.path.join(model_dir, latest_model)
                self.logger.info(f"모델 로드: {model_full_path}")
                
                model.load_state_dict(torch.load(model_full_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_dir}")
        else:
            raise FileNotFoundError(f"모델 디렉토리가 존재하지 않습니다: {model_dir}")
        
        model.eval()
        return model
    
    def preprocess_batch(self, face_images: List[Image.Image]) -> torch.Tensor:
        """이미지 배치 전처리"""
        batch_tensors = []
        
        for img in face_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            tensor = self.transform(img)
            batch_tensors.append(tensor)
        
        return torch.stack(batch_tensors).to(self.device)
    
    def predict_batch(self, face_images: List[Image.Image]) -> List[Dict]:
        """배치 예측"""
        if not face_images:
            return []
        
        try:
            # 전처리
            batch_tensor = self.preprocess_batch(face_images)
            
            # 추론
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(batch_tensor)
                inference_time = time.time() - start_time
                
                # 소프트맥스로 확률 변환
                probabilities = torch.softmax(outputs, dim=1)
                
                # 킹받는 확률 (클래스 1)
                angry_probs = probabilities[:, 1].cpu().numpy()
            
            # 결과 구성
            results = []
            for i, prob in enumerate(angry_probs):
                results.append({
                    'confidence': float(prob),
                    'is_angry': prob > self.config['classifier']['confidence_threshold']
                })
            
            # 로깅
            if self.config['logging']['batch_summary']:
                angry_count = sum(1 for r in results if r['is_angry'])
                avg_confidence = np.mean([r['confidence'] for r in results])
                
                self.logger.info(
                    f"분류 배치: {len(face_images)}개 → 킹받음 {angry_count}개 "
                    f"(평균 신뢰도: {avg_confidence:.3f}, {inference_time:.3f}초)"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"배치 예측 오류: {e}")
            return [{'confidence': 0.0, 'is_angry': False} for _ in face_images]
    
    def get_memory_usage(self) -> Dict:
        """GPU 메모리 사용량 조회"""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}