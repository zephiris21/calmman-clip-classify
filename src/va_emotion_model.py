import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import Union, List, Dict, Optional
import logging

class VAEmotionModel:
    """
    VA-MTL (Valence-Arousal Multi-Task Learning) 감정 인식 모델
    EfficientNet-B0 기반, AffectNet 데이터셋으로 훈련됨
    """
    
    def __init__(self, model_path: str = "models/affectnet_emotions/enet_b0_8_va_mtl.pt", 
                 device: Optional[str] = None):
        """
        모델 초기화
        
        Args:
            model_path (str): 모델 파일 경로
            device (str, optional): 연산 장치 ('cpu', 'cuda'). None이면 자동 감지
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 감정 레이블 정의 (실제 모델 순서에 맞춘 AffectNet 8개 감정)
        self.emotion_labels = [
            'Anger',       # 0
            'Contempt',    # 1
            'Disgust',     # 2
            'Fear',        # 3
            'Happiness',   # 4
            'Neutral',     # 5
            'Sadness',     # 6
            'Surprise'     # 7
        ]
        
        # 이미지 전처리 파이프라인 (ImageNet 정규화)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.model = self._load_model()
        self.feature_extractor = self._create_feature_extractor()
        
        print(f"✅ VA 감정 모델 로드 완료 (디바이스: {self.device})")
    
    def _load_model(self) -> nn.Module:
        """전체 모델 로드 (감정 분류용) - timm 호환성 개선"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        try:
            # 방법 1: 전체 모델 객체 로드 시도
            print("📦 모델 로드 시도 중...")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 전체 모델인지 확인
            if hasattr(checkpoint, 'eval') and hasattr(checkpoint, 'forward'):
                print("✅ 전체 모델 감지 - 직접 사용")
                model = checkpoint.to(self.device)
                model.eval()
                
                # 출력 차원 확인
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    output = model(dummy_input)
                    print(f"✅ 모델 출력 차원: {output.shape}")
                    assert output.shape[1] == 10, f"예상 출력 차원: 10, 실제: {output.shape[1]}"
                
                return model
            
            elif isinstance(checkpoint, dict):
                # state_dict인 경우 - timm으로 재구성
                print("✅ state_dict 감지 - timm으로 모델 재구성")
                return self._load_from_state_dict(checkpoint)
            
            else:
                raise ValueError(f"알 수 없는 모델 형태: {type(checkpoint)}")
            
        except Exception as e:
            print(f"⚠️ 원본 로드 실패: {e}")
            print("🔄 대안 로드 방식 시도...")
            return self._load_fallback()
    
    def _load_from_state_dict(self, state_dict: dict) -> nn.Module:
        """state_dict에서 모델 재구성"""
        try:
            import timm
            print(f"📦 timm 버전: {timm.__version__}")
            
            # EfficientNet-B0 기반 모델 생성 (10클래스)
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=10)
            
            # state_dict 로드 (호환성 문제 우회)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            
            print("✅ timm 기반 모델 재구성 성공")
            return model
            
        except Exception as e:
            raise RuntimeError(f"timm 기반 재구성 실패: {e}")
    
    def _load_fallback(self) -> nn.Module:
        """대안 로드 방식 - 더 관대한 설정"""
        try:
            # 더 관대한 로드 시도
            import pickle
            
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if hasattr(checkpoint, 'eval'):
                model = checkpoint.to(self.device)
                model.eval()
                print("✅ pickle 기반 로드 성공")
                return model
            
            raise RuntimeError("대안 로드 방식도 실패")
            
        except Exception as e:
            raise RuntimeError(f"모든 로드 방식 실패: {e}")
    
    def _create_feature_extractor(self) -> nn.Module:
        """특징 추출기 생성 (분류기 제거)"""
        try:
            import timm
            
            # EfficientNet-B0 백본만 로드 (num_classes=0으로 분류기 제거)
            feature_extractor = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
            
            # 훈련된 가중치 로드 (분류기 제외)
            model_state = self.model.state_dict()
            feature_state = {}
            
            for name, param in feature_extractor.state_dict().items():
                if name in model_state:
                    feature_state[name] = model_state[name]
                else:
                    print(f"⚠️ 가중치 누락: {name}")
            
            feature_extractor.load_state_dict(feature_state, strict=False)
            feature_extractor.eval()
            feature_extractor.to(self.device)
            
            # 특징 차원 확인
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                features = feature_extractor(dummy_input)
                assert features.shape[1] == 1280, f"예상 특징 차원: 1280, 실제: {features.shape[1]}"
            
            return feature_extractor
            
        except Exception as e:
            print(f"⚠️ 특징 추출기 생성 실패: {e}")
            return None
    
    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """이미지 전처리"""
        if isinstance(image, str):
            # 파일 경로인 경우
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # numpy 배열인 경우
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        # 전처리 적용
        tensor = self.transform(image).unsqueeze(0)  # 배치 차원 추가
        return tensor.to(self.device)
    
    def predict_emotions(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        감정 분류 예측
        
        Args:
            image: 입력 이미지 (파일 경로, PIL Image, 또는 numpy 배열)
            
        Returns:
            Dict: 감정 분류 결과
            {
                'emotions': [8개 감정 확률],
                'emotion_probs': {감정명: 확률},
                'predicted_emotion': '가장 높은 감정',
                'valence': float,
                'arousal': float,
                'raw_output': [10차원 원본 출력]
            }
        """
        try:
            # 이미지 전처리
            input_tensor = self._preprocess_image(image)
            
            # 모델 추론
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.cpu().numpy().flatten()
            
            # 8개 감정 + 2개 VA 분리
            emotions = output[:8]  # 감정 로짓
            valence = output[8]    # Valence
            arousal = output[9]    # Arousal
            
            # 감정 확률 계산 (소프트맥스)
            emotion_probs = F.softmax(torch.tensor(emotions), dim=0).numpy()
            
            # 결과 딕셔너리 생성
            emotion_dict = {emotion: float(prob) for emotion, prob in zip(self.emotion_labels, emotion_probs)}
            predicted_emotion = self.emotion_labels[np.argmax(emotion_probs)]
            
            return {
                'emotions': emotion_probs.tolist(),
                'emotion_probs': emotion_dict,
                'predicted_emotion': predicted_emotion,
                'valence': float(valence),
                'arousal': float(arousal),
                'raw_output': output.tolist()
            }
            
        except Exception as e:
            raise RuntimeError(f"감정 예측 실패: {e}")
    
    def extract_features(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        1280차원 특징 벡터 추출
        
        Args:
            image: 입력 이미지
            
        Returns:
            np.ndarray: 1280차원 특징 벡터
        """
        if self.feature_extractor is None:
            raise RuntimeError("특징 추출기가 초기화되지 않았습니다")
        
        try:
            # 이미지 전처리
            input_tensor = self._preprocess_image(image)
            
            # 특징 추출
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.cpu().numpy().flatten()
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"특징 추출 실패: {e}")
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]], 
                     mode: str = 'emotions') -> List[Dict]:
        """
        배치 예측
        
        Args:
            images: 이미지 리스트
            mode: 'emotions' (감정 분류) 또는 'features' (특징 추출)
            
        Returns:
            List[Dict]: 예측 결과 리스트
        """
        results = []
        
        for image in images:
            try:
                if mode == 'emotions':
                    result = self.predict_emotions(image)
                elif mode == 'features':
                    features = self.extract_features(image)
                    result = {'features': features.tolist(), 'feature_dim': len(features)}
                else:
                    raise ValueError(f"지원하지 않는 모드: {mode}")
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ 배치 처리 오류 (이미지 {len(results)}): {e}")
                results.append({'error': str(e)})
        
        return results
    
    def analyze_directory(self, directory_path: str, mode: str = 'emotions') -> Dict:
        """
        디렉토리 내 모든 이미지 분석
        
        Args:
            directory_path: 이미지 디렉토리 경로
            mode: 'emotions' 또는 'features'
            
        Returns:
            Dict: 분석 결과 요약
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory_path}")
        
        # 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(directory_path, file))
        
        if not image_files:
            return {'message': '이미지 파일을 찾을 수 없습니다', 'results': []}
        
        print(f"📁 {len(image_files)}개 이미지 분석 시작...")
        
        # 배치 처리
        results = self.predict_batch(image_files, mode=mode)
        
        # 요약 생성
        summary = {
            'total_images': len(image_files),
            'successful': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'results': [
                {
                    'filename': os.path.basename(image_files[i]),
                    'result': results[i]
                }
                for i in range(len(image_files))
            ]
        }
        
        if mode == 'emotions':
            # 감정 통계 추가
            successful_results = [r for r in results if 'error' not in r]
            if successful_results:
                emotion_counts = {}
                for result in successful_results:
                    emotion = result['predicted_emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                summary['emotion_statistics'] = emotion_counts
        
        print(f"✅ 분석 완료: {summary['successful']}/{summary['total_images']} 성공")
        
        return summary


def test_va_model():
    """테스트 함수"""
    print("🧪 VA 감정 모델 테스트 시작")
    
    try:
        # 모델 초기화
        va_model = VAEmotionModel()
        
        # data/input 디렉토리 분석
        input_dir = "data/processed/test_image"
        if os.path.exists(input_dir):
            print(f"\n📁 {input_dir} 디렉토리 분석...")
            results = va_model.analyze_directory(input_dir, mode='emotions')
            
            # 결과 출력
            print(f"\n📊 분석 결과:")
            print(f"   총 이미지: {results['total_images']}개")
            print(f"   성공: {results['successful']}개")
            print(f"   실패: {results['failed']}개")
            
            if 'emotion_statistics' in results:
                print(f"\n😊 감정 분포:")
                for emotion, count in results['emotion_statistics'].items():
                    print(f"   {emotion}: {count}개")
            
            # 개별 결과 출력 (처음 3개만)
            print(f"\n🔍 상세 결과 (처음 3개):")
            for i, item in enumerate(results['results'][:3]):
                filename = item['filename']
                result = item['result']
                
                if 'error' not in result:
                    emotion = result['predicted_emotion']
                    confidence = max(result['emotions'])
                    valence = result['valence']
                    arousal = result['arousal']
                    
                    print(f"   {filename}:")
                    print(f"     감정: {emotion} (신뢰도: {confidence:.3f})")
                    print(f"     Valence: {valence:.3f}, Arousal: {arousal:.3f}")
                    
                    # 8개 감정 확률 모두 출력
                    print(f"     8개 감정 확률:")
                    for emotion_name, prob in result['emotion_probs'].items():
                        print(f"       {emotion_name}: {prob:.3f}")
                    print()
                else:
                    print(f"   {filename}: 오류 - {result['error']}")
        
        else:
            print(f"⚠️ {input_dir} 디렉토리가 없습니다")
            print("테스트용 이미지를 data/input/ 폴더에 넣어주세요")
    
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    test_va_model()