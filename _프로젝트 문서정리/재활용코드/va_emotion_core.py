import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List, Dict
import os

class VAEmotionCore:
    """
    VA 감정 분석 핵심 기능만 포함
    멀티모달 비디오 프로세서에서 사용할 핵심 클래스
    """
    
    def __init__(self, model_path: str = "./models/affectnet_emotions/enet_b0_8_va_mtl.pt", 
                 device: str = None):
        """
        VA 감정 모델 초기화
        
        Args:
            model_path (str): 모델 파일 경로
            device (str): 연산 장치 ('cpu', 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 감정 레이블 (실제 모델 순서)
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
        
        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.model = self._load_model(model_path)
        print(f"✅ VA 감정 모델 로드 완료 (디바이스: {self.device})")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.eval()
        model.to(self.device)
        return model
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """이미지 전처리 (PIL Image 또는 numpy 배열)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"지원하지 않는 이미지 타입: {type(image)}")
        
        # RGB 변환 및 전처리
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0)  # 배치 차원 추가
        return tensor.to(self.device)
    
    def extract_emotion_features(self, face_image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        얼굴 이미지에서 10차원 감정 특징 추출
        
        Args:
            face_image: 224x224 크롭된 얼굴 이미지
            
        Returns:
            np.ndarray: 10차원 벡터 [8개 감정 + Valence + Arousal]
        """
        try:
            # 이미지 전처리
            input_tensor = self._preprocess_image(face_image)
            
            # 모델 추론
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output.cpu().numpy().flatten()
            
            return output  # 10차원 벡터 반환
            
        except Exception as e:
            print(f"⚠️ 감정 특징 추출 실패: {e}")
            return np.zeros(10)  # 실패 시 0벡터 반환
    
    def extract_emotion_features_batch(self, face_images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """
        배치 얼굴 이미지에서 감정 특징 추출
        
        Args:
            face_images: 얼굴 이미지 리스트
            
        Returns:
            np.ndarray: [batch_size, 10] 형태의 특징 배열
        """
        if not face_images:
            return np.empty((0, 10))
        
        try:
            # 배치 전처리
            batch_tensors = []
            for img in face_images:
                tensor = self._preprocess_image(img)
                batch_tensors.append(tensor.squeeze(0))  # 배치 차원 제거
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # 배치 추론
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                outputs = outputs.cpu().numpy()
            
            return outputs  # [batch_size, 10] 반환
            
        except Exception as e:
            print(f"⚠️ 배치 감정 특징 추출 실패: {e}")
            return np.zeros((len(face_images), 10))
    
    def get_emotion_interpretation(self, emotion_vector: np.ndarray) -> Dict:
        """
        10차원 감정 벡터 해석
        
        Args:
            emotion_vector: 10차원 감정 벡터
            
        Returns:
            Dict: 해석된 감정 정보
        """
        if len(emotion_vector) != 10:
            raise ValueError(f"10차원 벡터가 필요합니다. 입력 차원: {len(emotion_vector)}")
        
        # 8개 감정 + 2개 VA 분리
        emotions = emotion_vector[:8]
        valence = emotion_vector[8]
        arousal = emotion_vector[9]
        
        # 감정 확률 계산
        emotion_probs = F.softmax(torch.tensor(emotions), dim=0).numpy()
        predicted_emotion = self.emotion_labels[np.argmax(emotion_probs)]
        confidence = float(np.max(emotion_probs))
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'emotion_probs': {label: float(prob) for label, prob in zip(self.emotion_labels, emotion_probs)},
            'valence': float(valence),
            'arousal': float(arousal),
            'raw_vector': emotion_vector.tolist()
        }


# 사용 예시 및 테스트 함수
def test_va_core():
    """VA 감정 코어 테스트"""
    try:
        # 모델 초기화
        va_core = VAEmotionCore()
        
        # 테스트 이미지 로드
        test_image_path = "data/input"
        if os.path.exists(test_image_path):
            image_files = [f for f in os.listdir(test_image_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                # 첫 번째 이미지로 테스트
                img_path = os.path.join(test_image_path, image_files[0])
                test_img = Image.open(img_path)
                
                # 감정 특징 추출
                emotion_vector = va_core.extract_emotion_features(test_img)
                print(f"감정 벡터 shape: {emotion_vector.shape}")
                print(f"감정 벡터: {emotion_vector}")
                
                # 해석
                result = va_core.get_emotion_interpretation(emotion_vector)
                print(f"예측 감정: {result['predicted_emotion']} (신뢰도: {result['confidence']:.3f})")
                print(f"Valence: {result['valence']:.3f}, Arousal: {result['arousal']:.3f}")
                
                return True
        
        print("⚠️ 테스트 이미지를 찾을 수 없습니다")
        return False
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    test_va_core()