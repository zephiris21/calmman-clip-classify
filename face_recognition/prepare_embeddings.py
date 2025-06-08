#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from mtcnn_wrapper import FaceDetector


class FaceEmbeddingGenerator:
    """얼굴 임베딩 생성기"""
    
    def __init__(self, config_path: str = "config_recog.yaml"):
        """
        초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # 디바이스 설정
        self.device = torch.device(
            self.config['face_recognition']['mtcnn']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        
        self.logger.info(f"디바이스: {self.device}")
        
        # 모델 초기화
        self._init_models()
        
        # 출력 디렉토리 생성
        self._create_output_dirs()
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            # 상대경로면 스크립트 디렉토리 기준으로 변환
            if not os.path.isabs(config_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, config_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        self.logger = logging.getLogger('FaceEmbeddingGenerator')
        self.logger.setLevel(getattr(logging, self.config['face_recognition']['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (옵션)
        if self.config['face_recognition']['logging']['save_logs']:
            # 스크립트 디렉토리 기준으로 로그 디렉토리 경로 설정
            script_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(script_dir, 'logs')
            
            # 로그 디렉토리가 없으면 생성
            os.makedirs(logs_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"embedding_generation_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"로그 파일: {log_file}")
    
    def _init_models(self):
        """모델 초기화"""
        mtcnn_config = self.config['face_recognition']['mtcnn']
        facenet_config = self.config['face_recognition']['facenet']
        
        # MTCNN 얼굴 탐지기 (기존 wrapper 활용)
        self.face_detector = FaceDetector(
            image_size=mtcnn_config['image_size'],
            margin=mtcnn_config['margin'],
            prob_threshold=mtcnn_config['prob_threshold'],
            align_faces=mtcnn_config['align_faces'],
            device=self.device
        )
        
        # FaceNet 모델 (facenet-pytorch 직접 사용)
        self.facenet_model = InceptionResnetV1(
            pretrained=facenet_config['model_name']
        ).eval().to(self.device)
        
        self.logger.info("✅ 모델 초기화 완료")
    
    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        # 스크립트 디렉토리 기준으로 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 설정에서 상대경로 가져와서 절대경로로 변환
        embeddings_relative = self.config['face_recognition']['paths']['target_embeddings']
        embeddings_dir = os.path.join(script_dir, embeddings_relative.lstrip('./'))
        
        os.makedirs(embeddings_dir, exist_ok=True)
        self.logger.info(f"임베딩 저장 디렉토리: {embeddings_dir}")
        
        # 절대경로로 업데이트 (이후 사용을 위해)
        self.embeddings_dir = embeddings_dir
    
    def _get_face_embedding(self, face_image: Image.Image) -> Optional[np.ndarray]:
        """
        단일 얼굴 이미지에서 임베딩 추출
        
        Args:
            face_image (PIL.Image): 얼굴 이미지 (160x160 RGB)
            
        Returns:
            np.ndarray: 정규화된 임베딩 벡터 (512차원)
        """
        try:
            # PIL Image를 텐서로 변환 (facenet-pytorch 표준 전처리)
            img_array = np.array(face_image)
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            img_tensor = (img_tensor - 127.5) / 128.0  # 정규화 [-1, 1]
            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # 배치 차원 추가
            
            # 임베딩 추출
            with torch.no_grad():
                embedding = self.facenet_model(img_tensor)
                embedding = embedding.cpu().numpy().squeeze()
            
            # L2 정규화
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"임베딩 추출 실패: {e}")
            return None
    
    def generate_embeddings_for_person(self, person_name: str) -> bool:
        """
        특정 인물의 임베딩 생성
        
        Args:
            person_name (str): 인물 이름 (폴더명)
            
        Returns:
            bool: 성공 여부
        """
        # 스크립트 디렉토리 기준으로 입력 이미지 폴더 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_relative = self.config['face_recognition']['paths']['reference_images']
        images_base_dir = os.path.join(script_dir, images_relative.lstrip('./'))
        images_dir = os.path.join(images_base_dir, person_name)
        
        if not os.path.exists(images_dir):
            self.logger.error(f"이미지 폴더가 존재하지 않습니다: {images_dir}")
            return False
        
        # 이미지 파일 목록
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*.{ext}"))
            image_files.extend(Path(images_dir).glob(f"*.{ext.upper()}"))

        # 중복 제거
        image_files = list(dict.fromkeys(image_files))  
        
        if not image_files:
            self.logger.error(f"이미지 파일을 찾을 수 없습니다: {images_dir}")
            return False
        
        self.logger.info(f"📸 {person_name} 처리 시작 ({len(image_files)}개 이미지)")
        
        # 각 이미지에서 얼굴 추출 및 임베딩 계산
        valid_embeddings = []
        
        for img_path in image_files:
            try:
                self.logger.info(f"  처리 중: {img_path.name}")
                
                # MTCNN wrapper를 사용해 얼굴 탐지 및 전처리
                face_images = self.face_detector.process_image(str(img_path))
                
                if not face_images:
                    self.logger.warning(f"    얼굴을 찾을 수 없습니다: {img_path.name}")
                    continue
                
                # 첫 번째 얼굴만 사용 (가장 큰 얼굴)
                face_image = face_images[0]  # 이미 160x160으로 크롭/정렬된 상태
                
                # 임베딩 추출
                embedding = self._get_face_embedding(face_image)
                
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    self.logger.info(f"    ✅ 임베딩 추출 성공")
                else:
                    self.logger.warning(f"    ❌ 임베딩 추출 실패")
                    
            except Exception as e:
                self.logger.error(f"    ❌ 이미지 처리 실패 {img_path.name}: {e}")
        
        # 유효한 임베딩이 있는지 확인
        if not valid_embeddings:
            self.logger.error(f"유효한 임베딩을 생성할 수 없습니다: {person_name}")
            return False
        
        # 평균 임베딩 계산
        mean_embedding = np.mean(valid_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)  # 재정규화
        
        # 임베딩 저장 (이미 _create_output_dirs에서 설정된 절대경로 사용)
        output_path = os.path.join(self.embeddings_dir, f"{person_name}.npy")
        
        # 메타데이터와 함께 저장
        embedding_data = {
            'embedding': mean_embedding,
            'person_name': person_name,
            'num_images': len(valid_embeddings),
            'total_images': len(image_files),
            'generated_at': datetime.now().isoformat(),
            'config': self.config['face_recognition']
        }
        
        np.save(output_path, embedding_data)
        
        self.logger.info(f"✅ {person_name} 임베딩 생성 완료")
        self.logger.info(f"   사용된 이미지: {len(valid_embeddings)}/{len(image_files)}개")
        self.logger.info(f"   저장 경로: {output_path}")
        
        return True
    
    def generate_all_embeddings(self) -> Dict[str, bool]:
        """
        모든 인물의 임베딩 생성
        
        Returns:
            Dict[str, bool]: 각 인물별 성공 여부
        """
        # 스크립트 디렉토리 기준으로 이미지 베이스 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_relative = self.config['face_recognition']['paths']['reference_images']
        images_base_dir = os.path.join(script_dir, images_relative.lstrip('./'))
        
        if not os.path.exists(images_base_dir):
            self.logger.error(f"참조 이미지 디렉토리가 존재하지 않습니다: {images_base_dir}")
            return {}
        
        # 인물별 폴더 목록
        person_folders = [
            d for d in os.listdir(images_base_dir) 
            if os.path.isdir(os.path.join(images_base_dir, d))
        ]
        
        if not person_folders:
            self.logger.error(f"인물 폴더를 찾을 수 없습니다: {images_base_dir}")
            return {}
        
        self.logger.info(f"🎯 전체 임베딩 생성 시작 ({len(person_folders)}명)")
        
        results = {}
        for person_name in person_folders:
            self.logger.info(f"{'='*50}")
            success = self.generate_embeddings_for_person(person_name)
            results[person_name] = success
        
        # 결과 요약
        self.logger.info(f"{'='*50}")
        self.logger.info("🎉 전체 처리 완료!")
        
        successful = sum(results.values())
        total = len(results)
        
        self.logger.info(f"성공: {successful}/{total}명")
        
        for person, success in results.items():
            status = "✅" if success else "❌"
            self.logger.info(f"  {status} {person}")
        
        return results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='얼굴 임베딩 생성기')
    parser.add_argument('--person', help='특정 인물만 처리 (폴더명)')
    parser.add_argument('--config', default='config_recog.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        # 생성기 초기화
        generator = FaceEmbeddingGenerator(args.config)
        
        if args.person:
            # 특정 인물만 처리
            generator.logger.info(f"🎯 {args.person} 처리 시작")
            success = generator.generate_embeddings_for_person(args.person)
            
            if success:
                print(f"✅ {args.person} 임베딩 생성 완료!")
            else:
                print(f"❌ {args.person} 임베딩 생성 실패!")
        else:
            # 모든 인물 처리
            results = generator.generate_all_embeddings()
            
            successful = sum(results.values())
            total = len(results)
            
            print(f"🎉 전체 처리 완료! 성공: {successful}/{total}명")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()