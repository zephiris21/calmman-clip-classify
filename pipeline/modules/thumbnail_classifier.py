#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 및 분류기 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils
from src.pytorch_classifier import TorchFacialClassifier


class ThumbnailClassifier:
    """
    썸네일용 얼굴 분류기
    침착맨 얼굴 이미지들을 분류하여 썸네일용 과장된 표정을 찾아내는 클래스
    """
    
    def __init__(self, config: Dict):
        """
        썸네일 분류기 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
        self.thumbnail_config = config['thumbnail_classification']
        
        # PyTorch 분류기 초기화 (기존 TorchFacialClassifier 재활용)
        self.classifier = TorchFacialClassifier(self._adapt_config_for_classifier())
        
        print(f"✅ 썸네일 분류기 초기화 완료")
        print(f"   모델: {self.thumbnail_config['model_path']}")
        print(f"   신뢰도 임계값: {self.thumbnail_config['confidence_threshold']}")
        print(f"   최대 썸네일 수: {self.thumbnail_config['max_thumbnails']}")
    
    def _adapt_config_for_classifier(self) -> Dict:
        """
        기존 TorchFacialClassifier에 맞는 설정 형식으로 변환
        
        Returns:
            Dict: TorchFacialClassifier용 설정
        """
        return {
            'classifier': {
                'model_path': self.thumbnail_config['model_path'],
                'device': self.thumbnail_config['device'],
                'confidence_threshold': self.thumbnail_config['confidence_threshold'],
                'batch_size': self.thumbnail_config['batch_size'],
                'batch_timeout': 5.0  # 기본값 추가
            },
            'logging': {
                'batch_summary': True
            }
        }
    
    def classify_faces_from_directory(self, faces_dir: str, output_dir: str) -> Dict:
        """
        얼굴 디렉토리에서 썸네일용 얼굴 분류
        
        Args:
            faces_dir (str): 침착맨 얼굴 이미지 디렉토리
            output_dir (str): 썸네일 이미지 저장 디렉토리
            
        Returns:
            Dict: 분류 결과 정보
        """
        print(f"\n🎭 썸네일용 얼굴 분류 시작")
        print(f"   입력: {faces_dir}")
        print(f"   출력: {output_dir}")
        
        if not os.path.exists(faces_dir):
            raise ValueError(f"얼굴 이미지 디렉토리를 찾을 수 없습니다: {faces_dir}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 얼굴 이미지 파일들 찾기
        face_files = self._find_face_images(faces_dir)
        
        if not face_files:
            print("⚠️ 처리할 얼굴 이미지가 없습니다")
            return {'classified_count': 0, 'thumbnails': []}
        
        print(f"   총 얼굴 이미지: {len(face_files)}개")
        
        # 배치별로 분류 수행
        thumbnail_candidates = []
        batch_size = self.thumbnail_config['batch_size']
        
        for i in range(0, len(face_files), batch_size):
            batch_files = face_files[i:i + batch_size]
            batch_results = self._classify_face_batch(batch_files)
            thumbnail_candidates.extend(batch_results)
        
        # 신뢰도 순으로 정렬
        thumbnail_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 상위 N개 선택 및 저장
        max_thumbnails = self.thumbnail_config['max_thumbnails']
        selected_thumbnails = thumbnail_candidates[:max_thumbnails]
        
        saved_thumbnails = []
        for rank, thumbnail in enumerate(selected_thumbnails, 1):
            saved_info = self._save_thumbnail_image(thumbnail, rank, output_dir)
            if saved_info:
                saved_thumbnails.append(saved_info)
        
        print(f"\n✅ 썸네일 분류 완료: 총 {len(saved_thumbnails)}개 저장")
        
        return {
            'classified_count': len(face_files),
            'candidate_count': len(thumbnail_candidates),
            'saved_count': len(saved_thumbnails),
            'thumbnails': saved_thumbnails,
            'output_dir': output_dir
        }
    
    def classify_faces_from_hdf5(self, video_hdf5_path: str, output_dir: str) -> Dict:
        """
        HDF5 파일에서 얼굴 디렉토리를 찾아 분류
        
        Args:
            video_hdf5_path (str): 비디오 HDF5 파일 경로
            output_dir (str): 썸네일 이미지 저장 디렉토리
            
        Returns:
            Dict: 분류 결과 정보
        """
        print(f"\n🎭 HDF5에서 썸네일 분류 시작")
        print(f"   HDF5: {video_hdf5_path}")
        
        # 비디오 데이터 로드
        video_data = PipelineUtils.load_video_hdf5(video_hdf5_path)
        if video_data is None:
            raise ValueError(f"비디오 HDF5 로드 실패: {video_hdf5_path}")
        
        # 비디오별 하위 폴더 생성
        video_name = video_data['metadata']['video_name']
        timestamp = datetime.now().strftime("%Y%m%d")
        video_folder = f"{video_name}_{timestamp}"
        video_output_dir = os.path.join(output_dir, video_folder)
        
        print(f"   비디오별 출력: {video_output_dir}")
        
        # 침착맨 얼굴 디렉토리 찾기
        chimchakman_faces_dir = PipelineUtils.get_chimchakman_faces_directory(video_data)
        if not chimchakman_faces_dir:
            raise ValueError("침착맨 얼굴 이미지 디렉토리를 찾을 수 없습니다")
        
        print(f"   얼굴 디렉토리: {chimchakman_faces_dir}")
        
        # 디렉토리에서 분류 수행 (video_output_dir 사용)
        return self.classify_faces_from_directory(chimchakman_faces_dir, video_output_dir)
    
    def _find_face_images(self, faces_dir: str) -> List[str]:
        """
        얼굴 이미지 파일들 찾기
        
        Args:
            faces_dir (str): 얼굴 이미지 디렉토리
            
        Returns:
            List[str]: 얼굴 이미지 파일 경로 리스트
        """
        # 지원하는 이미지 확장자
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        face_files = []
        for ext in extensions:
            pattern = os.path.join(faces_dir, ext)
            face_files.extend(glob.glob(pattern))
        
        # 파일명 기준 정렬 (타임스탬프 순서)
        face_files.sort()
        
        return face_files
    
    def _classify_face_batch(self, face_files: List[str]) -> List[Dict]:
        """
        얼굴 이미지 배치 분류
        
        Args:
            face_files (List[str]): 얼굴 이미지 파일 경로 리스트
            
        Returns:
            List[Dict]: 썸네일 후보 정보 리스트
        """
        try:
            # 이미지 로드
            face_images = []
            valid_files = []
            
            for file_path in face_files:
                try:
                    with Image.open(file_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        face_images.append(img.copy())
                        valid_files.append(file_path)
                except Exception as e:
                    print(f"   ⚠️ 이미지 로드 실패: {os.path.basename(file_path)} - {e}")
            
            if not face_images:
                return []
            
            # 배치 분류
            predictions = self.classifier.predict_batch(face_images)
            
            # 썸네일 후보 추출
            thumbnail_candidates = []
            confidence_threshold = self.thumbnail_config['confidence_threshold']
            
            for file_path, prediction in zip(valid_files, predictions):
                if prediction['is_angry'] and prediction['confidence'] >= confidence_threshold:
                    # 파일명에서 타임스탬프 추출
                    timestamp = self._extract_timestamp_from_filename(file_path)
                    
                    thumbnail_info = {
                        'file_path': file_path,
                        'confidence': prediction['confidence'],
                        'timestamp': timestamp,
                        'filename': os.path.basename(file_path)
                    }
                    
                    thumbnail_candidates.append(thumbnail_info)
            
            return thumbnail_candidates
            
        except Exception as e:
            print(f"   ❌ 배치 분류 실패: {e}")
            return []
    
    def _extract_timestamp_from_filename(self, file_path: str) -> float:
        """
        파일명에서 타임스탬프 추출
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            float: 타임스탬프 (초 단위)
        """
        try:
            filename = os.path.basename(file_path)
            # timestamp_015750_face0_chimchakman_sim0.856.jpg 형태에서 추출
            parts = filename.split('_')
            
            if len(parts) >= 2 and parts[0] == 'timestamp':
                timestamp_ms = int(parts[1])
                return timestamp_ms / 1000.0  # 밀리초 → 초
            
        except (ValueError, IndexError):
            pass
        
        # 추출 실패 시 0 반환
        return 0.0
    
    def _save_thumbnail_image(self, thumbnail_info: Dict, rank: int, output_dir: str) -> Optional[Dict]:
        """
        썸네일 이미지 저장
        
        Args:
            thumbnail_info (Dict): 썸네일 정보
            rank (int): 순위
            output_dir (str): 출력 디렉토리
            
        Returns:
            Dict: 저장된 썸네일 정보
        """
        try:
            # 파일명 생성
            filename_format = self.thumbnail_config['filename_format']
            filename = filename_format.format(
                rank=rank,
                timestamp=thumbnail_info['timestamp'],
                confidence=thumbnail_info['confidence']
            )
            
            output_path = os.path.join(output_dir, filename)
            
            # 이미지 복사 및 저장
            with Image.open(thumbnail_info['file_path']) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 저장
                img.save(output_path, 'JPEG', quality=95)
            
            return {
                'rank': rank,
                'timestamp': thumbnail_info['timestamp'],
                'confidence': thumbnail_info['confidence'],
                'filename': filename,
                'output_path': output_path,
                'source_path': thumbnail_info['file_path']
            }
            
        except Exception as e:
            print(f"   ⚠️ 썸네일 이미지 저장 실패: {e}")
            return None
    
    def print_classification_summary(self, results: Dict):
        """
        분류 결과 요약 출력
        
        Args:
            results (Dict): 분류 결과 정보
        """
        print(f"\n📊 썸네일 분류 요약")
        print(f"{'='*50}")
        print(f"처리된 얼굴: {results['classified_count']}개")
        print(f"썸네일 후보: {results['candidate_count']}개")
        print(f"저장된 썸네일: {results['saved_count']}개")
        print(f"저장 위치: {results['output_dir']}")
        print(f"{'='*50}")
        
        if results['thumbnails']:
            print(f"상위 썸네일:")
            for thumbnail in results['thumbnails'][:5]:  # 상위 5개만 표시
                print(f"  #{thumbnail['rank']} {thumbnail['timestamp']:.1f}s "
                     f"(신뢰도: {thumbnail['confidence']:.3f}) - {thumbnail['filename']}")


def main():
    """독립 실행용 메인 함수"""
    parser = argparse.ArgumentParser(description='썸네일용 얼굴 분류기')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--faces-dir', help='침착맨 얼굴 이미지 디렉토리 경로')
    group.add_argument('--video-hdf5', help='비디오 HDF5 파일 경로')
    
    parser.add_argument('--output', '-o', default='outputs/classification', 
                       help='출력 디렉토리 (기본값: outputs/classification)')
    parser.add_argument('--config', '-c', default='pipeline/configs/integrated_config.yaml',
                       help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        # 설정 로드
        print("📋 설정 로드 중...")
        config = PipelineUtils.load_config(args.config)
        
        # 분류기 초기화
        classifier = ThumbnailClassifier(config)
        
        # 분류 수행
        if args.faces_dir:
            results = classifier.classify_faces_from_directory(args.faces_dir, args.output)
        else:  # args.video_hdf5
            results = classifier.classify_faces_from_hdf5(args.video_hdf5, args.output)
        
        # 결과 출력
        classifier.print_classification_summary(results)
        
        print(f"\n✅ 썸네일 분류 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()