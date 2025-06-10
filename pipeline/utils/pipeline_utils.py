#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import h5py
import logging
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple


# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# 기존 모듈들 import
from video_analyzer.inference_prep.video_preprocessor import LongVideoProcessor
from video_analyzer.inference_prep.audio_preprocessor import LongVideoAudioPreprocessor
from src.mtcnn_wrapper import FaceDetector
from src.va_emotion_core import VAEmotionCore
from tension_analyzer.tension_calculator import MultiEmotionTensionCalculator
from tension_analyzer.tension_visualizer import TensionVisualizer
from src.pytorch_classifier import TorchFacialClassifier


class PipelineUtils:
    """파이프라인 공통 유틸리티 클래스"""
    
    @staticmethod
    def load_config(config_path: str = "pipeline/configs/integrated_config.yaml") -> Dict:
        """
        통합 설정 파일 로드
        
        Args:
            config_path (str): 설정 파일 경로
            
        Returns:
            Dict: 로드된 설정
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    @staticmethod
    def setup_output_directories(config: Dict) -> Dict:
        """
        출력 디렉토리 생성
        
        Args:
            config (Dict): 설정 정보
            
        Returns:
            Dict: 생성된 디렉토리 경로들
        """
        base_dir = config['output']['base_dir']
        
        dirs = {
            'base': base_dir,
            'preprocessed': os.path.join(base_dir, config['output']['preprocessed']),
            'debug_faces': os.path.join(base_dir, config['output']['debug_faces']),
            'tension': os.path.join(base_dir, config['output']['tension']),
            'classification': os.path.join(base_dir, config['output']['classification']),
            'highlights': os.path.join(base_dir, config['output']['highlights']),
            'visualization': os.path.join(base_dir, config['output']['visualization']),
            'logs': os.path.join(base_dir, config['output']['logs'])
        }
        
        # 모든 디렉토리 생성
        for dir_name, dir_path in dirs.items():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 출력 디렉토리 생성 완료: {base_dir}")
        return dirs
    
    @staticmethod
    def setup_logging(config: Dict, output_dirs: Dict) -> logging.Logger:
        """
        로깅 시스템 설정
        
        Args:
            config (Dict): 설정 정보
            output_dirs (Dict): 출력 디렉토리 정보
            
        Returns:
            logging.Logger: 설정된 로거
        """
        logger = logging.getLogger('IntegratedPipeline')
        logger.setLevel(getattr(logging, config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        if config['logging']['console_output']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 파일 핸들러
        if config['logging']['file_output']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = config['logging']['log_filename'].format(timestamp=timestamp)
            log_path = os.path.join(output_dirs['logs'], log_filename)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"📄 로그 파일: {log_path}")
        
        return logger
    
    @staticmethod
    def load_video_hdf5(hdf5_path: str) -> Optional[Dict]:
        """
        비디오 전처리 HDF5 파일 로드
        
        Args:
            hdf5_path (str): HDF5 파일 경로
            
        Returns:
            Dict: 로드된 비디오 데이터
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                data = {
                    'metadata': {
                        'video_name': f.attrs.get('video_name', ''),
                        'video_path': f.attrs.get('video_path', ''),
                        'duration': f.attrs.get('duration', 0),
                        'fps': f.attrs.get('fps', 0),
                        'total_frames': f.attrs.get('total_frames', 0),
                        'face_detection_ratio': f.attrs.get('face_detection_ratio', 0),
                        'face_images_dir': f.attrs.get('face_images_dir', ''),
                        'chimchakman_faces_dir': f.attrs.get('chimchakman_faces_dir', '')
                    },
                    'sequences': {
                        'emotions': f['sequences/emotions'][:],
                        'face_detected': f['sequences/face_detected'][:],
                        'timestamps': f['sequences/timestamps'][:],
                        'frame_indices': f['sequences/frame_indices'][:]
                    }
                }
            
            return data
            
        except Exception as e:
            print(f"❌ 비디오 HDF5 로드 실패: {hdf5_path} - {e}")
            return None
    
    @staticmethod
    def load_audio_hdf5(hdf5_path: str) -> Optional[Dict]:
        """
        오디오 전처리 HDF5 파일 로드
        
        Args:
            hdf5_path (str): HDF5 파일 경로
            
        Returns:
            Dict: 로드된 오디오 데이터
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                data = {
                    'metadata': {
                        'video_name': f.attrs.get('video_name', ''),
                        'video_path': f.attrs.get('video_path', ''),
                        'duration': f.attrs.get('duration', 0),
                        'sample_rate': f.attrs.get('sample_rate', 16000),
                        'analysis_interval': f.attrs.get('analysis_interval', 0.05),
                        'total_frames': f.attrs.get('total_frames', 0)
                    },
                    'sequences': {
                        'rms_values': f['sequences/rms_values'][:],
                        'vad_labels': f['sequences/vad_labels'][:],
                        'timestamps': f['sequences/timestamps'][:]
                    }
                }
            
            return data
            
        except Exception as e:
            print(f"❌ 오디오 HDF5 로드 실패: {hdf5_path} - {e}")
            return None
    
    @staticmethod
    def find_hdf5_files(output_dirs: Dict, video_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        비디오 이름으로 해당하는 HDF5 파일들 찾기
        
        Args:
            output_dirs (Dict): 출력 디렉토리 정보
            video_name (str): 비디오 파일명 (확장자 제외)
            
        Returns:
            Tuple[str, str]: (video_hdf5_path, audio_hdf5_path)
        """
        video_hdf5_path = None
        audio_hdf5_path = None
        
        # 비디오 HDF5 찾기
        video_seq_dir = os.path.join(output_dirs['preprocessed'], 'video_sequences')
        if os.path.exists(video_seq_dir):
            for file in os.listdir(video_seq_dir):
                if file.startswith(f'video_seq_{video_name}') and file.endswith('.h5'):
                    video_hdf5_path = os.path.join(video_seq_dir, file)
                    break
        
        # 오디오 HDF5 찾기
        audio_seq_dir = os.path.join(output_dirs['preprocessed'], 'audio_sequences')
        if os.path.exists(audio_seq_dir):
            for file in os.listdir(audio_seq_dir):
                if file.startswith(f'audio_seq_{video_name}') and file.endswith('.h5'):
                    audio_hdf5_path = os.path.join(audio_seq_dir, file)
                    break
        
        return video_hdf5_path, audio_hdf5_path
    
    @staticmethod
    def get_face_images_directory(video_data: Dict) -> Optional[str]:
        """
        비디오 데이터에서 얼굴 이미지 디렉토리 경로 추출
        
        Args:
            video_data (Dict): 로드된 비디오 데이터
            
        Returns:
            str: 얼굴 이미지 디렉토리 경로
        """
        face_images_dir = video_data['metadata'].get('face_images_dir', '')
        if face_images_dir and os.path.exists(face_images_dir):
            return face_images_dir
        
        # 백업: chimchakman_faces_dir에서 상위 디렉토리 추출
        chimchakman_dir = video_data['metadata'].get('chimchakman_faces_dir', '')
        if chimchakman_dir and os.path.exists(chimchakman_dir):
            return os.path.dirname(chimchakman_dir)
        
        return None
    
    @staticmethod
    def get_chimchakman_faces_directory(video_data: Dict) -> Optional[str]:
        """
        침착맨 얼굴 이미지 디렉토리 경로 추출
        
        Args:
            video_data (Dict): 로드된 비디오 데이터
            
        Returns:
            str: 침착맨 얼굴 이미지 디렉토리 경로
        """
        chimchakman_dir = video_data['metadata'].get('chimchakman_faces_dir', '')
        if chimchakman_dir and os.path.exists(chimchakman_dir):
            return chimchakman_dir
        
        # 백업: face_images_dir + chimchakman
        face_images_dir = PipelineUtils.get_face_images_directory(video_data)
        if face_images_dir:
            chimchakman_path = os.path.join(face_images_dir, 'chimchakman')
            if os.path.exists(chimchakman_path):
                return chimchakman_path
        
        return None
    
    @staticmethod
    def print_step_banner(step_num: int, step_name: str, description: str):
        """
        단계별 배너 출력
        
        Args:
            step_num (int): 단계 번호
            step_name (str): 단계 이름
            description (str): 단계 설명
        """
        print(f"\n{'='*60}")
        print(f"🚀 {step_num}단계: {step_name}")
        print(f"📋 {description}")
        print(f"{'='*60}")
    
    @staticmethod
    def print_completion_banner(step_num: int, step_name: str, result_info: str = ""):
        """
        단계 완료 배너 출력
        
        Args:
            step_num (int): 단계 번호
            step_name (str): 단계 이름
            result_info (str): 결과 정보
        """
        print(f"\n✅ {step_num}단계: {step_name} 완료!")
        if result_info:
            print(f"📊 {result_info}")
    
    @staticmethod
    def wait_for_user_input(auto_mode: bool, step_name: str) -> bool:
        """
        사용자 입력 대기 (단계별 모드일 때)
        
        Args:
            auto_mode (bool): 자동 모드 여부 (False일 때만 대기)
            step_name (str): 단계 이름
            
        Returns:
            bool: 계속 진행 여부 (True: 계속, False: 종료)
        """
        if auto_mode:
            return True
        
        print(f"\n⏸️  {step_name} 단계가 완료되었습니다.")
        user_input = input("Enter를 눌러 다음 단계로 진행하거나 'q'를 입력하여 종료하세요: ").strip().lower()
        
        if user_input == 'q':
            print("🛑 사용자 요청으로 파이프라인을 종료합니다.")
            return False
        
        return True
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """
        안전한 파일명 생성 (특수문자 제거)
        
        Args:
            filename (str): 원본 파일명
            
        Returns:
            str: 안전한 파일명
        """
        import re
        # 특수문자를 언더스코어로 대체
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        return safe_name


def main():
    """테스트 실행"""
    # 설정 로드 테스트
    try:
        config = PipelineUtils.load_config()
        print("✅ 설정 로드 테스트 성공")
        
        # 출력 디렉토리 생성 테스트
        output_dirs = PipelineUtils.setup_output_directories(config)
        print("✅ 출력 디렉토리 생성 테스트 성공")
        
        # 로깅 설정 테스트
        logger = PipelineUtils.setup_logging(config, output_dirs)
        logger.info("✅ 로깅 시스템 테스트 성공")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    main()