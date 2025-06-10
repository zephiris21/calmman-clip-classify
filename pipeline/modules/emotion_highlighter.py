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

# 프로젝트 루트로 이동
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_root = os.path.dirname(current_dir)  # pipeline/
project_root = os.path.dirname(pipeline_root)  # project_root/
os.chdir(project_root)

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class EmotionHighlighter:
    """
    감정 하이라이트 추출기
    HDF5 비디오 데이터에서 각 감정별 상위 N개 프레임의 얼굴 이미지를 추출
    """
    
    def __init__(self, config: Dict):
        """
        감정 하이라이트 추출기 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
        self.highlight_config = config['emotion_highlights']
        
        # 감정 레이블 (VA 모델 순서)
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
        
        print(f"✅ 감정 하이라이트 추출기 초기화 완료")
        print(f"   추출할 감정: {self.highlight_config['include_emotions']}")
        print(f"   각 감정별 상위: {self.highlight_config['top_n_per_emotion']}개")
    
    def extract_highlights(self, video_hdf5_path: str, output_dir: str) -> Dict:
        """
        비디오 HDF5에서 감정 하이라이트 추출
        
        Args:
            video_hdf5_path (str): 비디오 HDF5 파일 경로
            output_dir (str): 하이라이트 이미지 저장 디렉토리
            
        Returns:
            Dict: 추출 결과 정보
        """
        print(f"\n🎭 감정 하이라이트 추출 시작")
        print(f"   입력: {video_hdf5_path}")
        print(f"   출력: {output_dir}")
        
        # 비디오 데이터 로드
        video_data = PipelineUtils.load_video_hdf5(video_hdf5_path)
        if video_data is None:
            raise ValueError(f"비디오 HDF5 로드 실패: {video_hdf5_path}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 얼굴 이미지 디렉토리 찾기
        chimchakman_faces_dir = PipelineUtils.get_chimchakman_faces_directory(video_data)
        if not chimchakman_faces_dir:
            raise ValueError("침착맨 얼굴 이미지 디렉토리를 찾을 수 없습니다")
        
        print(f"   얼굴 이미지 디렉토리: {chimchakman_faces_dir}")
        
        # 감정 데이터 추출
        emotions = video_data['sequences']['emotions']  # [N, 10]
        timestamps = video_data['sequences']['timestamps']
        face_detected = video_data['sequences']['face_detected']
        
        print(f"   총 프레임: {len(emotions)}개")
        print(f"   얼굴 탐지된 프레임: {np.sum(face_detected)}개")
        
        # 얼굴이 탐지된 프레임만 필터링
        valid_indices = np.where(face_detected)[0]
        if len(valid_indices) == 0:
            print("⚠️ 얼굴이 탐지된 프레임이 없습니다")
            return {'extracted_count': 0, 'highlights': {}}
        
        valid_emotions = emotions[valid_indices]
        valid_timestamps = timestamps[valid_indices]
        
        print(f"   유효한 프레임: {len(valid_indices)}개")
        
        # 각 감정별 하이라이트 추출
        extraction_results = {}
        total_extracted = 0
        
        for emotion in self.highlight_config['include_emotions']:
            try:
                highlights = self._extract_emotion_highlights(
                    emotion, valid_emotions, valid_timestamps, valid_indices,
                    chimchakman_faces_dir, output_dir
                )
                extraction_results[emotion] = highlights
                total_extracted += len(highlights)
                
                print(f"   {emotion}: {len(highlights)}개 추출")
                
            except Exception as e:
                print(f"   ⚠️ {emotion} 추출 실패: {e}")
                extraction_results[emotion] = []
        
        print(f"\n✅ 감정 하이라이트 추출 완료: 총 {total_extracted}개")
        
        return {
            'extracted_count': total_extracted,
            'highlights': extraction_results,
            'video_name': video_data['metadata']['video_name'],
            'output_dir': output_dir
        }
    
    def _extract_emotion_highlights(self, emotion: str, emotions: np.ndarray, 
                                  timestamps: np.ndarray, frame_indices: np.ndarray,
                                  faces_dir: str, output_dir: str) -> List[Dict]:
        """
        특정 감정의 하이라이트 추출
        
        Args:
            emotion (str): 감정 이름
            emotions (np.ndarray): 감정 데이터 [N, 10]
            timestamps (np.ndarray): 타임스탬프 배열
            frame_indices (np.ndarray): 프레임 인덱스 배열
            faces_dir (str): 얼굴 이미지 디렉토리
            output_dir (str): 출력 디렉토리
            
        Returns:
            List[Dict]: 추출된 하이라이트 정보 리스트
        """
        top_n = self.highlight_config['top_n_per_emotion']
        min_threshold = self.highlight_config['min_emotion_threshold']
        
        # 감정 값 추출
        if emotion == 'Valence':
            # Valence: 절댓값이 큰 순서 (극값)
            emotion_values = np.abs(emotions[:, 8])
        elif emotion == 'Arousal':
            # Arousal: 높은 값 순서
            emotion_values = emotions[:, 9]
        else:
            # 8개 기본 감정 중 하나
            if emotion not in self.emotion_labels:
                raise ValueError(f"알 수 없는 감정: {emotion}")
            
            emotion_idx = self.emotion_labels.index(emotion)
            emotion_values = emotions[:, emotion_idx]
        
        # 임계값 이상인 값들만 필터링
        valid_mask = emotion_values >= min_threshold
        if not np.any(valid_mask):
            return []
        
        valid_values = emotion_values[valid_mask]
        valid_timestamps = timestamps[valid_mask]
        valid_frame_indices = frame_indices[valid_mask]
        
        # 상위 N개 선택
        top_indices = np.argsort(valid_values)[-top_n:][::-1]  # 내림차순
        
        highlights = []
        
        for rank, idx in enumerate(top_indices):
            timestamp = valid_timestamps[idx]
            frame_idx = valid_frame_indices[idx]
            value = valid_values[idx]
            
            # 해당 프레임의 얼굴 이미지 찾기
            face_image_path = self._find_face_image(faces_dir, timestamp)
            
            if face_image_path and os.path.exists(face_image_path):
                # 하이라이트 이미지 저장
                highlight_info = self._save_highlight_image(
                    face_image_path, emotion, rank + 1, timestamp, value, output_dir
                )
                
                if highlight_info:
                    highlights.append(highlight_info)
        
        return highlights
    
    def _find_face_image(self, faces_dir: str, timestamp: float) -> Optional[str]:
        """
        타임스탬프에 해당하는 얼굴 이미지 찾기
        
        Args:
            faces_dir (str): 얼굴 이미지 디렉토리
            timestamp (float): 타임스탬프 (초)
            
        Returns:
            str: 찾은 얼굴 이미지 경로 (없으면 None)
        """
        # 타임스탬프를 밀리초 단위로 변환
        timestamp_ms = int(timestamp * 1000)
        
        # 패턴 매칭으로 해당 타임스탬프의 이미지 찾기
        # 파일명 형식: timestamp_{timestamp:06d}_face{face_idx}_{type}_sim{similarity:.3f}.jpg
        pattern = f"timestamp_{timestamp_ms:06d}_*.jpg"
        search_pattern = os.path.join(faces_dir, pattern)
        
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            # 여러 개 있으면 첫 번째 선택 (보통 face0)
            return matching_files[0]
        
        # 패턴이 안 맞으면 가장 가까운 타임스탬프 찾기
        all_images = glob.glob(os.path.join(faces_dir, "timestamp_*.jpg"))
        
        if not all_images:
            return None
        
        # 파일명에서 타임스탬프 추출하여 가장 가까운 것 찾기
        closest_file = None
        min_diff = float('inf')
        
        for image_path in all_images:
            filename = os.path.basename(image_path)
            try:
                # timestamp_015750_face0_chimchakman_sim0.856.jpg 형태에서 타임스탬프 추출
                parts = filename.split('_')
                if len(parts) >= 2 and parts[0] == 'timestamp':
                    file_timestamp_ms = int(parts[1])
                    diff = abs(file_timestamp_ms - timestamp_ms)
                    
                    if diff < min_diff:
                        min_diff = diff
                        closest_file = image_path
                        
                        # 100ms 이내면 정확한 매치로 간주
                        if diff <= 100:
                            break
                            
            except (ValueError, IndexError):
                continue
        
        return closest_file
    
    def _save_highlight_image(self, source_path: str, emotion: str, rank: int, 
                            timestamp: float, value: float, output_dir: str) -> Optional[Dict]:
        """
        하이라이트 이미지 저장
        
        Args:
            source_path (str): 원본 얼굴 이미지 경로
            emotion (str): 감정 이름
            rank (int): 순위
            timestamp (float): 타임스탬프
            value (float): 감정 값
            output_dir (str): 출력 디렉토리
            
        Returns:
            Dict: 저장된 하이라이트 정보
        """
        try:
            # 파일명 생성
            filename_format = self.highlight_config['filename_format']
            filename = filename_format.format(
                emotion=emotion,
                rank=rank,
                timestamp=timestamp,
                value=value
            )
            
            output_path = os.path.join(output_dir, filename)
            
            # 이미지 복사 및 저장
            with Image.open(source_path) as img:
                # RGB 변환 (필요한 경우)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 저장
                img.save(output_path, 'JPEG', quality=95)
            
            return {
                'emotion': emotion,
                'rank': rank,
                'timestamp': timestamp,
                'value': value,
                'filename': filename,
                'output_path': output_path,
                'source_path': source_path
            }
            
        except Exception as e:
            print(f"   ⚠️ 하이라이트 이미지 저장 실패: {e}")
            return None
    
    def print_extraction_summary(self, results: Dict):
        """
        추출 결과 요약 출력
        
        Args:
            results (Dict): 추출 결과 정보
        """
        print(f"\n📊 감정 하이라이트 추출 요약")
        print(f"{'='*50}")
        print(f"비디오: {results['video_name']}")
        print(f"총 추출 개수: {results['extracted_count']}개")
        print(f"저장 위치: {results['output_dir']}")
        print(f"{'='*50}")
        
        for emotion, highlights in results['highlights'].items():
            if highlights:
                print(f"{emotion:>10}: {len(highlights):2d}개")
                for highlight in highlights[:3]:  # 상위 3개만 표시
                    print(f"          └─ #{highlight['rank']} {highlight['timestamp']:.1f}s (값: {highlight['value']:.3f})")
            else:
                print(f"{emotion:>10}:  0개")


def main():
    """독립 실행용 메인 함수"""
    parser = argparse.ArgumentParser(description='감정 하이라이트 추출기')
    parser.add_argument('video_hdf5', help='비디오 HDF5 파일 경로')
    parser.add_argument('--output', '-o', default='outputs/highlights', 
                       help='출력 디렉토리 (기본값: outputs/highlights)')
    parser.add_argument('--config', '-c', default='pipeline/configs/integrated_config.yaml',
                       help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        # 설정 로드
        print("📋 설정 로드 중...")
        config = PipelineUtils.load_config(args.config)
        
        # 추출기 초기화
        highlighter = EmotionHighlighter(config)
        
        # 하이라이트 추출
        results = highlighter.extract_highlights(args.video_hdf5, args.output)
        
        # 결과 출력
        highlighter.print_extraction_summary(results)
        
        print(f"\n✅ 감정 하이라이트 추출 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()