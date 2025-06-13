#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/modules/clip_refiner.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class ClipRefiner:
    """
    VAD(Voice Activity Detection) 기반으로 클립 경계를 자연스럽게 조정하는 모듈
    
    선택된 클립의 시작/끝 부분을 VAD 정보를 활용하여 더 자연스러운 지점으로 조정합니다.
    - 발화가 중간에 끊기지 않도록 조정
    - 최대 3초까지 클립 경계 확장 가능
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        초기화
        
        Args:
            config (Dict): 설정 정보
            logger (logging.Logger, optional): 로거 객체
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # VAD 관련 설정 (구조 통일)
        refiner_config = config.get('clip_refiner', {})
        
        # 키 이름 통일 (이전 코드 호환성 유지)
        self.max_extend_seconds = refiner_config.get('max_extend_seconds', 3.0)
        self.silence_padding = refiner_config.get('silence_padding', 0.05)
        self.extend_on_speech_boundary = refiner_config.get('extend_on_speech_boundary', True)
        
        self.logger.info(f"클립 경계 조정 모듈 초기화: 최대 확장 {self.max_extend_seconds}초")
    
    def refine_clips(self, selected_clips: List[Dict], audio_data: Dict, video_name: str) -> List[Dict]:
        """
        선택된 클립들의 경계 조정
        
        Args:
            selected_clips (List[Dict]): 선택된 클립 목록
            audio_data (Dict): 오디오 전처리 데이터 (VAD 포함)
            video_name (str): 비디오 이름
            
        Returns:
            List[Dict]: 경계가 조정된 클립 목록
        """
        self.logger.info(f"클립 경계 조정 시작: {len(selected_clips)}개 클립")
        
        # VAD 데이터 준비
        vad_labels = audio_data['sequences']['vad_labels']
        timestamps = audio_data['sequences']['timestamps']
        
        refined_clips = []
        
        for clip in selected_clips:
            # NumPy 타입을 Python 기본 타입으로 변환
            clip_copy = self._convert_numpy_types(clip)
            
            start_time = clip_copy['start_time']
            end_time = clip_copy['end_time']
            
            # 원본 경계 저장
            clip_copy['original_start_time'] = start_time
            clip_copy['original_end_time'] = end_time
            
            # 경계 조정 (단순화된 로직)
            new_start, new_end = self._adjust_boundaries_simple(
                start_time, end_time, vad_labels, timestamps
            )
            
            # 조정된 경계 적용
            clip_copy['start_time'] = float(new_start)
            clip_copy['end_time'] = float(new_end)
            clip_copy['duration'] = float(new_end - new_start)
            
            # 경계 조정 정보 추가
            clip_copy['boundary_refined'] = True
            clip_copy['start_extended'] = bool(new_start < start_time)
            clip_copy['end_extended'] = bool(new_end > end_time)
            
            refined_clips.append(clip_copy)
            
            self.logger.debug(
                f"클립 경계 조정: {start_time:.2f}-{end_time:.2f} → "
                f"{new_start:.2f}-{new_end:.2f} (확장: {new_end-new_start-end_time+start_time:.2f}초)"
            )
        
        # 결과 저장
        # 출력 디렉토리 경로
        safe_video_name = PipelineUtils.safe_filename(video_name)
        
        # 기본 출력 디렉토리 확인 (config에 없는 경우 기본값 사용)
        base_dir = self.config.get('output', {}).get('base_dir', 'outputs/clip_analysis')
        
        output_dir = os.path.join(
            base_dir,
            safe_video_name
        )
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = PipelineUtils.get_timestamp()
        output_path = os.path.join(
            output_dir, 
            f"refined_clips_{safe_video_name}_{timestamp}.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'video_name': video_name,
                'timestamp': timestamp,
                'clips': refined_clips,
                'refiner_config': {
                    'max_extend_seconds': self.max_extend_seconds,
                    'silence_padding': self.silence_padding,
                    'extend_on_speech_boundary': self.extend_on_speech_boundary
                }
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"클립 경계 조정 완료: {len(refined_clips)}개 클립, 저장 경로: {output_path}")
        
        return refined_clips
    
    def _adjust_boundaries_simple(
        self, 
        start_time: float, 
        end_time: float, 
        vad_labels: np.ndarray, 
        timestamps: np.ndarray
    ) -> Tuple[float, float]:
        """
        단순화된 VAD 기반 클립 경계 조정
        
        Args:
            start_time (float): 원본 시작 시간 (초)
            end_time (float): 원본 종료 시간 (초)
            vad_labels (np.ndarray): VAD 레이블 배열 (1: 음성, 0: 비음성)
            timestamps (np.ndarray): 타임스탬프 배열
            
        Returns:
            Tuple[float, float]: 조정된 (시작 시간, 종료 시간)
        """
        if not self.extend_on_speech_boundary:
            return start_time, end_time
        
        # 시작 시간 기준 인덱스 찾기
        start_idx = np.searchsorted(timestamps, start_time, side='right') - 1
        start_idx = max(0, start_idx)
        
        # 종료 시간 기준 인덱스 찾기
        end_idx = np.searchsorted(timestamps, end_time, side='left')
        end_idx = min(end_idx, len(timestamps) - 1)
        
        # 시작 경계 조정 (단순화된 로직)
        new_start = start_time
        if start_idx > 0:
            # 현재 지점이 음성이면, 음성 시작 부분을 찾아 확장
            if vad_labels[start_idx] == 1:
                # 최대 3초 이내의 앞쪽 무음 지점 찾기
                for i in range(start_idx, -1, -1):
                    if vad_labels[i] == 0:  # 무음 지점 발견
                        # 무음에서 silence_padding 초 앞으로 이동
                        new_start = max(timestamps[i] - self.silence_padding, 0)
                        break
                    # 최대 확장 범위 도달
                    if start_time - timestamps[i] > self.max_extend_seconds:
                        new_start = start_time - self.max_extend_seconds
                        break
        
        # 종료 경계 조정 (단순화된 로직)
        new_end = end_time
        if end_idx < len(vad_labels) - 1:
            # 현재 지점이 음성이면, 음성 종료 부분을 찾아 확장
            if vad_labels[end_idx] == 1:
                # 최대 3초 이내의 뒤쪽 무음 지점 찾기
                for i in range(end_idx, len(vad_labels)):
                    if vad_labels[i] == 0:  # 무음 지점 발견
                        # 무음에서 silence_padding 초 뒤로 이동
                        new_end = min(timestamps[i] + self.silence_padding, timestamps[-1])
                        break
                    # 최대 확장 범위 도달
                    if timestamps[i] - end_time > self.max_extend_seconds:
                        new_end = end_time + self.max_extend_seconds
                        break
        
        # 최대 확장 범위 제한
        new_start = max(new_start, start_time - self.max_extend_seconds)
        new_end = min(new_end, end_time + self.max_extend_seconds)
        
        return new_start, new_end

    def _convert_numpy_types(self, obj):
        """NumPy 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        else:
            return obj


def main():
    """
    직접 실행 시 테스트 함수
    """
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description="VAD 기반 클립 경계 조정 모듈")
    parser.add_argument("clips_json", nargs='?', help="선택된 클립 JSON 파일 경로")
    parser.add_argument("audio_h5", nargs='?', help="오디오 전처리 HDF5 파일 경로")
    parser.add_argument("--clips", help="선택된 클립 JSON 파일 경로 (named 인자)")
    parser.add_argument("--audio", help="오디오 전처리 HDF5 파일 경로 (named 인자)")
    parser.add_argument("--config", default="pipeline/configs/funclip_extraction_config.yaml", 
                        help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # 클립 경로 결정 (positional 또는 named 인자)
    clips_path = args.clips_json if args.clips_json else args.clips
    if not clips_path:
        print("❌ 오류: 클립 JSON 파일 경로가 필요합니다. positional 인자 또는 --clips 옵션을 사용하세요.")
        parser.print_help()
        return
    
    # 오디오 경로 결정 (positional 또는 named 인자)
    audio_path = args.audio_h5 if args.audio_h5 else args.audio
    if not audio_path:
        print("❌ 오류: 오디오 HDF5 파일 경로가 필요합니다. positional 인자 또는 --audio 옵션을 사용하세요.")
        parser.print_help()
        return
    
    # 설정 로드
    config = PipelineUtils.load_config(args.config)
    
    # 출력 디렉토리 설정 제거 (오류 발생)
    # 대신 출력 디렉토리 설정이 필요한 경우 직접 처리
    if 'output' not in config:
        config['output'] = {'base_dir': 'outputs/clip_analysis'}
    elif 'base_dir' not in config['output']:
        config['output']['base_dir'] = 'outputs/clip_analysis'
    
    # 로깅 설정 (logging 키가 없을 경우 기본 로깅 설정 사용)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 콘솔 핸들러 추가
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
    
    logger.info("🔍 클립 경계 조정 프로세스 시작")
    
    # 클립 데이터 로드
    try:
        with open(clips_path, 'r', encoding='utf-8') as f:
            clips_data = json.load(f)
        
        # 비디오 이름 추출
        video_name = clips_data.get('metadata', {}).get('video_name', None)
        if video_name is None:
            video_name = os.path.splitext(os.path.basename(clips_path))[0]
            if video_name.startswith('selected_clips_'):
                video_name = video_name[14:]  # 'selected_clips_' 제거
        
        selected_clips = clips_data.get('clips', [])
        logger.info(f"📊 클립 데이터 로드 완료: {len(selected_clips)}개 클립")
        logger.info(f"   비디오: {video_name}")
        
        # 오디오 데이터 로드
        audio_data = PipelineUtils.load_audio_hdf5(audio_path)
        if not audio_data:
            logger.error(f"❌ 오디오 데이터 로드 실패: {audio_path}")
            return
        
        # 클립 경계 조정
        refiner = ClipRefiner(config, logger)
        refined_clips = refiner.refine_clips(selected_clips, audio_data, video_name)
        
        print(f"\n✅ 클립 경계 조정 완료!")
        print(f"📊 {len(refined_clips)}개 클립 경계 조정됨")
        
    except Exception as e:
        logger.error(f"❌ 클립 경계 조정 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 