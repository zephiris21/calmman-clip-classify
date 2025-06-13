#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/modules/clip_selector.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class ClipSelector:
    """
    점수 기반 클립 자동 선별기
    - scored_windows.json에서 윈도우 불러오기
    - config 기준 필터링 및 NMS 적용
    - 최종 클립 selected_clips.json 저장
    """
    def __init__(self, config_path: str = None):
        """
        초기화
        
        Args:
            config_path (str): config 파일 경로
        """
        self.logger = logging.getLogger(__name__)
        
        # Config 로드
        if config_path is None:
            config_path = "pipeline/configs/funclip_extraction_config.yaml"
        
        self.config = PipelineUtils.load_config(config_path)
        
        # selection 섹션에서 파라미터 추출
        selection_config = self.config.get('selection', {})
        self.iou_threshold = selection_config.get('iou_threshold', 0.5)
        self.target_clips = selection_config.get('target_clips', 5)
        self.min_score = selection_config.get('min_score', 0.0)
        self.min_duration = selection_config.get('min_duration', 15)
        self.max_duration = selection_config.get('max_duration', 40)

    def load_scored_windows(self, scored_windows_path: str) -> Dict:
        """
        scored_windows.json에서 윈도우 및 메타데이터 로드
        
        Args:
            scored_windows_path (str): 점수 윈도우 JSON 파일 경로
            
        Returns:
            Dict: 로드된 윈도우 데이터
        """
        try:
            with open(scored_windows_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 전체 데이터(메타데이터 포함) 반환
            self.logger.info(f"📊 윈도우 로드 완료: {len(data['windows'])}개")
            if 'metadata' in data and 'video_name' in data['metadata']:
                self.logger.info(f"   비디오: {data['metadata']['video_name']}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 윈도우 로드 실패: {e}")
            raise

    def filter_windows(self, data: Dict) -> Tuple[Dict, List[Dict]]:
        """
        윈도우 필터링 (메타데이터 유지)
        
        Args:
            data (Dict): 원본 윈도우 데이터
            
        Returns:
            Tuple[Dict, List[Dict]]: (원본 데이터, 필터링된 윈도우 리스트)
        """
        windows = data['windows']
        filtered = [
            w for w in windows
            if w['fun_score'] >= self.min_score and self.min_duration <= w['duration'] <= self.max_duration
        ]
        self.logger.info(f"1차 필터링: {len(filtered)}개 (점수≥{self.min_score}, 길이 {self.min_duration}~{self.max_duration})")
        return data, filtered

    def calculate_iou(self, win1: Dict, win2: Dict) -> float:
        """
        두 윈도우 간 IoU(Intersection over Union) 계산
        
        Args:
            win1 (Dict): 첫 번째 윈도우
            win2 (Dict): 두 번째 윈도우
            
        Returns:
            float: IoU 값 (0~1)
        """
        # 시간 구간 IoU 계산
        s1, e1 = win1['start_time'], win1['end_time']
        s2, e2 = win2['start_time'], win2['end_time']
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    def non_max_suppression(self, windows: List[Dict]) -> List[Dict]:
        """
        Non-Maximum Suppression 적용 (중복 윈도우 제거)
        
        Args:
            windows (List[Dict]): 필터링된 윈도우 리스트
            
        Returns:
            List[Dict]: NMS 적용 후 선택된 윈도우 리스트
        """
        # 점수 내림차순 정렬
        windows = sorted(windows, key=lambda w: w['fun_score'], reverse=True)
        selected = []
        for win in windows:
            if all(self.calculate_iou(win, sel) < self.iou_threshold for sel in selected):
                selected.append(win)
            if len(selected) >= self.target_clips:
                break
        self.logger.info(f"NMS 적용 후: {len(selected)}개 (IoU<{self.iou_threshold})")
        return selected

    def select_clips(self, scored_windows_path: str) -> Tuple[Dict, List[Dict]]:
        """
        점수 기반 클립 선별 (메타데이터 유지)
        
        Args:
            scored_windows_path (str): 점수 윈도우 JSON 파일 경로
            
        Returns:
            Tuple[Dict, List[Dict]]: (원본 데이터, 선택된 클립 리스트)
        """
        data = self.load_scored_windows(scored_windows_path)
        data, filtered = self.filter_windows(data)
        selected = self.non_max_suppression(filtered)
        return data, selected

    def save_selected_clips(self, original_data: Dict, clips: List[Dict], output_path: Optional[str] = None, refine_boundaries: bool = False) -> str:
        """
        선별된 클립 저장 (메타데이터 포함)
        
        Args:
            original_data (Dict): 원본 윈도우 데이터
            clips (List[Dict]): 선택된 클립 리스트
            output_path (str, optional): 출력 파일 경로 (None이면 자동 생성)
            refine_boundaries (bool): 경계 조정 여부 (다음 단계로 연결)
            
        Returns:
            str: 저장된 파일 경로
        """
        # 원본 메타데이터 가져오기
        metadata = original_data.get('metadata', {})
        video_name = metadata.get('video_name', 'unknown')
        
        # 출력 경로 자동 생성 (지정되지 않은 경우)
        if output_path is None:
            # 출력 디렉토리 확인
            safe_video_name = PipelineUtils.safe_filename(video_name)
            output_dir = os.path.join(
                self.config['output']['base_dir'],
                safe_video_name
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # 타임스탬프 생성
            timestamp = PipelineUtils.get_timestamp()
            
            # 파일명 생성
            output_path = os.path.join(
                output_dir,
                f"selected_clips_{safe_video_name}_{timestamp}.json"
            )
        else:
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # config 정보 수집
        config_used = {
            'iou_threshold': self.iou_threshold,
            'target_clips': self.target_clips,
            'min_score': self.min_score,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration,
            'refine_boundaries': refine_boundaries
        }
        
        # 결과 데이터 구성
        result = {
            'metadata': {
                'video_name': video_name,
                'source_scored_windows': os.path.basename(original_data.get('source_path', 'unknown')),
                'total_clips': len(clips),
                'total_original_windows': len(original_data.get('windows', [])),
                'original_score_statistics': metadata.get('score_statistics', {}),
                'timestamp': datetime.now().isoformat()
            },
            'selection_info': {
                'selected_at': datetime.now().isoformat(),
                'config_used': config_used
            },
            'clips': [
                {
                    'rank': i+1,
                    'start_time': c['start_time'],
                    'end_time': c['end_time'],
                    'duration': c['duration'],
                    'fun_score': c['fun_score'],
                    'cluster_id': c.get('cluster_id', -1)
                } for i, c in enumerate(clips)
            ]
        }
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 클립 선별 결과 저장 완료: {output_path}")
        self.logger.info(f"   비디오: {video_name}")
        self.logger.info(f"   최종 클립: {len(clips)}개")
        
        return output_path


def main():
    """테스트 실행"""
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='점수 기반 클립 자동 선별')
    parser.add_argument('scored_windows', help='scored_windows.json 경로')
    parser.add_argument('--config', help='Config 파일 경로')
    parser.add_argument('--output', help='출력 JSON 경로')
    parser.add_argument('--refine', action='store_true', help='클립 경계 조정 여부 (기본: False)')
    parser.add_argument('--audio', help='오디오 HDF5 파일 (경계 조정 시 필요)')
    
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # 설정 로드 및 클립 선별
        selector = ClipSelector(config_path=args.config)
        
        # 원본 파일 경로 저장
        original_data, selected = selector.select_clips(args.scored_windows)
        original_data['source_path'] = args.scored_windows
        
        # 클립 저장
        clips_path = selector.save_selected_clips(
            original_data, 
            selected, 
            args.output,
            refine_boundaries=args.refine
        )
        
        print(f"\n✅ 클립 선별 완료!")
        print(f"📊 {len(selected)}개 클립 저장: {os.path.basename(clips_path)}")
        
        # 경계 조정 여부 확인
        if args.refine and args.audio:
            print(f"\n🔍 클립 경계 조정 시작...")
            
            # 설정 로드
            config = PipelineUtils.load_config(args.config if args.config else "pipeline/configs/funclip_extraction_config.yaml")
            
            # 출력 디렉토리 설정
            output_dirs = PipelineUtils.setup_output_directories(config)
            
            # 로깅 설정
            logger = PipelineUtils.setup_logging(config, output_dirs)
            
            # 오디오 데이터 로드
            audio_data = PipelineUtils.load_audio_hdf5(args.audio)
            if audio_data:
                # clip_refiner 모듈 동적 import
                from pipeline.modules.clip_refiner import ClipRefiner
                
                # 메타데이터에서 비디오 이름 추출
                video_name = original_data.get('metadata', {}).get('video_name', 'unknown')
                
                # 클립 데이터 준비
                with open(clips_path, 'r', encoding='utf-8') as f:
                    clips_data = json.load(f)
                
                # 클립 경계 조정
                refiner = ClipRefiner(config, logger)
                refined_clips = refiner.refine_clips(clips_data['clips'], audio_data, video_name)
                
                print(f"✅ 클립 경계 조정 완료: {len(refined_clips)}개")
            else:
                print(f"❌ 오디오 데이터 로드 실패: {args.audio}")
                print(f"   경계 조정을 건너뜁니다.")
        
    except Exception as e:
        print(f"❌ 클립 선별 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 