#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
from threading import Lock

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/modules/fun_clip_extractor.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class FunClipExtractor:
    """
    재미 클립 추출기 - refined_clips.json에서 실제 MP4 클립 생성
    
    주요 기능:
    - refined_clips.json 파싱
    - FFmpeg 기반 클립 생성
    - 배치 처리 지원
    - 진행 상황 모니터링
    """
    
    def __init__(self, config_path: str = None):
        """
        재미 클립 추출기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 프로젝트 루트로 작업 디렉토리 변경
        os.chdir(project_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Config 로드
        if config_path is None:
            config_path = "pipeline/configs/clip_generation_config.yaml"
        
        self.config = PipelineUtils.load_config(config_path)
        
        # 클립 생성 설정 추출
        extraction_config = self.config['clip_extraction']
        self.video_crf = extraction_config['video_crf']
        self.preset = extraction_config['preset']
        self.filename_format = extraction_config['filename_format']
        self.output_base = extraction_config['output_base']
        self.batch_size = extraction_config['batch_size']
        self.overwrite_existing = extraction_config['overwrite_existing']
        self.create_merged_clips = extraction_config['create_merged_clips']
        self.create_separate_tracks = extraction_config['create_separate_tracks']
        self.avoid_negative_ts = extraction_config['avoid_negative_ts']
        self.accurate_seek = extraction_config['accurate_seek']
        
        # 스레드 안전성을 위한 락
        self.progress_lock = Lock()
        self.stats = {'created': 0, 'skipped': 0, 'failed': 0}
        
        self.logger.info("✅ 재미 클립 추출기 초기화 완료")
        self.logger.info(f"   출력 경로: {self.output_base}")
        self.logger.info(f"   배치 크기: {self.batch_size}")
        self.logger.info(f"   품질 설정: CRF {self.video_crf}, {self.preset}")
    
    def load_refined_clips(self, json_path: str) -> Dict:
        """
        refined_clips.json 파일 로드
        
        Args:
            json_path (str): refined_clips.json 파일 경로
            
        Returns:
            Dict: 로드된 클립 데이터
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_name = data.get('video_name', 'unknown')
            clips = data.get('clips', [])
            
            self.logger.info(f"📊 클립 데이터 로드 완료: {len(clips)}개")
            self.logger.info(f"   비디오: {video_name}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 클립 데이터 로드 실패: {e}")
            raise
    
    def create_clip_queue(self, clips_data: Dict, video_path: str) -> List[Dict]:
        """
        클립 생성 작업 큐 생성
        
        Args:
            clips_data (Dict): 로드된 클립 데이터
            video_path (str): 원본 비디오 파일 경로
            
        Returns:
            List[Dict]: 클립 생성 작업 큐
        """
        video_name = clips_data.get('video_name', 'unknown')
        clips = clips_data.get('clips', [])
        
        # 출력 디렉토리 생성
        safe_video_name = PipelineUtils.safe_filename(video_name)
        output_dir = os.path.join(self.output_base, safe_video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        clip_queue = []
        
        for clip in clips:
            # 파일명 생성
            filename = self.filename_format.format(
                rank=clip['rank'],
                video_name=safe_video_name,
                start=clip['start_time'],
                end=clip['end_time']
            )
            
            # 클립 정보 구성
            clip_info = {
                'rank': clip['rank'],
                'start_time': clip['start_time'],
                'end_time': clip['end_time'],
                'duration': clip['duration'],
                'fun_score': clip['fun_score'],
                'video_path': video_path,
                'output_dir': output_dir,
                'filename': filename,
                'output_path': os.path.join(output_dir, f"{filename}.mp4")
            }
            
            # 기존 파일 확인
            if os.path.exists(clip_info['output_path']) and not self.overwrite_existing:
                self.logger.info(f"⚠️ 파일 존재 (건너뛰기): {filename}.mp4")
                with self.progress_lock:
                    self.stats['skipped'] += 1
                continue
            
            clip_queue.append(clip_info)
        
        self.logger.info(f"📋 클립 생성 큐: {len(clip_queue)}개 (건너뛴 것: {self.stats['skipped']}개)")
        return clip_queue
    
    def create_single_clip(self, clip_info: Dict) -> Tuple[bool, str]:
        """
        단일 클립 생성 (FFmpeg 실행)
        
        Args:
            clip_info (Dict): 클립 정보
            
        Returns:
            Tuple[bool, str]: (성공 여부, 메시지)
        """
        try:
            # FFmpeg 명령 구성
            cmd = [
                'ffmpeg',
                '-y',  # 덮어쓰기 허용
            ]
            
            # 정확한 seek 옵션
            if self.accurate_seek:
                cmd.extend(['-ss', str(clip_info['start_time'])])
            
            # 입력 파일
            cmd.extend(['-i', clip_info['video_path']])
            
            # seek이 입력 후에 오는 경우
            if not self.accurate_seek:
                cmd.extend(['-ss', str(clip_info['start_time'])])
            
            # 지속 시간
            cmd.extend(['-t', str(clip_info['duration'])])
            
            # 비디오 설정
            cmd.extend([
                '-c:v', 'libx264',
                '-crf', str(self.video_crf),
                '-preset', self.preset
            ])
            
            # 오디오 설정 (merged 비디오이므로 copy)
            if self.create_merged_clips:
                cmd.extend(['-c:a', 'aac'])
            
            # 타임스탬프 오류 방지
            if self.avoid_negative_ts:
                cmd.extend(['-avoid_negative_ts', 'make_zero'])
            
            # 출력 파일
            cmd.append(clip_info['output_path'])
            
            # FFmpeg 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                # 파일 크기 확인
                if os.path.exists(clip_info['output_path']):
                    file_size = os.path.getsize(clip_info['output_path']) / (1024 * 1024)  # MB
                    return True, f"성공 ({file_size:.1f}MB)"
                else:
                    return False, "출력 파일이 생성되지 않음"
            else:
                return False, f"FFmpeg 오류: {result.stderr}"
                
        except Exception as e:
            return False, f"클립 생성 오류: {e}"
    
    def process_clip_batch(self, clip_queue: List[Dict]) -> Dict:
        """
        클립 배치 처리 (병렬 실행)
        
        Args:
            clip_queue (List[Dict]): 클립 생성 작업 큐
            
        Returns:
            Dict: 처리 결과 통계
        """
        if not clip_queue:
            self.logger.info("📋 처리할 클립이 없습니다.")
            return self.stats
        
        self.logger.info(f"🚀 클립 생성 시작: {len(clip_queue)}개")
        
        # 배치 처리
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # 작업 submit
            future_to_clip = {
                executor.submit(self.create_single_clip, clip_info): clip_info
                for clip_info in clip_queue
            }
            
            # 결과 처리
            for future in concurrent.futures.as_completed(future_to_clip):
                clip_info = future_to_clip[future]
                
                try:
                    success, message = future.result()
                    
                    with self.progress_lock:
                        if success:
                            self.stats['created'] += 1
                            self.logger.info(f"✅ {clip_info['filename']}.mp4 - {message}")
                        else:
                            self.stats['failed'] += 1
                            self.logger.error(f"❌ {clip_info['filename']}.mp4 - {message}")
                        
                        # 진행 상황 출력
                        total_processed = self.stats['created'] + self.stats['failed']
                        progress = (total_processed / len(clip_queue)) * 100
                        
                        if total_processed % max(1, len(clip_queue) // 10) == 0:  # 10% 단위로 출력
                            self.logger.info(f"📊 진행 상황: {total_processed}/{len(clip_queue)} ({progress:.1f}%)")
                
                except Exception as e:
                    with self.progress_lock:
                        self.stats['failed'] += 1
                        self.logger.error(f"❌ {clip_info['filename']}.mp4 - 처리 오류: {e}")
        
        return self.stats
    
    def extract_clips(self, refined_clips_json: str, video_path: str) -> Dict:
        """
        전체 클립 추출 프로세스
        
        Args:
            refined_clips_json (str): refined_clips.json 파일 경로
            video_path (str): 원본 비디오 파일 경로
            
        Returns:
            Dict: 처리 결과 통계
        """
        self.logger.info("🎬 재미 클립 추출 프로세스 시작")
        
        # 1. 클립 데이터 로드
        clips_data = self.load_refined_clips(refined_clips_json)
        
        # 2. 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        file_size = os.path.getsize(video_path) / (1024 * 1024 * 1024)  # GB
        self.logger.info(f"📁 원본 비디오: {os.path.basename(video_path)} ({file_size:.2f}GB)")
        
        # 3. 클립 생성 큐 생성
        clip_queue = self.create_clip_queue(clips_data, video_path)
        
        # 4. 배치 처리
        stats = self.process_clip_batch(clip_queue)
        
        # 5. 결과 요약
        self.logger.info("🎬 재미 클립 추출 완료!")
        self.logger.info(f"   생성됨: {stats['created']}개")
        self.logger.info(f"   건너뜀: {stats['skipped']}개")
        self.logger.info(f"   실패함: {stats['failed']}개")
        
        if stats['created'] > 0:
            video_name = clips_data.get('video_name', 'unknown')
            safe_video_name = PipelineUtils.safe_filename(video_name)
            output_dir = os.path.join(self.output_base, safe_video_name)
            self.logger.info(f"📁 출력 위치: {output_dir}")
        
        return stats
    
    def generate_output_path(self, clip_info: Dict) -> str:
        """
        클립 출력 경로 생성
        
        Args:
            clip_info (Dict): 클립 정보
            
        Returns:
            str: 출력 파일 경로
        """
        return os.path.join(clip_info['output_dir'], f"{clip_info['filename']}.mp4")


def main():
    """테스트 실행"""
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='재미 클립 추출기')
    parser.add_argument('clips_json', nargs='?', help='refined_clips.json 파일 경로')
    parser.add_argument('video_path', nargs='?', help='원본 비디오 파일 경로')
    parser.add_argument('--clips', help='refined_clips.json 파일 경로 (named 인자)')
    parser.add_argument('--video', help='원본 비디오 파일 경로 (named 인자)')
    parser.add_argument('--config', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 인자 처리 (positional 또는 named)
    clips_json = args.clips_json if args.clips_json else args.clips
    video_path = args.video_path if args.video_path else args.video
    
    # 필수 인자 확인
    if not clips_json:
        print("❌ 오류: refined_clips.json 파일 경로가 필요합니다.")
        print("사용법: python fun_clip_extractor.py clips.json video.mp4")
        print("   또는: python fun_clip_extractor.py --clips clips.json --video video.mp4")
        parser.print_help()
        return
    
    if not video_path:
        print("❌ 오류: 원본 비디오 파일 경로가 필요합니다.")
        print("사용법: python fun_clip_extractor.py clips.json video.mp4")
        print("   또는: python fun_clip_extractor.py --clips clips.json --video video.mp4")
        parser.print_help()
        return
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 클립 추출기 실행
        extractor = FunClipExtractor(config_path=args.config)
        stats = extractor.extract_clips(clips_json, video_path)
        
        print(f"\n✅ 클립 추출 완료!")
        print(f"📊 생성됨: {stats['created']}개")
        print(f"   건너뜀: {stats['skipped']}개")
        print(f"   실패함: {stats['failed']}개")
        
        if stats['created'] > 0:
            print(f"📁 클립 저장 위치: {extractor.output_base}")
        
    except Exception as e:
        print(f"❌ 클립 추출 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()