#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 파이프라인 모듈들 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils
from utils.preprocessing_checker import PreprocessingChecker
from modules.highlight_clusterer import HighlightClusterer
from modules.window_generator import WindowGenerator
from modules.clip_selector import ClipSelector
from modules.clip_refiner import ClipRefiner
from modules.fun_clip_extractor import FunClipExtractor


class ClipExtractPipeline:
    """
    재미 클립 추출 통합 파이프라인
    긴 영상에서 재밌는 클립들을 자동으로 찾아서 추출
    """
    
    def __init__(self, config_path: str = "pipeline/configs/funclip_extraction_config.yaml"):
        """
        클립 추출 파이프라인 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        print(f"\n{'='*60}")
        print(f"🎬 재미 클립 추출 시스템 v1.0")
        print(f"{'='*60}")
        
        # 프로젝트 루트로 작업 디렉토리 변경
        os.chdir(project_root)
        
        # 설정 로드
        self.config = PipelineUtils.load_config(config_path)
        
        # 로깅 설정 (간단히)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        # 전처리 체커 초기화
        self.preprocessing_checker = PreprocessingChecker(config_path)
        
        # 실행 모드 (기본: 단계별)
        self.auto_mode = False
        
        # 결과 추적
        self.results = {
            'video_path': None,
            'video_name': None,
            'required_files': {},
            'clusters_json': None,
            'scored_windows_json': None,
            'selected_clips_json': None,
            'refined_clips_json': None,
            'extracted_clips_dir': None,
            'processing_times': {},
            'total_start_time': None
        }
        
        self.logger.info("✅ 클립 추출 파이프라인 초기화 완료")
    
    def run_pipeline(self, video_path: str, auto_mode: bool = False) -> Dict:
        """
        전체 클립 추출 파이프라인 실행
        
        Args:
            video_path (str): 비디오 파일 경로
            auto_mode (bool): 자동 실행 모드
            
        Returns:
            Dict: 전체 실행 결과
        """
        self.results['video_path'] = video_path
        self.results['total_start_time'] = time.time()
        self.auto_mode = auto_mode
        
        try:
            # 실행 모드 확인
            if self.auto_mode:
                print(f"\n🔄 자동 모드로 전체 파이프라인을 실행합니다...")
            else:
                print(f"\n🎮 단계별 모드로 실행합니다. 각 단계마다 Enter를 눌러 진행하세요.")
            
            print(f"📁 처리할 영상: {video_path}")
            
            # 0단계: 전처리 파일 확인
            if not self._check_preprocessing():
                return self.results
            
            # 1단계: 하이라이트 클러스터링
            if not self._run_step_1_clustering():
                return self.results
            
            # 2단계: 윈도우 생성 및 점수 계산
            if not self._run_step_2_window_generation():
                return self.results
            
            # 3단계: 클립 선별
            if not self._run_step_3_clip_selection():
                return self.results
            
            # 4단계: 클립 경계 조정
            if not self._run_step_4_clip_refinement():
                return self.results
            
            # 5단계: 실제 클립 추출
            if not self._run_step_5_clip_extraction():
                return self.results
            
            # 최종 결과 출력
            self._print_final_results()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def _check_preprocessing(self) -> bool:
        """0단계: 전처리 파일 확인 및 준비"""
        PipelineUtils.print_step_banner(0, "전처리 확인", "HDF5 및 텐션 JSON 파일 존재 여부 확인")
        
        try:
            # 전처리 상태 확인 및 자동 수정
            status = self.preprocessing_checker.ensure_preprocessing_complete(
                self.results['video_path'], 
                auto_fix=True
            )
            
            self.results['video_name'] = status['video_name']
            
            # 클립 추출 가능 여부 확인
            if not status['can_extract_clips']:
                print(f"\n❌ 클립 추출 불가능!")
                
                if status.get('requires_full_preprocessing'):
                    print(f"💡 해결 방법: 전체 전처리가 필요합니다.")
                    print(f"   다음 명령을 먼저 실행하세요:")
                    print(f"   python pipeline/integrated_pipeline.py \"{self.results['video_path']}\"")
                elif status.get('auto_fix_failed'):
                    print(f"💡 해결 방법: 텐션 계산을 수동으로 시도해보세요.")
                    print(f"   python tension_analyzer/tension_calculator.py \"{status['video_name']}\"")
                
                return False
            
            # 필요한 파일들 저장
            self.results['required_files'] = self.preprocessing_checker.get_required_files(
                self.results['video_path']
            )
            
            PipelineUtils.print_completion_banner(
                0, "전처리 확인",
                f"상태: {status['status']}, "
                f"비디오 HDF5: ✅, 오디오 HDF5: ✅, 텐션 JSON: ✅"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "전처리 확인")
            
        except Exception as e:
            self.logger.error(f"❌ 전처리 확인 단계 오류: {e}")
            return False
    
    def _run_step_1_clustering(self) -> bool:
        """1단계: 하이라이트 클러스터링"""
        PipelineUtils.print_step_banner(1, "하이라이트 클러스터링", "DBSCAN으로 텐션 하이라이트들을 클러스터링")
        
        try:
            step_start = time.time()
            
            # 클러스터링 실행
            clusterer = HighlightClusterer()
            result = clusterer.process_highlights(self.results['required_files']['tension_json'])
            
            # 결과 저장
            video_name = self.results['video_name']
            safe_video_name = PipelineUtils.safe_filename(video_name)
            output_dir = os.path.join(self.config['output']['base_dir'], safe_video_name)
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = PipelineUtils.get_timestamp()
            clusters_json_path = os.path.join(output_dir, f"clusters_{safe_video_name}_{timestamp}.json")
            
            clusterer.save_clusters(result, clusters_json_path)
            self.results['clusters_json'] = clusters_json_path
            
            step_time = time.time() - step_start
            self.results['processing_times']['clustering'] = step_time
            
            PipelineUtils.print_completion_banner(
                1, "하이라이트 클러스터링",
                f"하이라이트: {result['metadata']['total_highlights']}개 → "
                f"클러스터: {result['metadata']['total_clusters']}개, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "하이라이트 클러스터링")
            
        except Exception as e:
            self.logger.error(f"❌ 1단계 클러스터링 오류: {e}")
            return False
    
    def _run_step_2_window_generation(self) -> bool:
        """2단계: 윈도우 생성 및 점수 계산"""
        PipelineUtils.print_step_banner(2, "윈도우 생성 및 점수 계산", "XGBoost로 각 윈도우별 재미도 점수 계산")
        
        try:
            step_start = time.time()
            
            # 윈도우 생성기 실행
            generator = WindowGenerator()
            result = generator.generate_and_score_windows(
                self.results['clusters_json'],
                self.results['required_files']['video_h5'],
                self.results['required_files']['audio_h5'],
                self.results['required_files']['tension_json']
            )
            
            # 결과 저장
            video_name = self.results['video_name']
            safe_video_name = PipelineUtils.safe_filename(video_name)
            output_dir = os.path.join(self.config['output']['base_dir'], safe_video_name)
            
            timestamp = PipelineUtils.get_timestamp()
            scored_windows_path = os.path.join(output_dir, f"scored_windows_{safe_video_name}_{timestamp}.json")
            
            generator.save_scored_windows(result, scored_windows_path)
            self.results['scored_windows_json'] = scored_windows_path
            
            step_time = time.time() - step_start
            self.results['processing_times']['window_generation'] = step_time
            
            # 통계 정보
            stats = result['metadata']['score_statistics']
            
            PipelineUtils.print_completion_banner(
                2, "윈도우 생성 및 점수 계산",
                f"윈도우: {result['metadata']['total_windows']}개, "
                f"평균 재미도: {stats['mean']:.3f}, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "윈도우 생성 및 점수 계산")
            
        except Exception as e:
            self.logger.error(f"❌ 2단계 윈도우 생성 오류: {e}")
            return False
    
    def _run_step_3_clip_selection(self) -> bool:
        """3단계: 클립 선별"""
        PipelineUtils.print_step_banner(3, "클립 선별", "NMS로 중복 제거하여 최종 클립들 선별")
        
        try:
            step_start = time.time()
            
            # 클립 선별기 실행
            selector = ClipSelector()
            original_data, selected_clips = selector.select_clips(self.results['scored_windows_json'])
            
            # 결과 저장
            clips_path = selector.save_selected_clips(original_data, selected_clips)
            self.results['selected_clips_json'] = clips_path
            
            step_time = time.time() - step_start
            self.results['processing_times']['clip_selection'] = step_time
            
            # 선별 정보
            selection_config = self.config['selection']
            
            PipelineUtils.print_completion_banner(
                3, "클립 선별",
                f"원본 윈도우: {len(original_data['windows'])}개 → "
                f"선별된 클립: {len(selected_clips)}개, "
                f"IoU 임계값: {selection_config['iou_threshold']}, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "클립 선별")
            
        except Exception as e:
            self.logger.error(f"❌ 3단계 클립 선별 오류: {e}")
            return False
    
    def _run_step_4_clip_refinement(self) -> bool:
        """4단계: 클립 경계 조정"""
        PipelineUtils.print_step_banner(4, "클립 경계 조정", "VAD 기반으로 클립 경계를 자연스럽게 조정")
        
        try:
            step_start = time.time()
            
            # 오디오 데이터 로드
            audio_data = PipelineUtils.load_audio_hdf5(self.results['required_files']['audio_h5'])
            if not audio_data:
                self.logger.error("❌ 오디오 데이터 로드 실패")
                return False
            
            # 선택된 클립 데이터 로드
            import json
            with open(self.results['selected_clips_json'], 'r', encoding='utf-8') as f:
                clips_data = json.load(f)
            
            # 클립 경계 조정
            refiner = ClipRefiner(self.config)
            refined_clips = refiner.refine_clips(
                clips_data['clips'], 
                audio_data, 
                self.results['video_name']
            )
            
            # 조정된 클립 파일 찾기 (refiner가 자동 저장)
            video_name = self.results['video_name']
            safe_video_name = PipelineUtils.safe_filename(video_name)
            output_dir = os.path.join(self.config['output']['base_dir'], safe_video_name)
            
            # 최신 refined_clips 파일 찾기
            refined_pattern = os.path.join(output_dir, f"refined_clips_{safe_video_name}_*.json")
            refined_files = glob.glob(refined_pattern)
            if refined_files:
                self.results['refined_clips_json'] = max(refined_files, key=os.path.getmtime)
            
            step_time = time.time() - step_start
            self.results['processing_times']['clip_refinement'] = step_time
            
            # 조정 통계
            extended_count = sum(1 for clip in refined_clips if clip.get('start_extended') or clip.get('end_extended'))
            
            PipelineUtils.print_completion_banner(
                4, "클립 경계 조정",
                f"조정된 클립: {len(refined_clips)}개, "
                f"경계 확장: {extended_count}개, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "클립 경계 조정")
            
        except Exception as e:
            self.logger.error(f"❌ 4단계 클립 경계 조정 오류: {e}")
            return False
    
    def _run_step_5_clip_extraction(self) -> bool:
        """5단계: 실제 클립 추출"""
        PipelineUtils.print_step_banner(5, "실제 클립 추출", "FFmpeg로 MP4 클립 파일들 생성")
        
        try:
            step_start = time.time()
            
            # 클립 추출기 설정 로드
            clip_config_path = "pipeline/configs/clip_generation_config.yaml"
            
            # 클립 추출기 실행
            extractor = FunClipExtractor(clip_config_path)
            stats = extractor.extract_clips(
                self.results['refined_clips_json'], 
                self.results['video_path']
            )
            
            # 추출된 클립 디렉토리 저장
            video_name = self.results['video_name']
            safe_video_name = PipelineUtils.safe_filename(video_name)
            self.results['extracted_clips_dir'] = os.path.join(extractor.output_base, safe_video_name)
            
            step_time = time.time() - step_start
            self.results['processing_times']['clip_extraction'] = step_time
            
            PipelineUtils.print_completion_banner(
                5, "실제 클립 추출",
                f"생성된 클립: {stats['created']}개, "
                f"건너뛴 클립: {stats['skipped']}개, "
                f"실패한 클립: {stats['failed']}개, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "실제 클립 추출")
            
        except Exception as e:
            self.logger.error(f"❌ 5단계 클립 추출 오류: {e}")
            return False
    
    def _print_final_results(self):
        """최종 결과 출력"""
        total_time = time.time() - self.results['total_start_time']
        
        print(f"\n{'='*60}")
        print(f"🎯 재미 클립 추출 완료!")
        print(f"{'='*60}")
        
        # 처리된 파일 정보
        video_name = Path(self.results['video_path']).name
        print(f"📁 처리된 영상: {video_name}")
        print(f"⏱️ 총 소요 시간: {total_time//60:.0f}분 {total_time%60:.0f}초")
        
        # 각 단계별 소요 시간
        print(f"\n📊 단계별 소요시간:")
        step_names = {
            'clustering': '1. 하이라이트 클러스터링',
            'window_generation': '2. 윈도우 생성 및 점수 계산',
            'clip_selection': '3. 클립 선별',
            'clip_refinement': '4. 클립 경계 조정',
            'clip_extraction': '5. 실제 클립 추출'
        }
        
        for step, time_taken in self.results['processing_times'].items():
            step_name = step_names.get(step, step)
            print(f"   {step_name}: {time_taken:.1f}초")
        
        # 결과 파일 위치
        print(f"\n📂 생성된 파일들:")
        
        if self.results['clusters_json']:
            print(f"   🎪 클러스터: {os.path.relpath(self.results['clusters_json'])}")
        
        if self.results['scored_windows_json']:
            print(f"   📊 점수 윈도우: {os.path.relpath(self.results['scored_windows_json'])}")
        
        if self.results['selected_clips_json']:
            print(f"   🎯 선별된 클립: {os.path.relpath(self.results['selected_clips_json'])}")
        
        if self.results['refined_clips_json']:
            print(f"   🔧 조정된 클립: {os.path.relpath(self.results['refined_clips_json'])}")
        
        if self.results['extracted_clips_dir']:
            clips_count = len(glob.glob(os.path.join(self.results['extracted_clips_dir'], "*.mp4")))
            print(f"   🎬 추출된 클립: {clips_count}개 MP4 파일")
            print(f"      위치: {os.path.relpath(self.results['extracted_clips_dir'])}/")
        
        print(f"\n✨ 클립 추출 완료: {self.results['video_name']}")
        print(f"{'='*60}")


def find_video_file(input_path: str, base_dir: str = "data/videos") -> Optional[str]:
    """
    비디오 파일 찾기 (integrated_pipeline과 동일한 로직)
    
    Args:
        input_path (str): 입력 경로 또는 파일명
        base_dir (str): 기본 검색 디렉토리
        
    Returns:
        str: 찾은 비디오 파일 경로 (없으면 None)
    """
    # 1. 절대 경로인지 확인
    if os.path.isabs(input_path) and os.path.exists(input_path):
        return input_path
    
    # 2. 상대 경로인지 확인
    if os.path.exists(input_path):
        return input_path
    
    # 3. 현재 디렉토리에서 확인
    if os.path.exists(os.path.basename(input_path)):
        return os.path.basename(input_path)
    
    # 4. 기본 디렉토리에서 패턴 매칭
    if os.path.exists(base_dir):
        # 정확한 파일명 매칭
        exact_path = os.path.join(base_dir, input_path)
        if os.path.exists(exact_path):
            return exact_path
        
        # 패턴 매칭 (확장자 없이 입력된 경우)
        name_without_ext = os.path.splitext(input_path)[0]
        patterns = [
            f"{input_path}",
            f"{name_without_ext}.*",
            f"*{input_path}*",
            f"*{name_without_ext}*"
        ]
        
        for pattern in patterns:
            search_pattern = os.path.join(base_dir, pattern)
            matches = glob.glob(search_pattern)
            
            # 비디오 파일만 필터링
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            video_matches = [m for m in matches 
                           if any(m.lower().endswith(ext) for ext in video_extensions)]
            
            if video_matches:
                return video_matches[0]  # 첫 번째 매치 반환
    
    return None


def get_video_input() -> Optional[str]:
    """
    대화형 비디오 파일 입력 (integrated_pipeline과 동일한 로직)
    
    Returns:
        str: 선택된 비디오 파일 경로
    """
    print(f"\n📁 비디오 파일을 지정하세요:")
    print(f"   1. 파일명만 입력 (data/videos/에서 검색)")
    print(f"   2. 상대/절대 경로 입력")
    print(f"   예시: clip.mp4, videos/clip.mp4, D:/videos/clip.mp4")
    
    while True:
        user_input = input("입력: ").strip()
        
        if not user_input:
            print("❌ 파일명을 입력해주세요.")
            continue
        
        # 비디오 파일 찾기
        video_path = find_video_file(user_input)
        
        if video_path:
            # 파일 정보 표시
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"✅ 비디오 파일 확인: {video_path}")
            print(f"   크기: {file_size:.1f}MB")
            return video_path
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {user_input}")
            print(f"   data/videos/ 디렉토리를 확인하거나 전체 경로를 입력해주세요.")
            
            retry = input("다시 시도하시겠습니까? (y/n): ").strip().lower()
            if retry != 'y':
                return None


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='재미 클립 추출 통합 파이프라인')
    parser.add_argument('video_input', nargs='?', help='비디오 파일 경로 또는 파일명')
    parser.add_argument('--config', '-c', default='pipeline/configs/funclip_extraction_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='자동 모드로 실행 (단계별 대기 없음)')
    
    args = parser.parse_args()
    
    try:
        # 1. 비디오 파일 경로 결정
        if args.video_input:
            # 명령줄 인자로 제공됨
            video_path = find_video_file(args.video_input)
            if not video_path:
                print(f"❌ 비디오 파일을 찾을 수 없습니다: {args.video_input}")
                return 1
        else:
            # 대화형 입력
            video_path = get_video_input()
            if not video_path:
                print("❌ 비디오 파일이 지정되지 않았습니다.")
                return 1
        
        # 2. 파이프라인 실행
        pipeline = ClipExtractPipeline(args.config)
        results = pipeline.run_pipeline(video_path, auto_mode=args.auto)
        
        # 3. 성공 여부 확인
        if results.get('extracted_clips_dir'):
            return 0
        else:
            return 1
        
    except KeyboardInterrupt:
        print(f"\n🛑 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())