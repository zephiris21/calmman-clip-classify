#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
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
from modules.emotion_highlighter import EmotionHighlighter
from modules.thumbnail_classifier import ThumbnailClassifier

# 기존 전처리 모듈들 import
from video_analyzer.inference_prep.video_preprocessor import LongVideoProcessor
from video_analyzer.inference_prep.audio_preprocessor import LongVideoAudioPreprocessor
from tension_analyzer.tension_calculator import MultiEmotionTensionCalculator
from tension_analyzer.tension_visualizer import TensionVisualizer


class IntegratedPipeline:
    """
    침착맨 AI 재미도 분석 통합 파이프라인
    비디오 전처리부터 시각화까지 전체 과정을 관리
    """
    
    def __init__(self, config_path: str = "pipeline/configs/integrated_config.yaml"):
        """
        통합 파이프라인 초기화
        
        Args:
            config_path (str): 통합 설정 파일 경로
        """
        print(f"\n{'='*60}")
        print(f"🎬 침착맨 AI 재미도 분석 시스템 v2.0")
        print(f"{'='*60}")
        
        # 설정 로드
        self.config = PipelineUtils.load_config(config_path)
        
        # 출력 디렉토리 생성
        self.output_dirs = PipelineUtils.setup_output_directories(self.config)
        
        # 로깅 설정
        self.logger = PipelineUtils.setup_logging(self.config, self.output_dirs)
        
        # 파이프라인 실행 모드
        self.auto_mode = self.config['pipeline']['execution']['auto_mode'] == 0
        
        # 결과 추적
        self.results = {
            'video_path': None,
            'video_hdf5': None,
            'audio_hdf5': None,
            'tension_json': None,
            'highlights_dir': None,
            'classification_dir': None,
            'visualization_dir': None,
            'processing_times': {},
            'total_start_time': None
        }
        
        self.logger.info("✅ 통합 파이프라인 초기화 완료")
    
    def run_pipeline(self, video_path: str) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            Dict: 전체 실행 결과
        """
        self.results['video_path'] = video_path
        self.results['total_start_time'] = time.time()
        
        try:
            # 실행 모드 확인
            if self.auto_mode:
                print(f"\n🔄 자동 모드로 전체 파이프라인을 실행합니다...")
            else:
                print(f"\n🎮 단계별 모드로 실행합니다. 각 단계마다 Enter를 눌러 진행하세요.")
            
            print(f"📁 처리할 영상: {video_path}")
            
            # 1단계: 비디오 전처리
            if not self._run_step_1_video_preprocessing():
                return self.results
            
            # 2단계: 오디오 전처리
            if not self._run_step_2_audio_preprocessing():
                return self.results
            
            # 3단계: 텐션 계산
            if not self._run_step_3_tension_calculation():
                return self.results
            
            # 4단계: 감정 하이라이트 추출
            if not self._run_step_4_emotion_highlights():
                return self.results
            
            # 5단계: 썸네일 분류
            if not self._run_step_5_thumbnail_classification():
                return self.results
            
            # 시각화 단계
            if not self._run_step_viz_visualization():
                return self.results
            
            # 최종 결과 출력
            self._print_final_results()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return self.results
    
    def _run_step_1_video_preprocessing(self) -> bool:
        """1단계: 비디오 전처리"""
        PipelineUtils.print_step_banner(1, "비디오 전처리", "얼굴 탐지 + 감정 분석하여 HDF5 저장")
        
        try:
            step_start = time.time()
            
            # 비디오 전처리기 초기화 (기존 설정 활용)
            video_config_path = "pipeline/configs/integrated_config.yaml"
            processor = LongVideoProcessor(video_config_path)
            
            # 비디오 처리
            result = processor.process_long_video(self.results['video_path'])
            
            if result and 'hdf5_path' in result:
                self.results['video_hdf5'] = result['hdf5_path']
                
                step_time = time.time() - step_start
                self.results['processing_times']['video_preprocessing'] = step_time
                
                # 결과 정보 추출
                stats = result.get('stats', {})
                total_faces = stats.get('chimchakman_faces', 0) + stats.get('other_faces_filtered', 0)
                
                PipelineUtils.print_completion_banner(
                    1, "비디오 전처리", 
                    f"프레임: {stats.get('frames_processed', 0):,}개, "
                    f"침착맨 얼굴: {stats.get('chimchakman_faces', 0):,}개, "
                    f"소요시간: {step_time:.1f}초"
                )
                
                # 단계별 모드에서 사용자 입력 대기
                return PipelineUtils.wait_for_user_input(self.auto_mode, "비디오 전처리")
            else:
                self.logger.error("❌ 비디오 전처리 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 1단계 비디오 전처리 오류: {e}")
            return False
    
    def _run_step_2_audio_preprocessing(self) -> bool:
        """2단계: 오디오 전처리"""
        PipelineUtils.print_step_banner(2, "오디오 전처리", "VAD + RMS 분석하여 HDF5 저장")
        
        try:
            step_start = time.time()
            
            # 오디오 전처리기 초기화 (기존 설정 활용)
            audio_config_path = "pipeline/configs/integrated_config.yaml"
            preprocessor = LongVideoAudioPreprocessor(audio_config_path)
            
            # 오디오 처리
            result = preprocessor.preprocess_long_video_audio(self.results['video_path'])
            
            if result and 'hdf5_path' in result:
                self.results['audio_hdf5'] = result['hdf5_path']
                
                step_time = time.time() - step_start
                self.results['processing_times']['audio_preprocessing'] = step_time
                
                # VAD 통계 정보
                vad_stats = result['metadata']['vad_statistics']
                
                PipelineUtils.print_completion_banner(
                    2, "오디오 전처리",
                    f"발화 비율: {vad_stats['voice_activity_ratio']:.1%}, "
                    f"총 프레임: {result['metadata']['total_frames']:,}개, "
                    f"소요시간: {step_time:.1f}초"
                )
                
                return PipelineUtils.wait_for_user_input(self.auto_mode, "오디오 전처리")
            else:
                self.logger.error("❌ 오디오 전처리 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 2단계 오디오 전처리 오류: {e}")
            return False
    
    def _run_step_3_tension_calculation(self) -> bool:
        """3단계: 텐션 계산"""
        PipelineUtils.print_step_banner(3, "텐션 계산", "감정 + 오디오 데이터로 텐션 타임라인 생성")
        
        try:
            step_start = time.time()
            
            # 텐션 계산기 초기화
            tension_config_path = "pipeline/configs/integrated_config.yaml"
            calculator = MultiEmotionTensionCalculator(tension_config_path)
            
            # 파일명 패턴 추출 (비디오명에서)
            video_name = Path(self.results['video_path']).stem
            filename_pattern = video_name
            
            # 텐션 계산
            result = calculator.calculate_tension(filename_pattern)
            
            if result:
                # JSON 파일 경로 찾기 (생성된 파일에서)
                tension_output_dir = calculator.tension_output_dir
                json_files = glob.glob(os.path.join(tension_output_dir, f"tension_{filename_pattern}*.json"))
                
                if json_files:
                    self.results['tension_json'] = json_files[-1]  # 최신 파일
                
                step_time = time.time() - step_start
                self.results['processing_times']['tension_calculation'] = step_time
                
                # 통계 정보
                stats = result['statistics']
                
                PipelineUtils.print_completion_banner(
                    3, "텐션 계산",
                    f"평균 텐션: {stats['avg_tension']:.2f}, "
                    f"하이라이트: {stats['highlight_count']}개, "
                    f"소요시간: {step_time:.1f}초"
                )
                
                return PipelineUtils.wait_for_user_input(self.auto_mode, "텐션 계산")
            else:
                self.logger.error("❌ 텐션 계산 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 3단계 텐션 계산 오류: {e}")
            return False
    
    def _run_step_4_emotion_highlights(self) -> bool:
        """4단계: 감정 하이라이트 추출"""
        PipelineUtils.print_step_banner(4, "감정 하이라이트 추출", "각 감정별 상위 순간들의 얼굴 이미지 저장")
        
        try:
            step_start = time.time()
            
            # 감정 하이라이트 추출기 초기화
            highlighter = EmotionHighlighter(self.config)
            
            # 출력 디렉토리 설정
            highlights_dir = os.path.join(self.output_dirs['highlights'])
            
            # 하이라이트 추출
            result = highlighter.extract_highlights(self.results['video_hdf5'], highlights_dir)
            
            if result['extracted_count'] > 0:
                self.results['highlights_dir'] = highlights_dir
                
                step_time = time.time() - step_start
                self.results['processing_times']['emotion_highlights'] = step_time
                
                PipelineUtils.print_completion_banner(
                    4, "감정 하이라이트 추출",
                    f"추출된 하이라이트: {result['extracted_count']}개, "
                    f"소요시간: {step_time:.1f}초"
                )
                
                return PipelineUtils.wait_for_user_input(self.auto_mode, "감정 하이라이트 추출")
            else:
                self.logger.warning("⚠️ 추출된 감정 하이라이트가 없습니다")
                return PipelineUtils.wait_for_user_input(self.auto_mode, "감정 하이라이트 추출")
                
        except Exception as e:
            self.logger.error(f"❌ 4단계 감정 하이라이트 추출 오류: {e}")
            return False
    
    def _run_step_5_thumbnail_classification(self) -> bool:
        """5단계: 썸네일 분류"""
        PipelineUtils.print_step_banner(5, "썸네일 분류", "과장된 표정의 썸네일용 얼굴 이미지 선별")
        
        try:
            step_start = time.time()
            
            # 썸네일 분류기 초기화
            classifier = ThumbnailClassifier(self.config)
            
            # 출력 디렉토리 설정
            classification_dir = os.path.join(self.output_dirs['classification'])
            
            # 썸네일 분류 (HDF5에서 자동으로 얼굴 폴더 찾기)
            result = classifier.classify_faces_from_hdf5(self.results['video_hdf5'], classification_dir)
            
            self.results['classification_dir'] = classification_dir
            
            step_time = time.time() - step_start
            self.results['processing_times']['thumbnail_classification'] = step_time
            
            PipelineUtils.print_completion_banner(
                5, "썸네일 분류",
                f"처리된 얼굴: {result['classified_count']}개, "
                f"썸네일: {result['saved_count']}개, "
                f"소요시간: {step_time:.1f}초"
            )
            
            return PipelineUtils.wait_for_user_input(self.auto_mode, "썸네일 분류")
                
        except Exception as e:
            self.logger.error(f"❌ 5단계 썸네일 분류 오류: {e}")
            return False
    
    def _run_step_viz_visualization(self) -> bool:
        """시각화 단계"""
        PipelineUtils.print_step_banner("viz", "시각화", "텐션 곡선 및 감정 분석 그래프 생성")
        
        try:
            step_start = time.time()
            
            if not self.results['tension_json']:
                self.logger.warning("⚠️ 텐션 데이터가 없어 시각화를 건너뜁니다")
                return True
            
            # OpenMP 충돌 방지
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            # matplotlib 백엔드 설정
            import matplotlib
            matplotlib.use('Agg')  # GUI 없는 백엔드 사용
            import matplotlib.pyplot as plt
            
            # 시각화 도구 초기화
            visualizer = TensionVisualizer(config=self.config)
            
            # 텐션 데이터 로드
            if visualizer.load_tension_data(self.results['tension_json']):
                # 비디오명 추출 및 안전한 폴더명 생성
                video_name = Path(self.results['video_path']).stem
                safe_video_name = PipelineUtils.safe_filename(video_name)
                
                # 비디오별 시각화 디렉토리 생성
                viz_dir = os.path.join(self.output_dirs['visualization'], safe_video_name)
                os.makedirs(viz_dir, exist_ok=True)
                
                self.results['visualization_dir'] = viz_dir
                
                # 자동으로 얼굴 폴더 감지
                visualizer.auto_find_faces_dir()
                
                # 1. 텐션 곡선 저장
                visualizer.plot_tension_curves(figsize=(16, 10))
                tension_plot_path = os.path.join(viz_dir, "tension_curves.png")
                plt.savefig(tension_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # 2. 감정 분포 저장
                visualizer.plot_emotion_pie_chart(figsize=(12, 8))
                emotion_plot_path = os.path.join(viz_dir, "emotion_distribution.png")
                plt.savefig(emotion_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # 3. 감정 피크 얼굴 저장
                visualizer.show_emotion_peak_faces(figsize=(16, 10))
                faces_plot_path = os.path.join(viz_dir, "emotion_peak_faces.png")
                plt.savefig(faces_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                step_time = time.time() - step_start
                self.results['processing_times']['visualization'] = step_time
                
                PipelineUtils.print_completion_banner(
                    "viz", "시각화",
                    f"생성된 그래프: 3개 (텐션곡선, 감정분포, 피크얼굴), "
                    f"소요시간: {step_time:.1f}초"
                )
                
                return PipelineUtils.wait_for_user_input(self.auto_mode, "시각화")
            else:
                self.logger.error("❌ 텐션 데이터 로드 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 시각화 단계 오류: {e}")
            return False
    
    def _print_final_results(self):
        """최종 결과 출력"""
        total_time = time.time() - self.results['total_start_time']
        
        print(f"\n{'='*60}")
        print(f"🎯 전체 파이프라인 완료!")
        print(f"{'='*60}")
        
        # 처리된 파일 정보
        video_name = Path(self.results['video_path']).name
        print(f"📁 처리된 영상: {video_name}")
        print(f"⏱️ 총 소요 시간: {total_time//60:.0f}분 {total_time%60:.0f}초")
        
        # 각 단계별 소요 시간
        print(f"\n📊 단계별 소요시간:")
        for step, time_taken in self.results['processing_times'].items():
            step_name = {
                'video_preprocessing': '비디오 전처리',
                'audio_preprocessing': '오디오 전처리', 
                'tension_calculation': '텐션 계산',
                'emotion_highlights': '감정 하이라이트',
                'thumbnail_classification': '썸네일 분류',
                'visualization': '시각화'
            }.get(step, step)
            print(f"   {step_name}: {time_taken:.1f}초")
        
        # 결과 파일 위치
        print(f"\n📂 결과 저장 위치: {self.output_dirs['base']}/")
        
        if self.results['video_hdf5']:
            print(f"   📊 비디오 HDF5: {os.path.relpath(self.results['video_hdf5'])}")
        if self.results['audio_hdf5']:
            print(f"   🎵 오디오 HDF5: {os.path.relpath(self.results['audio_hdf5'])}")
        if self.results['tension_json']:
            print(f"   ⚡ 텐션 분석: {os.path.relpath(self.results['tension_json'])}")
        if self.results['highlights_dir']:
            highlights_count = len(glob.glob(os.path.join(self.results['highlights_dir'], "*.jpg")))
            print(f"   🎭 감정 하이라이트: {highlights_count}개 이미지")
        if self.results['classification_dir']:
            thumbnails_count = len(glob.glob(os.path.join(self.results['classification_dir'], "*.jpg")))
            print(f"   🖼️ 썸네일: {thumbnails_count}개 이미지")
        if self.results['visualization_dir']:
            viz_count = len(glob.glob(os.path.join(self.results['visualization_dir'], "*.png")))
            print(f"   📈 시각화: {viz_count}개 그래프")
        
        print(f"\n✨ 분석 완료: {video_name}")
        print(f"{'='*60}")


def find_video_file(input_path: str, base_dir: str = "data/videos") -> Optional[str]:
    """
    비디오 파일 찾기 (패턴 매칭 지원)
    
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
    대화형 비디오 파일 입력
    
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
    parser = argparse.ArgumentParser(description='침착맨 AI 재미도 분석 통합 파이프라인')
    parser.add_argument('video_input', nargs='?', help='비디오 파일 경로 또는 파일명')
    parser.add_argument('--config', '-c', default='pipeline/configs/integrated_config.yaml',
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
        
        # 2. 설정 파일에서 실행 모드 조정 (명령줄 옵션 우선)
        if args.auto:
            # 임시로 설정 수정
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            config['pipeline']['execution']['auto_mode'] = 0
            
            # 임시 설정 파일 생성
            temp_config = args.config.replace('.yaml', '_temp.yaml')
            with open(temp_config, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            args.config = temp_config
        
        # 3. 파이프라인 실행
        pipeline = IntegratedPipeline(args.config)
        results = pipeline.run_pipeline(video_path)
        
        # 4. 임시 설정 파일 정리
        if args.auto and os.path.exists(args.config) and 'temp' in args.config:
            os.remove(args.config)
        
        return 0
        
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