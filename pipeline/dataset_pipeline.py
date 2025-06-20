#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import yaml
import argparse
from pathlib import Path

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 모듈들 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils
from modules.emotion_highlighter import EmotionHighlighter
from modules.thumbnail_classifier import ThumbnailClassifier
from utils.batch_processor import BatchProcessor

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
    
    def __init__(self, config_path: str = "pipeline/configs/dataset_config.yaml"):
        """통합 파이프라인 초기화"""
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
    
    def run_pipeline(self, video_path: str) -> dict:
        """전체 파이프라인 실행"""
        self.results['video_path'] = video_path
        self.results['total_start_time'] = time.time()
        
        try:
            # 실행 모드 확인
            if self.auto_mode:
                print(f"\n🔄 자동 모드로 전체 파이프라인을 실행합니다...")
            else:
                print(f"\n🎮 단계별 모드로 실행합니다. 각 단계마다 Enter를 눌러 진행하세요.")
            
            print(f"📁 처리할 영상: {video_path}")
            
            # 1~5단계 + 시각화
            if not self._run_step_1_video_preprocessing():
                return self.results
            if not self._run_step_2_audio_preprocessing():
                return self.results
            if not self._run_step_3_tension_calculation():
                return self.results
            if not self._run_step_4_emotion_highlights():
                return self.results
            if not self._run_step_5_thumbnail_classification():
                return self.results
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
            
            video_config_path = "pipeline/configs/dataset_config.yaml"
            processor = LongVideoProcessor(video_config_path)
            
            result = processor.process_long_video(self.results['video_path'])
            
            if result and 'hdf5_path' in result:
                self.results['video_hdf5'] = result['hdf5_path']
                
                step_time = time.time() - step_start
                self.results['processing_times']['video_preprocessing'] = step_time
                
                stats = result.get('stats', {})
                
                PipelineUtils.print_completion_banner(
                    1, "비디오 전처리", 
                    f"프레임: {stats.get('frames_processed', 0):,}개, "
                    f"침착맨 얼굴: {stats.get('chimchakman_faces', 0):,}개, "
                    f"소요시간: {step_time:.1f}초"
                )
                
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
            
            audio_config_path = "pipeline/configs/dataset_config.yaml"
            preprocessor = LongVideoAudioPreprocessor(audio_config_path)
            
            result = preprocessor.preprocess_long_video_audio(self.results['video_path'])
            
            if result and 'hdf5_path' in result:
                self.results['audio_hdf5'] = result['hdf5_path']
                
                step_time = time.time() - step_start
                self.results['processing_times']['audio_preprocessing'] = step_time
                
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
            
            tension_config_path = "pipeline/configs/dataset_config.yaml"
            calculator = MultiEmotionTensionCalculator(tension_config_path)
            
            video_name = Path(self.results['video_path']).stem
            filename_pattern = video_name
            
            result = calculator.calculate_tension(filename_pattern)
            
            if result:
                tension_output_dir = calculator.tension_output_dir
                json_files = glob.glob(os.path.join(tension_output_dir, f"tension_{filename_pattern}*.json"))
                
                if json_files:
                    self.results['tension_json'] = json_files[-1]
                
                step_time = time.time() - step_start
                self.results['processing_times']['tension_calculation'] = step_time
                
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
            
            highlighter = EmotionHighlighter(self.config)
            highlights_dir = os.path.join(self.output_dirs['highlights'])
            
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
            
            classifier = ThumbnailClassifier(self.config)
            classification_dir = os.path.join(self.output_dirs['classification'])
            
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
            
            visualizer = TensionVisualizer(config=self.config)
            
            if visualizer.load_tension_data(self.results['tension_json']):
                video_name = Path(self.results['video_path']).stem
                safe_video_name = PipelineUtils.safe_filename(video_name)
                
                viz_dir = os.path.join(self.output_dirs['visualization'], safe_video_name)
                os.makedirs(viz_dir, exist_ok=True)
                
                self.results['visualization_dir'] = viz_dir
                
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
        
        video_name = Path(self.results['video_path']).name
        print(f"📁 처리된 영상: {video_name}")
        print(f"⏱️ 총 소요 시간: {total_time//60:.0f}분 {total_time%60:.0f}초")
        
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


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='침착맨 AI 재미도 분석 통합 파이프라인')
    parser.add_argument('video_input', nargs='?', help='비디오 파일 경로 또는 파일명')
    parser.add_argument('--config', '-c', default='pipeline/configs/dataset_config.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='자동 모드로 실행 (단계별 대기 없음)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='배치 처리 모드로 실행')
    
    args = parser.parse_args()
    
    try:
        # 1. 처리 모드 결정
        if args.batch:
            print("DEBUG: 배치 모드 선택됨")
            mode = 'batch'
        elif args.video_input:
            print("DEBUG: 단일 파일 모드 선택됨")
            mode = 'single'
        else:
            print("DEBUG: 대화형 모드로 진입")
            mode = PipelineUtils.get_user_choice()
            if mode == 'quit':
                print("👋 프로그램을 종료합니다.")
                return 0
        
        print(f"DEBUG: 최종 모드 = {mode}")
        
        # 2. 배치 처리 모드
        if mode == 'batch':
            # 설정 로드
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 배치 처리기 생성 및 실행
            batch_processor = BatchProcessor(config)
            results = batch_processor.run_batch_processing()
            
            if results['success'] and results['failed'] == 0:
                return 0
            else:
                return 1
        
        # 3. 단일 파일 처리 모드
        else:
            if not args.video_input:
                # 대화형 입력
                video_path = PipelineUtils.get_video_input()
                if not video_path:
                    print("❌ 비디오 파일이 지정되지 않았습니다.")
                    return 1
            else:
                # 명령줄 인자로 제공됨
                video_path = PipelineUtils.find_video_file(args.video_input)
                if not video_path:
                    print(f"❌ 비디오 파일을 찾을 수 없습니다: {args.video_input}")
                    return 1
            
            # 자동 모드 설정 조정
            if args.auto:
                with open(args.config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                config['pipeline']['execution']['auto_mode'] = 0
                
                temp_config = args.config.replace('.yaml', '_temp.yaml')
                with open(temp_config, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                args.config = temp_config
            
            # 단일 파일 파이프라인 실행
            pipeline = IntegratedPipeline(args.config)
            results = pipeline.run_pipeline(video_path)
            
            # 임시 설정 파일 정리
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