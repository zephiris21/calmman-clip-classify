#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
from pathlib import Path
from typing import Dict, List
import sys

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.pipeline_utils import PipelineUtils
from tension_analyzer.tension_calculator import MultiEmotionTensionCalculator


class BatchProcessor:
    """
    배치 처리 전용 클래스
    여러 비디오 파일을 순차적으로 처리
    """
    
    def __init__(self, config: Dict):
        """
        배치 처리기 초기화
        
        Args:
            config (Dict): 설정 정보
        """
        self.config = config
    
    def scan_clips_directory(self) -> List[Dict]:
        """
        클립 디렉토리 스캔하여 처리할 파일 목록 생성
        
        Returns:
            List[Dict]: 처리할 파일 정보 리스트
        """
        clips_info = []
        base_dir = self.config['dataset']['input_base_dir']
        extensions = self.config['dataset']['clip_extensions']
        label_mapping = self.config['dataset']['label_mapping']
        
        print(f"📁 클립 디렉토리 스캔: {base_dir}")
        
        for category, label in label_mapping.items():
            category_dir = os.path.join(base_dir, category)
            
            if not os.path.exists(category_dir):
                print(f"⚠️ 디렉토리 없음: {category_dir}")
                continue
            
            # 비디오 파일들 찾기
            video_files = []
            for ext in extensions:
                pattern = os.path.join(category_dir, f"*{ext}")
                video_files.extend(glob.glob(pattern))
            
            for video_path in video_files:
                video_name = Path(video_path).stem
                clips_info.append({
                    'path': video_path,
                    'name': video_name,
                    'label': label,
                    'category': category
                })
        
        # 파일명으로 정렬
        clips_info.sort(key=lambda x: x['name'])
        
        return clips_info
    
    def check_processing_status(self, clip_info: Dict) -> Dict:
        """
        클립 처리 상태 상세 확인 (3단계 파일 모두)
        
        Returns:
            Dict: {
                'video_hdf5': str|None,
                'audio_hdf5': str|None,
                'tension_json': str|None,
                'missing_steps': List[str],
                'status': str  # 'complete', 'partial', 'none'
            }
        """
        output_dirs = {
            'preprocessed': os.path.join(self.config['output']['base_dir'], 
                                       self.config['output']['preprocessed_dir'])
        }
        
        # 1. HDF5 파일들 찾기
        video_hdf5_path, audio_hdf5_path = PipelineUtils.find_hdf5_files(output_dirs, clip_info['name'])
        
        # 2. 텐션 JSON 파일 찾기
        tension_json_path = None
        tension_dir = os.path.join(self.config['output']['base_dir'], 
                          self.config['output']['tension_analysis_dir'])
        if os.path.exists(tension_dir):
            for file in os.listdir(tension_dir):
                if file.startswith(f'tension_{clip_info["name"]}') and file.endswith('.json'):
                    tension_json_path = os.path.join(tension_dir, file)
                    break
        
        # 3. 상태 판정
        missing_steps = []
        
        if not video_hdf5_path or not os.path.exists(video_hdf5_path):
            missing_steps.extend(['video_preprocessing', 'audio_preprocessing', 'tension_calculation'])
        elif not audio_hdf5_path or not os.path.exists(audio_hdf5_path):
            missing_steps.extend(['audio_preprocessing', 'tension_calculation'])
        elif not tension_json_path or not os.path.exists(tension_json_path):
            missing_steps.append('tension_calculation')
        
        # 상태 결정
        if not missing_steps:
            status = 'complete'
        elif len(missing_steps) == 3:
            status = 'none'
        else:
            status = 'partial'
        
        return {
            'video_hdf5': video_hdf5_path,
            'audio_hdf5': audio_hdf5_path,
            'tension_json': tension_json_path,
            'missing_steps': missing_steps,
            'status': status
        }
    
    def print_batch_summary(self, clips_info: List[Dict], processing_statuses: List[Dict]) -> bool:
        """배치 처리 요약 정보 출력"""
        print(f"\n🎬 배치 처리 준비 완료!")
        print(f"{'='*60}")
        
        # 통계 계산
        total_count = len(clips_info)
        complete_count = len([s for s in processing_statuses if s['status'] == 'complete'])
        partial_count = len([s for s in processing_statuses if s['status'] == 'partial'])
        none_count = len([s for s in processing_statuses if s['status'] == 'none'])
        
        # 카테고리별 분류
        funny_total = len([c for c in clips_info if c['category'] == 'funny'])
        boring_total = len([c for c in clips_info if c['category'] == 'boring'])
        
        print(f"📊 전체 파일: {total_count}개")
        print(f"   📁 funny: {funny_total}개")
        print(f"   📁 boring: {boring_total}개")
        print(f"")
        print(f"✅ 완전 처리됨: {complete_count}개")
        print(f"🔄 부분 처리됨: {partial_count}개 (텐션 계산만 필요)")
        print(f"⏳ 미처리: {none_count}개")
        
        # 처리 필요한 파일이 없으면
        need_processing = partial_count + none_count
        if need_processing == 0:
            print(f"🎉 모든 파일이 이미 처리되었습니다!")
            return False
        
        print(f"")
        print(f"🚀 처리 예정: {need_processing}개")
        print(f"{'='*60}")
        return True
    
    def run_partial_pipeline(self, clip_info: Dict, processing_status: Dict) -> Dict:
        """
        부분 파이프라인 실행 (필요한 단계만)
        
        Returns:
            Dict: 실행 결과
        """
        missing_steps = processing_status['missing_steps']
        
        print(f"🔄 부분 처리: {clip_info['name']} - {', '.join(missing_steps)}")
        
        # 전체 파이프라인 실행이 필요한 경우
        if 'video_preprocessing' in missing_steps:
            # IntegratedPipeline 임포트 (순환 임포트 방지)
            from dataset_pipeline import IntegratedPipeline
            
            pipeline = IntegratedPipeline(config_path="pipeline/configs/dataset_config.yaml")
            pipeline.auto_mode = True  # 자동 모드 강제
            result = pipeline.run_pipeline(clip_info['path'])
            
            # 수정: video_hdf5가 있으면 성공으로 판정
            if result and result.get('video_hdf5'):
                return {'success': True, 'full_pipeline': True}
            else:
                return {'success': False, 'error': 'pipeline_failed'}
        
        # 텐션 계산만 필요한 경우
        elif 'tension_calculation' in missing_steps:
            try:
                tension_config_path = "pipeline/configs/dataset_config.yaml"
                calculator = MultiEmotionTensionCalculator(tension_config_path)
                
                result = calculator.calculate_tension(clip_info['name'])
                
                if result:
                    return {'success': True, 'partial': True, 'steps': ['tension_calculation']}
                else:
                    return {'success': False, 'error': 'tension_calculation_failed'}
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': True, 'skipped': True}
    
    def run_batch_processing(self) -> Dict:
        """
        배치 처리 메인 함수
        
        Returns:
            Dict: 배치 처리 결과
        """
        batch_start_time = time.time()
        
        # 1. 클립 디렉토리 스캔
        clips_info = self.scan_clips_directory()
        
        if not clips_info:
            print("❌ 처리할 클립 파일을 찾을 수 없습니다.")
            return {'success': False, 'total': 0, 'processed': 0, 'failed': 0}
        
        # 2. 모든 파일의 처리 상태 확인
        print("🔍 파일별 처리 상태 확인 중...")
        processing_statuses = []
        for clip_info in clips_info:
            status = self.check_processing_status(clip_info)
            processing_statuses.append(status)
        
        # 3. 배치 요약 출력
        if not self.print_batch_summary(clips_info, processing_statuses):
            return {'success': True, 'total': len(clips_info), 'processed': 0, 'failed': 0}
        
        # 4. 처리 필요한 파일들 필터링
        need_processing = []
        for i, (clip_info, status) in enumerate(zip(clips_info, processing_statuses)):
            if status['status'] != 'complete':
                need_processing.append((clip_info, status))
        
        # 5. 배치 처리 시작
        print(f"🚀 배치 처리 시작...\n")
        
        results = {
            'success': True,
            'total': len(clips_info),
            'processed': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for i, (clip_info, status) in enumerate(need_processing):
            clip_start_time = time.time()
            clip_name = Path(clip_info['path']).name
            
            try:
                print(f"\n[{i+1}/{len(need_processing)}] 처리 중: {clip_name}")
                print(f"   상태: {status['status']} - 필요한 단계: {', '.join(status['missing_steps'])}")
                
                # 부분 파이프라인 실행
                result = self.run_partial_pipeline(clip_info, status)
                
                # 처리 시간 계산
                clip_elapsed = time.time() - clip_start_time
                
                if result.get('success'):
                    if result.get('skipped'):
                        print(f"   ⏭️ 스킵됨 ({clip_elapsed:.1f}초)")
                    elif result.get('partial'):
                        print(f"   ✅ 부분 처리 완료 ({clip_elapsed:.1f}초)")
                    else:
                        print(f"   ✅ 전체 처리 완료 ({clip_elapsed:.1f}초)")
                    results['processed'] += 1
                else:
                    print(f"   ❌ 처리 실패: {result.get('error', 'unknown')} ({clip_elapsed:.1f}초)")
                    results['failed'] += 1
                    results['failed_files'].append(clip_info['path'])
                
            except Exception as e:
                clip_elapsed = time.time() - clip_start_time
                print(f"   ❌ 예외 발생: {str(e)} ({clip_elapsed:.1f}초)")
                
                results['failed'] += 1
                results['failed_files'].append(clip_info['path'])
                
                # 에러 시 계속 진행할지 결정
                continue_on_error = self.config['dataset'].get('error_handling', {}).get('continue_on_error', True)
                if not continue_on_error:
                    print(f"\n❌ 처리 중 오류 발생으로 배치 처리를 중단합니다.")
                    results['success'] = False
                    break
        
        # 6. 최종 결과 출력
        batch_elapsed = time.time() - batch_start_time
        self.print_batch_results(results, batch_elapsed)
        
        return results
    
    def print_batch_results(self, results: Dict, elapsed_time: float):
        """배치 처리 결과 출력"""
        print(f"\n{'='*60}")
        print(f"🎯 배치 처리 완료!")
        print(f"{'='*60}")
        
        print(f"⏱️ 총 소요 시간: {elapsed_time//60:.0f}분 {elapsed_time%60:.0f}초")
        print(f"📊 처리 결과:")
        print(f"   ✅ 성공: {results['processed']}개")
        print(f"   ❌ 실패: {results['failed']}개")
        print(f"   📁 전체: {results['total']}개")
        
        if results['failed_files']:
            print(f"\n❌ 실패한 파일들:")
            for failed_file in results['failed_files']:
                print(f"   - {os.path.basename(failed_file)}")
        
        print(f"{'='*60}")