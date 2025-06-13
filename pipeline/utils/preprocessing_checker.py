#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/utils/preprocessing_checker.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils

# 텐션 계산기 import
from tension_analyzer.tension_calculator import MultiEmotionTensionCalculator


class PreprocessingChecker:
    """
    전처리 파일 상태 확인 및 누락된 단계 자동 실행
    - HDF5 파일 존재 확인 (video, audio)
    - 텐션 JSON 파일 존재 확인
    - 누락된 파일 자동 생성
    """
    
    def __init__(self, config_path: str = None):
        """
        전처리 체커 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 프로젝트 루트로 작업 디렉토리 변경
        os.chdir(project_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Config 로드 (기본값 사용)
        if config_path is None:
            config_path = "pipeline/configs/funclip_extraction_config.yaml"
        
        self.config = PipelineUtils.load_config(config_path)
        
        # 출력 디렉토리 설정
        self.output_dirs = {
            'preprocessed': os.path.join('outputs', 'preprocessed_data'),
            'video_sequences': os.path.join('outputs', 'preprocessed_data', 'video_sequences'),
            'audio_sequences': os.path.join('outputs', 'preprocessed_data', 'audio_sequences'),
            'tension_data': os.path.join('outputs', 'tension_data')
        }
        
        self.logger.info("✅ 전처리 체커 초기화 완료")
    
    def extract_video_name(self, video_path: str) -> str:
        """
        비디오 파일 경로에서 이름 추출
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            str: 추출된 비디오 이름
        """
        return Path(video_path).stem
    
    def find_hdf5_files(self, video_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        비디오 이름으로 HDF5 파일들 찾기
        
        Args:
            video_name (str): 비디오 이름
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (video_h5_path, audio_h5_path)
        """
        video_h5_path = None
        audio_h5_path = None
        
        # 비디오 HDF5 파일 찾기
        video_pattern = os.path.join(self.output_dirs['video_sequences'], f"video_seq_{video_name}_*.h5")
        video_matches = glob.glob(video_pattern)
        if video_matches:
            video_h5_path = max(video_matches, key=os.path.getmtime)  # 최신 파일
        
        # 오디오 HDF5 파일 찾기
        audio_pattern = os.path.join(self.output_dirs['audio_sequences'], f"audio_seq_{video_name}_*.h5")
        audio_matches = glob.glob(audio_pattern)
        if audio_matches:
            audio_h5_path = max(audio_matches, key=os.path.getmtime)  # 최신 파일
        
        return video_h5_path, audio_h5_path
    
    def find_tension_json(self, video_name: str) -> Optional[str]:
        """
        비디오 이름으로 텐션 JSON 파일 찾기
        
        Args:
            video_name (str): 비디오 이름
            
        Returns:
            Optional[str]: 텐션 JSON 파일 경로 (없으면 None)
        """
        tension_pattern = os.path.join(self.output_dirs['tension_data'], f"tension_{video_name}_*.json")
        tension_matches = glob.glob(tension_pattern)
        
        if tension_matches:
            return max(tension_matches, key=os.path.getmtime)  # 최신 파일
        
        return None
    
    def check_files_status(self, video_path: str) -> Dict:
        """
        비디오 파일의 전처리 상태 상세 확인
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            Dict: {
                'video_name': str,
                'video_path': str,
                'video_h5': str|None,
                'audio_h5': str|None, 
                'tension_json': str|None,
                'missing_steps': List[str],
                'status': str,  # 'complete', 'partial', 'none'
                'can_extract_clips': bool
            }
        """
        video_name = self.extract_video_name(video_path)
        
        # 파일들 찾기
        video_h5_path, audio_h5_path = self.find_hdf5_files(video_name)
        tension_json_path = self.find_tension_json(video_name)
        
        # 누락된 단계 확인
        missing_steps = []
        
        if not video_h5_path or not os.path.exists(video_h5_path):
            missing_steps.extend(['video_preprocessing', 'audio_preprocessing', 'tension_calculation'])
        elif not audio_h5_path or not os.path.exists(audio_h5_path):
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
        
        # 클립 추출 가능 여부 (최소 video_h5 + audio_h5 필요)
        can_extract_clips = (video_h5_path and audio_h5_path and 
                           os.path.exists(video_h5_path) and os.path.exists(audio_h5_path))
        
        result = {
            'video_name': video_name,
            'video_path': video_path,
            'video_h5': video_h5_path,
            'audio_h5': audio_h5_path,
            'tension_json': tension_json_path,
            'missing_steps': missing_steps,
            'status': status,
            'can_extract_clips': can_extract_clips
        }
        
        # 로그 출력
        self.logger.info(f"📊 전처리 상태 확인: {video_name}")
        self.logger.info(f"   상태: {status}")
        if missing_steps:
            self.logger.info(f"   누락 단계: {', '.join(missing_steps)}")
        if video_h5_path:
            self.logger.info(f"   비디오 HDF5: {os.path.basename(video_h5_path)}")
        if audio_h5_path:
            self.logger.info(f"   오디오 HDF5: {os.path.basename(audio_h5_path)}")
        if tension_json_path:
            self.logger.info(f"   텐션 JSON: {os.path.basename(tension_json_path)}")
        
        return result
    
    def generate_missing_tension(self, video_name: str, tension_config_path: str = None) -> bool:
        """
        누락된 텐션 JSON 파일 생성
        
        Args:
            video_name (str): 비디오 이름
            tension_config_path (str, optional): 텐션 계산 설정 파일 경로
            
        Returns:
            bool: 생성 성공 여부
        """
        self.logger.info(f"⚡ 텐션 계산 시작: {video_name}")
        
        try:
            # 텐션 계산기 초기화
            if tension_config_path is None:
                tension_config_path = "pipeline/configs/integrated_config.yaml"
            
            calculator = MultiEmotionTensionCalculator(tension_config_path)
            
            # 텐션 계산 실행
            result = calculator.calculate_tension(video_name)
            
            if result:
                self.logger.info(f"✅ 텐션 계산 완료: {video_name}")
                
                # 생성된 파일 경로 찾기
                tension_json_path = self.find_tension_json(video_name)
                if tension_json_path:
                    self.logger.info(f"   저장 위치: {os.path.relpath(tension_json_path)}")
                
                return True
            else:
                self.logger.error(f"❌ 텐션 계산 실패: {video_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 텐션 계산 중 오류: {video_name} - {e}")
            return False
    
    def ensure_preprocessing_complete(self, video_path: str, auto_fix: bool = True) -> Dict:
        """
        전처리 완료 보장 (누락된 단계 자동 실행)
        
        Args:
            video_path (str): 비디오 파일 경로
            auto_fix (bool): 자동 수정 여부
            
        Returns:
            Dict: 처리 결과 정보
        """
        # 1. 현재 상태 확인
        status = self.check_files_status(video_path)
        
        # 2. 완료 상태면 바로 반환
        if status['status'] == 'complete':
            self.logger.info(f"✅ 전처리 완료됨: {status['video_name']}")
            return status
        
        # 3. 자동 수정 비활성화면 상태만 반환
        if not auto_fix:
            return status
        
        # 4. 부분 처리 상태 (텐션 JSON만 없음)
        if status['status'] == 'partial' and 'tension_calculation' in status['missing_steps']:
            self.logger.info(f"🔄 텐션 계산 누락 감지, 자동 생성 시작...")
            
            if self.generate_missing_tension(status['video_name']):
                # 상태 다시 확인
                updated_status = self.check_files_status(video_path)
                return updated_status
            else:
                status['auto_fix_failed'] = True
                return status
        
        # 5. 전체 전처리 필요한 경우
        elif status['status'] == 'none' or 'video_preprocessing' in status['missing_steps']:
            self.logger.warning(f"⚠️ 전체 전처리 필요: {status['video_name']}")
            status['requires_full_preprocessing'] = True
            return status
        
        return status
    
    def print_status_summary(self, status: Dict) -> None:
        """
        상태 요약 정보 출력
        
        Args:
            status (Dict): check_files_status 또는 ensure_preprocessing_complete 결과
        """
        print(f"\n📊 전처리 상태: {status['video_name']}")
        print(f"{'='*50}")
        
        # 상태별 메시지
        if status['status'] == 'complete':
            print(f"✅ 모든 전처리 완료")
        elif status['status'] == 'partial':
            print(f"🔄 부분 처리됨 (누락: {', '.join(status['missing_steps'])})")
        else:
            print(f"⏳ 전처리 필요 (누락: {', '.join(status['missing_steps'])})")
        
        # 파일 상태
        print(f"\n📁 파일 상태:")
        print(f"   비디오 HDF5: {'✅' if status['video_h5'] else '❌'}")
        print(f"   오디오 HDF5: {'✅' if status['audio_h5'] else '❌'}")
        print(f"   텐션 JSON: {'✅' if status['tension_json'] else '❌'}")
        
        # 클립 추출 가능 여부
        print(f"\n🎬 클립 추출 가능: {'✅' if status['can_extract_clips'] else '❌'}")
        
        # 추가 정보
        if status.get('requires_full_preprocessing'):
            print(f"\n💡 해결 방법:")
            print(f"   전체 전처리가 필요합니다.")
            print(f"   다음 명령을 먼저 실행하세요:")
            print(f"   python pipeline/integrated_pipeline.py \"{status['video_path']}\"")
        
        elif status.get('auto_fix_failed'):
            print(f"\n⚠️ 자동 수정 실패:")
            print(f"   텐션 계산을 수동으로 다시 시도해보세요.")
        
        print(f"{'='*50}")
    
    def get_required_files(self, video_path: str) -> Dict:
        """
        클립 추출에 필요한 파일들 반환 (존재하는 것만)
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            Dict: 필요한 파일 경로들
        """
        status = self.check_files_status(video_path)
        
        required_files = {
            'video_path': video_path,
            'video_name': status['video_name']
        }
        
        # 존재하는 파일만 추가
        if status['video_h5'] and os.path.exists(status['video_h5']):
            required_files['video_h5'] = status['video_h5']
        
        if status['audio_h5'] and os.path.exists(status['audio_h5']):
            required_files['audio_h5'] = status['audio_h5']
        
        if status['tension_json'] and os.path.exists(status['tension_json']):
            required_files['tension_json'] = status['tension_json']
        
        return required_files


def main():
    """테스트 실행"""
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='전처리 파일 상태 확인 및 자동 수정')
    parser.add_argument('video_path', help='비디오 파일 경로')
    parser.add_argument('--check-only', action='store_true', help='상태 확인만 (자동 수정 안함)')
    parser.add_argument('--config', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 전처리 체커 실행
        checker = PreprocessingChecker(config_path=args.config)
        
        if args.check_only:
            # 상태 확인만
            status = checker.check_files_status(args.video_path)
        else:
            # 자동 수정 포함
            status = checker.ensure_preprocessing_complete(args.video_path)
        
        # 결과 출력
        checker.print_status_summary(status)
        
        # 클립 추출 가능 여부에 따른 종료 코드
        if status['can_extract_clips']:
            print(f"\n🎬 클립 추출 준비 완료!")
            return 0
        else:
            print(f"\n⚠️ 클립 추출 불가능")
            return 1
        
    except Exception as e:
        print(f"❌ 전처리 체크 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())