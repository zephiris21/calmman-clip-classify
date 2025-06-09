#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import h5py
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

class LongVideoAudioPreprocessor:
    """
    긴 영상 오디오 전처리기
    0.1초 단위로 RMS + 변화율 추출하여 원시 시퀀스 저장
    슬라이딩 윈도우용 데이터 준비
    """
    
    def __init__(self, config_path: str = "video_analyzer/configs/inference_config.yaml"):
        """
        긴 영상 오디오 전처리기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._create_output_dirs()
        
        # 오디오 처리 파라미터
        self.sample_rate = self.config['audio']['sample_rate']
        self.analysis_interval = self.config['audio']['analysis_interval']  # 0.1초
        
        self.logger.info("✅ 긴 영상 오디오 전처리기 초기화 완료")
        self.logger.info(f"   샘플레이트: {self.sample_rate}Hz")
        self.logger.info(f"   분석 간격: {self.analysis_interval}초")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        default_config = {
            'audio': {
                'sample_rate': 22050,
                'analysis_interval': 0.1
            },
            'output': {
                'base_dir': 'video_analyzer',
                'preprocessed_dir': 'preprocessed_data',
                'audio_sequence_dir': 'audio_sequences'
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # 기본값과 병합
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
        else:
            config = default_config
            print(f"⚠️ 설정 파일 없음, 기본값 사용: {config_path}")
        
        return config
    
    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('LongVideoAudioPreprocessor')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        base_dir = self.config['output']['base_dir']
        self.preprocessed_dir = os.path.join(base_dir, self.config['output']['preprocessed_dir'])
        self.audio_sequence_dir = os.path.join(self.preprocessed_dir, self.config['output']['audio_sequence_dir'])
        
        os.makedirs(self.audio_sequence_dir, exist_ok=True)
    
    def preprocess_long_video_audio(self, video_path: str) -> Optional[Dict]:
        """
        긴 영상에서 오디오 전처리
        
        Args:
            video_path (str): 비디오 파일 경로
            
        Returns:
            Dict: 전처리된 오디오 시퀀스 정보
        """
        if not os.path.exists(video_path):
            self.logger.error(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
            return None
        
        try:
            video_name = Path(video_path).stem
            self.logger.info(f"🎬 긴 영상 오디오 전처리 시작: {video_name}")
            
            # 오디오 로드
            y, sr = librosa.load(video_path, sr=self.sample_rate)
            duration = len(y) / sr
            
            self.logger.info(f"   영상 길이: {duration:.1f}초 ({duration/60:.1f}분)")
            
            # 0.1초 간격으로 RMS 시퀀스 추출
            rms_sequence = self._extract_rms_sequence(y, sr)
            
            # 변화율 시퀀스 계산
            change_rate_sequence = self._calculate_change_rates(rms_sequence)
            
            # 타임스탬프 생성
            timestamps = np.arange(len(rms_sequence)) * self.analysis_interval
            
            # 결과 구성
            result = {
                'video_name': video_name,
                'video_path': video_path,
                'duration': duration,
                'sample_rate': sr,
                'analysis_interval': self.analysis_interval,
                'sequences': {
                    'rms_values': rms_sequence,
                    'change_rates': change_rate_sequence,
                    'timestamps': timestamps
                },
                'metadata': {
                    'total_frames': len(rms_sequence),
                    'frames_per_second': 1.0 / self.analysis_interval,  # 10 FPS
                    'processed_at': datetime.now().isoformat()
                }
            }
            
            # HDF5로 저장
            hdf5_path = self._save_to_hdf5(result)
            result['hdf5_path'] = hdf5_path
            
            self.logger.info(f"   RMS 시퀀스: {len(rms_sequence)}개 프레임")
            self.logger.info(f"   변화율 시퀀스: {len(change_rate_sequence)}개 프레임")
            self.logger.info(f"   저장 위치: {hdf5_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 오디오 전처리 실패: {video_path} - {e}")
            return None
    
    def _extract_rms_sequence(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        0.1초 간격으로 RMS 값 시퀀스 추출
        
        Args:
            y (np.ndarray): 오디오 신호
            sr (int): 샘플 레이트
            
        Returns:
            np.ndarray: RMS 값 시퀀스
        """
        # 0.1초당 샘플 수
        samples_per_interval = int(sr * self.analysis_interval)
        
        # 전체 구간 수
        num_intervals = len(y) // samples_per_interval
        
        rms_values = []
        
        for i in range(num_intervals):
            start_idx = i * samples_per_interval
            end_idx = start_idx + samples_per_interval
            
            # 해당 구간의 RMS 계산
            segment = y[start_idx:end_idx]
            rms = np.sqrt(np.mean(segment ** 2))
            rms_values.append(rms)
        
        return np.array(rms_values)
    
    def _calculate_change_rates(self, rms_values: np.ndarray) -> np.ndarray:
        """
        RMS 변화율 시퀀스 계산
        
        Args:
            rms_values (np.ndarray): RMS 값 시퀀스
            
        Returns:
            np.ndarray: 변화율 시퀀스 (길이 = len(rms_values) - 1)
        """
        if len(rms_values) < 2:
            return np.array([])
        
        # 연속 프레임 간 차이 계산
        diff = np.diff(rms_values)
        
        # 절댓값 변화율 (변화 크기)
        change_rates = np.abs(diff)
        
        return change_rates
    
    def _save_to_hdf5(self, result: Dict) -> str:
        """
        전처리 결과를 HDF5 파일로 저장
        
        Args:
            result (Dict): 전처리 결과
            
        Returns:
            str: 저장된 HDF5 파일 경로
        """
        try:
            # 안전한 파일명 생성
            safe_name = result['video_name'].replace('*', '_').replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hdf5_filename = f"audio_seq_{safe_name}_{timestamp}.h5"
            hdf5_path = os.path.join(self.audio_sequence_dir, hdf5_filename)
            
            with h5py.File(hdf5_path, 'w') as f:
                # 메타데이터
                f.attrs['video_name'] = result['video_name']
                f.attrs['video_path'] = result['video_path']
                f.attrs['duration'] = result['duration']
                f.attrs['sample_rate'] = result['sample_rate']
                f.attrs['analysis_interval'] = result['analysis_interval']
                f.attrs['total_frames'] = result['metadata']['total_frames']
                f.attrs['frames_per_second'] = result['metadata']['frames_per_second']
                f.attrs['processed_at'] = result['metadata']['processed_at']
                
                # 시퀀스 데이터
                sequences_group = f.create_group('sequences')
                sequences_group.create_dataset('rms_values', 
                                             data=result['sequences']['rms_values'],
                                             compression='gzip')
                sequences_group.create_dataset('change_rates', 
                                             data=result['sequences']['change_rates'],
                                             compression='gzip')
                sequences_group.create_dataset('timestamps', 
                                             data=result['sequences']['timestamps'],
                                             compression='gzip')
            
            return hdf5_path
            
        except Exception as e:
            self.logger.error(f"❌ HDF5 저장 실패: {e}")
            raise
    
    def load_from_hdf5(self, hdf5_path: str) -> Optional[Dict]:
        """
        HDF5 파일에서 오디오 시퀀스 로드
        
        Args:
            hdf5_path (str): HDF5 파일 경로
            
        Returns:
            Dict: 로드된 오디오 시퀀스 데이터
        """
        try:
            with h5py.File(hdf5_path, 'r') as f:
                result = {
                    'video_name': f.attrs['video_name'],
                    'video_path': f.attrs['video_path'],
                    'duration': f.attrs['duration'],
                    'sample_rate': f.attrs['sample_rate'],
                    'analysis_interval': f.attrs['analysis_interval'],
                    'sequences': {
                        'rms_values': f['sequences/rms_values'][:],
                        'change_rates': f['sequences/change_rates'][:],
                        'timestamps': f['sequences/timestamps'][:]
                    },
                    'metadata': {
                        'total_frames': f.attrs['total_frames'],
                        'frames_per_second': f.attrs['frames_per_second'],
                        'processed_at': f.attrs['processed_at']
                    },
                    'hdf5_path': hdf5_path
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ HDF5 로드 실패: {hdf5_path} - {e}")
            return None
    
    def get_audio_statistics(self, hdf5_path: str) -> Optional[Dict]:
        """
        오디오 시퀀스 통계 정보 계산
        
        Args:
            hdf5_path (str): HDF5 파일 경로
            
        Returns:
            Dict: 통계 정보
        """
        data = self.load_from_hdf5(hdf5_path)
        if data is None:
            return None
        
        rms_values = data['sequences']['rms_values']
        change_rates = data['sequences']['change_rates']
        
        stats = {
            'duration_minutes': data['duration'] / 60,
            'total_frames': len(rms_values),
            'rms_statistics': {
                'mean': float(np.mean(rms_values)),
                'std': float(np.std(rms_values)),
                'min': float(np.min(rms_values)),
                'max': float(np.max(rms_values)),
                'median': float(np.median(rms_values))
            },
            'change_rate_statistics': {
                'mean': float(np.mean(change_rates)),
                'std': float(np.std(change_rates)),
                'min': float(np.min(change_rates)),
                'max': float(np.max(change_rates)),
                'median': float(np.median(change_rates))
            },
            'activity_analysis': {
                'high_activity_ratio': float(np.sum(rms_values > np.percentile(rms_values, 75)) / len(rms_values)),
                'low_activity_ratio': float(np.sum(rms_values < np.percentile(rms_values, 25)) / len(rms_values)),
                'silence_ratio': float(np.sum(rms_values < 0.01) / len(rms_values))
            }
        }
        
        return stats


def main():
    """테스트 실행"""
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='긴 영상 오디오 전처리')
    parser.add_argument('video_path', help='처리할 비디오 파일 경로')
    parser.add_argument('--config', default='video_analyzer/configs/inference_config.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 설정 파일 생성 (없는 경우)
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    
    if not os.path.exists(args.config):
        default_config = {
            'audio': {
                'sample_rate': 22050,
                'analysis_interval': 0.1
            },
            'output': {
                'base_dir': 'video_analyzer',
                'preprocessed_dir': 'preprocessed_data',
                'audio_sequence_dir': 'audio_sequences'
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with open(args.config, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 기본 설정 파일 생성: {args.config}")
    
    # 전처리기 실행
    preprocessor = LongVideoAudioPreprocessor(args.config)
    
    # 비디오 처리
    result = preprocessor.preprocess_long_video_audio(args.video_path)
    
    if result:
        print("\n✅ 전처리 완료!")
        print(f"HDF5 파일: {result['hdf5_path']}")
        
        # 통계 정보 출력
        stats = preprocessor.get_audio_statistics(result['hdf5_path'])
        if stats:
            print(f"\n📊 오디오 통계:")
            print(f"길이: {stats['duration_minutes']:.1f}분")
            print(f"RMS 평균: {stats['rms_statistics']['mean']:.4f}")
            print(f"RMS 표준편차: {stats['rms_statistics']['std']:.4f}")
            print(f"높은 활동 비율: {stats['activity_analysis']['high_activity_ratio']:.1%}")
            print(f"조용한 구간 비율: {stats['activity_analysis']['silence_ratio']:.1%}")
    else:
        print("❌ 전처리 실패")


if __name__ == "__main__":
    main()