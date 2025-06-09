#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import h5py
import numpy as np
import librosa
import webrtcvad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class LongVideoAudioPreprocessor:
    """
    긴 영상 오디오 전처리기 (VAD 통합 버전)
    0.05초 단위로 RMS + VAD 추출하여 원시 시퀀스 저장
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
        self.analysis_interval = self.config['audio']['analysis_interval']  # 0.05초
        
        # VAD 초기화
        self._init_vad()
        
        self.logger.info("✅ 긴 영상 오디오 전처리기 초기화 완료 (VAD 통합)")
        self.logger.info(f"   샘플레이트: {self.sample_rate}Hz")
        self.logger.info(f"   분석 간격: {self.analysis_interval}초")
        self.logger.info(f"   VAD 민감도: {self.vad_sensitivity}")
    
    def _init_vad(self):
        """WebRTC VAD 초기화"""
        try:
            self.vad_sensitivity = self.config['audio']['vad_sensitivity']
            self.vad_frame_duration = self.config['audio']['vad_frame_duration']  # 10ms
            self.vad_aggregation_frames = self.config['audio']['vad_aggregation_frames']  # 5개
            self.vad_majority_threshold = self.config['audio']['vad_majority_threshold']  # 3개
            
            # WebRTC VAD 객체 생성
            self.vad = webrtcvad.Vad(self.vad_sensitivity)
            
            # VAD 프레임 크기 계산 (16kHz 기준)
            self.vad_frame_size = int(self.sample_rate * self.vad_frame_duration / 1000)  # 160 samples
            
            self.logger.info(f"✅ WebRTC VAD 초기화 완료")
            self.logger.info(f"   프레임 길이: {self.vad_frame_duration}ms ({self.vad_frame_size} samples)")
            self.logger.info(f"   집계 프레임: {self.vad_aggregation_frames}개 → {self.analysis_interval}초")
            
        except Exception as e:
            self.logger.error(f"❌ VAD 초기화 실패: {e}")
            raise
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        default_config = {
            'audio': {
                'sample_rate': 16000,
                'analysis_interval': 0.05,
                'vad_sensitivity': 2,
                'vad_frame_duration': 10,
                'vad_aggregation_frames': 5,
                'vad_majority_threshold': 3
            },
            'output': {
                'base_dir': 'video_analyzer',
                'preprocessed_dir': 'preprocessed_data',
                'audio_sequence_dir': 'audio_sequences'
            },
            'logging': {
                'level': 'INFO',
                'save_audio_statistics': True
            },
            'debug': {
                'save_vad_debug_info': False
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
        긴 영상에서 오디오 전처리 (VAD 통합)
        
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
            
            # 오디오 로드 (16kHz로 직접 로드)
            y, sr = librosa.load(video_path, sr=self.sample_rate)
            duration = len(y) / sr
            
            self.logger.info(f"   영상 길이: {duration:.1f}초 ({duration/60:.1f}분)")
            self.logger.info(f"   로드된 샘플레이트: {sr}Hz")
            
            # 0.05초 간격으로 RMS 시퀀스 추출
            rms_sequence = self._extract_rms_sequence(y, sr)
            
            # VAD 시퀀스 추출 (NEW)
            vad_sequence = self._extract_vad_sequence(y, sr)
            
            # 시퀀스 길이 맞추기 (RMS와 VAD 길이가 다를 수 있음)
            min_length = min(len(rms_sequence), len(vad_sequence))
            rms_sequence = rms_sequence[:min_length]
            vad_sequence = vad_sequence[:min_length]
            
            # 타임스탬프 생성
            timestamps = np.arange(min_length) * self.analysis_interval
            
            # VAD 통계 계산
            vad_stats = self._calculate_vad_statistics(vad_sequence)
            
            # 결과 구성
            result = {
                'video_name': video_name,
                'video_path': video_path,
                'duration': duration,
                'sample_rate': sr,
                'analysis_interval': self.analysis_interval,
                'sequences': {
                    'rms_values': rms_sequence,
                    'vad_labels': vad_sequence,  # change_rates 대신 VAD
                    'timestamps': timestamps
                },
                'metadata': {
                    'total_frames': min_length,
                    'frames_per_second': 1.0 / self.analysis_interval,  # 20 FPS
                    'processed_at': datetime.now().isoformat(),
                    'vad_statistics': vad_stats
                }
            }
            
            # HDF5로 저장
            hdf5_path = self._save_to_hdf5(result)
            result['hdf5_path'] = hdf5_path
            
            self.logger.info(f"   RMS 시퀀스: {len(rms_sequence)}개 프레임")
            self.logger.info(f"   VAD 시퀀스: {len(vad_sequence)}개 프레임")
            self.logger.info(f"   발화 비율: {vad_stats['voice_activity_ratio']:.1%}")
            self.logger.info(f"   저장 위치: {hdf5_path}")
            
            # 오디오 통계 파일 저장
            if self.config['logging']['save_audio_statistics']:
                self._save_audio_statistics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 오디오 전처리 실패: {video_path} - {e}")
            return None
    
    def _extract_rms_sequence(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        0.05초 간격으로 RMS 값 시퀀스 추출
        
        Args:
            y (np.ndarray): 오디오 신호
            sr (int): 샘플 레이트
            
        Returns:
            np.ndarray: RMS 값 시퀀스
        """
        # 0.05초당 샘플 수
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
    
    def _extract_vad_sequence(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        VAD 시퀀스 추출 (WebRTC VAD 사용)
        
        Args:
            y (np.ndarray): 오디오 신호 (16kHz)
            sr (int): 샘플 레이트 (16000이어야 함)
            
        Returns:
            np.ndarray: VAD 시퀀스 (0/1 배열)
        """
        if sr != 16000:
            raise ValueError(f"VAD는 16kHz만 지원합니다. 현재: {sr}Hz")
        
        try:
            # 10ms 단위로 VAD 처리
            vad_results = []
            
            for i in range(0, len(y) - self.vad_frame_size, self.vad_frame_size):
                frame = y[i:i + self.vad_frame_size]
                
                # int16으로 변환 (WebRTC 요구사항)
                frame_int16 = (frame * 32767).astype(np.int16)
                
                # VAD 판정
                try:
                    is_speech = self.vad.is_speech(frame_int16.tobytes(), sr)
                    vad_results.append(1 if is_speech else 0)
                except Exception as e:
                    # VAD 실패 시 0으로 처리
                    vad_results.append(0)
            
            # 0.05초 단위로 집계 (5개 10ms 프레임 → 다수결)
            aggregated_vad = []
            for i in range(0, len(vad_results), self.vad_aggregation_frames):
                window = vad_results[i:i + self.vad_aggregation_frames]
                
                # 설정된 임계값 이상이면 발화로 판정
                speech_count = sum(window)
                is_speech = speech_count >= self.vad_majority_threshold
                aggregated_vad.append(1 if is_speech else 0)
            
            return np.array(aggregated_vad)
            
        except Exception as e:
            self.logger.error(f"❌ VAD 처리 실패: {e}")
            # 실패 시 모든 프레임을 침묵으로 처리
            num_frames = len(y) // int(sr * self.analysis_interval)
            return np.zeros(num_frames, dtype=int)
    
    def _calculate_vad_statistics(self, vad_sequence: np.ndarray) -> Dict:
        """VAD 통계 계산"""
        if len(vad_sequence) == 0:
            return {
                'voice_activity_ratio': 0.0,
                'silence_ratio': 1.0,
                'speech_burst_count': 0,
                'silence_burst_count': 0,
                'max_speech_duration': 0.0,
                'max_silence_duration': 0.0,
                'avg_speech_duration': 0.0,
                'avg_silence_duration': 0.0
            }
        
        # 기본 비율
        voice_ratio = float(np.mean(vad_sequence))
        silence_ratio = 1.0 - voice_ratio
        
        # 연속 구간 분석
        speech_bursts = []
        silence_bursts = []
        
        current_state = vad_sequence[0]
        current_duration = 1
        
        for i in range(1, len(vad_sequence)):
            if vad_sequence[i] == current_state:
                current_duration += 1
            else:
                # 상태 변화
                duration_seconds = current_duration * self.analysis_interval
                
                if current_state == 1:
                    speech_bursts.append(duration_seconds)
                else:
                    silence_bursts.append(duration_seconds)
                
                current_state = vad_sequence[i]
                current_duration = 1
        
        # 마지막 구간 처리
        duration_seconds = current_duration * self.analysis_interval
        if current_state == 1:
            speech_bursts.append(duration_seconds)
        else:
            silence_bursts.append(duration_seconds)
        
        return {
            'voice_activity_ratio': voice_ratio,
            'silence_ratio': silence_ratio,
            'speech_burst_count': len(speech_bursts),
            'silence_burst_count': len(silence_bursts),
            'max_speech_duration': float(max(speech_bursts)) if speech_bursts else 0.0,
            'max_silence_duration': float(max(silence_bursts)) if silence_bursts else 0.0,
            'avg_speech_duration': float(np.mean(speech_bursts)) if speech_bursts else 0.0,
            'avg_silence_duration': float(np.mean(silence_bursts)) if silence_bursts else 0.0
        }
    
    def _save_to_hdf5(self, result: Dict) -> str:
        """
        전처리 결과를 HDF5 파일로 저장 (VAD 포함)
        
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
                
                # VAD 설정 메타데이터 추가
                f.attrs['vad_sensitivity'] = self.vad_sensitivity
                f.attrs['vad_frame_duration'] = self.vad_frame_duration
                f.attrs['vad_aggregation_frames'] = self.vad_aggregation_frames
                f.attrs['vad_majority_threshold'] = self.vad_majority_threshold
                
                # VAD 통계
                vad_stats = result['metadata']['vad_statistics']
                for key, value in vad_stats.items():
                    f.attrs[f'vad_{key}'] = value
                
                # 시퀀스 데이터 (압축 없음)
                sequences_group = f.create_group('sequences')
                sequences_group.create_dataset('rms_values', 
                                             data=result['sequences']['rms_values'])
                sequences_group.create_dataset('vad_labels', 
                                             data=result['sequences']['vad_labels'])
                sequences_group.create_dataset('timestamps', 
                                             data=result['sequences']['timestamps'])
            
            return hdf5_path
            
        except Exception as e:
            self.logger.error(f"❌ HDF5 저장 실패: {e}")
            raise
    
    def _save_audio_statistics(self, result: Dict):
        """오디오 통계를 텍스트 파일로 저장"""
        try:
            video_name = result['video_name']
            stats_filename = f"audio_stats_{video_name}.txt"
            stats_path = os.path.join(self.audio_sequence_dir, stats_filename)
            
            vad_stats = result['metadata']['vad_statistics']
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write(f"오디오 처리 통계 보고서\n")
                f.write(f"{'='*50}\n")
                f.write(f"영상: {video_name}\n")
                f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"📊 기본 정보\n")
                f.write(f"├─ 길이: {result['duration']:.1f}초 ({result['duration']/60:.1f}분)\n")
                f.write(f"├─ 샘플레이트: {result['sample_rate']}Hz\n")
                f.write(f"├─ 분석 간격: {result['analysis_interval']}초\n")
                f.write(f"└─ 총 프레임: {result['metadata']['total_frames']}개\n\n")
                
                f.write(f"🎙️ VAD 통계\n")
                f.write(f"├─ 발화 비율: {vad_stats['voice_activity_ratio']:.1%}\n")
                f.write(f"├─ 침묵 비율: {vad_stats['silence_ratio']:.1%}\n")
                f.write(f"├─ 발화 구간 수: {vad_stats['speech_burst_count']}개\n")
                f.write(f"├─ 침묵 구간 수: {vad_stats['silence_burst_count']}개\n")
                f.write(f"├─ 최대 연속 발화: {vad_stats['max_speech_duration']:.1f}초\n")
                f.write(f"├─ 최대 연속 침묵: {vad_stats['max_silence_duration']:.1f}초\n")
                f.write(f"├─ 평균 발화 길이: {vad_stats['avg_speech_duration']:.1f}초\n")
                f.write(f"└─ 평균 침묵 길이: {vad_stats['avg_silence_duration']:.1f}초\n\n")
                
                f.write(f"⚙️ VAD 설정\n")
                f.write(f"├─ 민감도: {self.vad_sensitivity}\n")
                f.write(f"├─ 프레임 길이: {self.vad_frame_duration}ms\n")
                f.write(f"├─ 집계 프레임: {self.vad_aggregation_frames}개\n")
                f.write(f"└─ 다수결 임계값: {self.vad_majority_threshold}개\n")
            
            self.logger.info(f"📄 오디오 통계 파일 저장: {stats_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 통계 파일 저장 실패: {e}")
    
    def load_from_hdf5(self, hdf5_path: str) -> Optional[Dict]:
        """
        HDF5 파일에서 오디오 시퀀스 로드 (VAD 포함)
        
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
                        'vad_labels': f['sequences/vad_labels'][:],
                        'timestamps': f['sequences/timestamps'][:]
                    },
                    'metadata': {
                        'total_frames': f.attrs['total_frames'],
                        'frames_per_second': f.attrs['frames_per_second'],
                        'processed_at': f.attrs['processed_at'],
                        'vad_statistics': {}
                    },
                    'hdf5_path': hdf5_path
                }
                
                # VAD 통계 로드
                for key in f.attrs.keys():
                    if key.startswith('vad_'):
                        stat_key = key[4:]  # 'vad_' 제거
                        result['metadata']['vad_statistics'][stat_key] = f.attrs[key]
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ HDF5 로드 실패: {hdf5_path} - {e}")
            return None
    
    def get_audio_statistics(self, hdf5_path: str) -> Optional[Dict]:
        """
        오디오 시퀀스 통계 정보 계산 (VAD 포함)
        
        Args:
            hdf5_path (str): HDF5 파일 경로
            
        Returns:
            Dict: 통계 정보
        """
        data = self.load_from_hdf5(hdf5_path)
        if data is None:
            return None
        
        rms_values = data['sequences']['rms_values']
        vad_labels = data['sequences']['vad_labels']
        
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
            'vad_statistics': data['metadata']['vad_statistics'],
            'activity_analysis': {
                'high_rms_ratio': float(np.sum(rms_values > np.percentile(rms_values, 75)) / len(rms_values)),
                'low_rms_ratio': float(np.sum(rms_values < np.percentile(rms_values, 25)) / len(rms_values)),
                'rms_silence_ratio': float(np.sum(rms_values < 0.01) / len(rms_values)),
                'vad_voice_ratio': float(np.mean(vad_labels)),
                'vad_silence_ratio': float(1 - np.mean(vad_labels))
            }
        }
        
        return stats


def main():
    """테스트 실행"""
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='긴 영상 오디오 전처리 (VAD 통합)')
    parser.add_argument('video_path', help='처리할 비디오 파일 경로')
    parser.add_argument('--config', default='video_analyzer/configs/inference_config.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # WebRTC VAD 설치 확인
    try:
        import webrtcvad
    except ImportError:
        print("❌ webrtcvad 라이브러리가 설치되지 않았습니다.")
        print("설치 명령: pip install webrtcvad")
        return
    
    # 설정 파일 생성 (없는 경우)
    config_dir = "video_analyzer/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    if not os.path.exists(args.config):
        # VAD 통합 기본 설정
        default_config = {
            'audio': {
                'sample_rate': 16000,
                'analysis_interval': 0.05,
                'vad_sensitivity': 2,
                'vad_frame_duration': 10,
                'vad_aggregation_frames': 5,
                'vad_majority_threshold': 3
            },
            'output': {
                'base_dir': 'video_analyzer',
                'preprocessed_dir': 'preprocessed_data',
                'audio_sequence_dir': 'audio_sequences'
            },
            'logging': {
                'level': 'INFO',
                'save_audio_statistics': True
            },
            'debug': {
                'save_vad_debug_info': False
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
            print(f"발화 비율: {stats['vad_statistics']['voice_activity_ratio']:.1%}")
            print(f"침묵 비율: {stats['vad_statistics']['silence_ratio']:.1%}")
            print(f"발화 구간 수: {stats['vad_statistics']['speech_burst_count']}개")
    else:
        print("❌ 전처리 실패")


if __name__ == "__main__":
    main()