#!/usr/bin/env python
# -*- coding: utf-8 -*-

print("=== 텐션 분석기 시작 ===")

import os
import sys
print(f"✅ Python 버전: {sys.version}")
print(f"✅ 현재 디렉토리: {os.getcwd()}")

try:
    import yaml
    print("✅ yaml 임포트 완료")
except Exception as e:
    print(f"❌ yaml 임포트 실패: {e}")
    sys.exit(1)

try:
    import h5py
    print("✅ h5py 임포트 완료")
except Exception as e:
    print(f"❌ h5py 임포트 실패: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy 임포트 완료")
except Exception as e:
    print(f"❌ numpy 임포트 실패: {e}")
    sys.exit(1)

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

print("✅ 모든 라이브러리 임포트 완료")

class MultiEmotionTensionCalculator:
    """
    멀티감정 기반 텐션 계산 시스템
    - 중립 제외 7가지 감정 + Arousal*10 조합
    - VAD 기반 Voice RMS 계산
    - 얼굴 없을 때 이전 값 유지 + Decay
    """
    
    def __init__(self, config_path: str = "tension_analyzer/configs/tension_config.yaml"):
        print(f"🔧 텐션 계산기 초기화 시작")
        print(f"   설정 파일 경로: {config_path}")
        
        self.config = self._load_config(config_path)
        print("✅ 설정 로드 완료")
        
        self._setup_logging()
        print("✅ 로깅 설정 완료")
        
        self._create_output_dirs()
        print("✅ 출력 디렉토리 생성 완료")
        
        # HDF5 파일 기본 경로
        self.audio_sequences_dir = "video_analyzer/preprocessed_data/audio_sequences"
        self.video_sequences_dir = "video_analyzer/preprocessed_data/video_sequences"
        print(f"   오디오 디렉토리: {self.audio_sequences_dir}")
        print(f"   비디오 디렉토리: {self.video_sequences_dir}")
        
        # 감정 레이블 (MTCNN 순서)
        self.emotion_labels = [
            'Anger',       # 0
            'Contempt',    # 1 
            'Disgust',     # 2
            'Fear',        # 3
            'Happiness',   # 4
            'Neutral',     # 5 ← 제외할 감정
            'Sadness',     # 6
            'Surprise'     # 7
        ]
        self.neutral_idx = 5  # Neutral 인덱스
        print(f"   중립 감정 인덱스: {self.neutral_idx}")
        
        # 텐션 계산 파라미터
        self.window_duration = self.config['tension']['window_duration']  # 0.5초
        self.emotion_weight = self.config['tension']['emotion_weight']    # 0.7
        self.audio_weight = self.config['tension']['audio_weight']        # 0.3
        self.arousal_multiplier = self.config['tension']['arousal_multiplier']  # 10
        
        # Voice RMS 정규화
        self.voice_rms_max = self.config['audio']['voice_rms_max']  # 0.1
        
        # Decay 파라미터
        self.decay_rate = self.config['decay']['decay_rate']              # 0.95
        self.silence_3sec_decay = self.config['decay']['silence_3sec_decay']  # 0.85
        self.silence_threshold = self.config['decay']['silence_threshold_seconds']  # 1.0
        
        # 편집 탐지 파라미터
        self.highlight_sensitivity = self.config['editing']['highlight_sensitivity']  # 2.0
        self.change_threshold = self.config['editing']['change_threshold']            # 0.2
        self.low_tension_threshold = self.config['editing']['low_tension_threshold']  # 3.0
        
        self.logger.info("✅ 멀티감정 텐션 계산기 초기화 완료")
        self._print_config_summary()
        print("=== 초기화 완료 ===")
    
    def _find_h5_files(self, filename_pattern: str) -> Tuple[str, str]:
        """HDF5 파일 쌍 찾기 (패턴 하나로 오디오+비디오 모두 찾기)"""
        print(f"🔍 파일 패턴으로 검색 중: {filename_pattern}")
        
        audio_file = None
        video_file = None
        
        # 오디오 파일 찾기
        if os.path.exists(self.audio_sequences_dir):
            for file in os.listdir(self.audio_sequences_dir):
                if filename_pattern in file and file.endswith('.h5'):
                    audio_file = os.path.join(self.audio_sequences_dir, file)
                    print(f"✅ 오디오 파일 발견: {file}")
                    break
        
        # 비디오 파일 찾기
        if os.path.exists(self.video_sequences_dir):
            for file in os.listdir(self.video_sequences_dir):
                if filename_pattern in file and file.endswith('.h5'):
                    video_file = os.path.join(self.video_sequences_dir, file)
                    print(f"✅ 비디오 파일 발견: {file}")
                    break
        
        if audio_file is None:
            print(f"❌ 오디오 파일을 찾을 수 없음: {filename_pattern}")
            if os.path.exists(self.audio_sequences_dir):
                files = os.listdir(self.audio_sequences_dir)
                print(f"   오디오 디렉토리 파일들: {files[:3]}...")
            raise FileNotFoundError(f"오디오 HDF5 파일을 찾을 수 없습니다: {filename_pattern}")
        
        if video_file is None:
            print(f"❌ 비디오 파일을 찾을 수 없음: {filename_pattern}")
            if os.path.exists(self.video_sequences_dir):
                files = os.listdir(self.video_sequences_dir)
                print(f"   비디오 디렉토리 파일들: {files[:3]}...")
            raise FileNotFoundError(f"비디오 HDF5 파일을 찾을 수 없습니다: {filename_pattern}")
        
        return audio_file, video_file
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        print(f"📋 설정 파일 로드 시작: {config_path}")
        
        default_config = {
            'tension': {
                'window_duration': 0.5,
                'emotion_weight': 0.7,
                'audio_weight': 0.3,
                'arousal_multiplier': 10
            },
            'audio': {
                'voice_rms_max': 0.1,
                'vad_activity_threshold': 0.2
            },
            'decay': {
                'decay_rate': 0.95,
                'silence_3sec_decay': 0.85,
                'silence_threshold_seconds': 1.0
            },
            'editing': {
                'highlight_sensitivity': 2.0,
                'change_threshold': 0.2,
                'low_tension_threshold': 3.0
            },
            'output': {
                'base_dir': 'tension_analyzer',
                'tension_analysis_dir': 'outputs/tension_data'
            },
            'logging': {
                'level': 'INFO',
                'save_detailed_log': True
            }
        }
        
        if os.path.exists(config_path):
            print(f"✅ 설정 파일 발견, 로드 중...")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 설정 파일 로드 완료")
            
            # 기본값과 병합
            def merge_dict(default, loaded):
                for key, value in default.items():
                    if key not in loaded:
                        loaded[key] = value
                        print(f"   기본값 추가: {key}")
                    elif isinstance(value, dict) and isinstance(loaded[key], dict):
                        merge_dict(value, loaded[key])
                return loaded
            config = merge_dict(default_config, config)
        else:
            print(f"⚠️ 설정 파일 없음, 기본값 사용: {config_path}")
            config = default_config
            
            # 기본 설정 파일 생성
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            print(f"✅ 기본 설정 파일 생성: {config_path}")
        
        return config
    
    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('MultiEmotionTensionCalculator')
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
        self.tension_output_dir = os.path.join(base_dir, self.config['output']['tension_analysis_dir'])
        os.makedirs(self.tension_output_dir, exist_ok=True)
        print(f"   출력 디렉토리: {self.tension_output_dir}")
    
    def _print_config_summary(self):
        """설정 요약 출력"""
        print("📋 멀티감정 텐션 계산 설정:")
        print(f"   윈도우: {self.window_duration}초")
        print(f"   가중치 - 감정: {self.emotion_weight}, 오디오: {self.audio_weight}")
        print(f"   Arousal 배수: {self.arousal_multiplier}")
        print(f"   Decay - 일반: {self.decay_rate}, 3초침묵: {self.silence_3sec_decay}")
        print(f"   Voice RMS 최대: {self.voice_rms_max}")
    
    def calculate_tension(self, filename_pattern: str, youtube_url: str = None) -> Optional[Dict]:
        """
        멀티감정 기반 텐션 계산
        
        Args:
            filename_pattern (str): 파일명 패턴 (오디오+비디오 모두 매칭)
            youtube_url (str): 유튜브 URL (옵션)
            
        Returns:
            Dict: 텐션 분석 결과 (JSON 구조)
        """
        try:
            print(f"🎬 멀티감정 텐션 분석 시작")
            print(f"   파일명 패턴: {filename_pattern}")
            
            # HDF5 파일 쌍 찾기
            audio_h5_path, video_h5_path = self._find_h5_files(filename_pattern)
            print(f"✅ 파일 쌍 확인 완료")
            
            # 1. 데이터 로드 및 동기화
            print("📊 데이터 로드 및 동기화 시작...")
            data = self._load_and_sync_data(audio_h5_path, video_h5_path)
            if data is None:
                print("❌ 데이터 로드 실패")
                return None
            print("✅ 데이터 동기화 완료")
            
            # 2. 윈도우별 텐션 계산
            print("⚡ 윈도우별 텐션 계산 시작...")
            tension_results = self._calculate_windowed_tension(data)
            print("✅ 텐션 계산 완료")
            
            # 3. 편집 포인트 탐지
            print("✂️ 편집 포인트 탐지 시작...")
            edit_suggestions = self._detect_edit_opportunities(tension_results, data)
            print("✅ 편집 포인트 탐지 완료")
            
            # 4. JSON 결과 생성
            print("📝 JSON 결과 생성 중...")
            result = self._generate_json_result(
                data, tension_results, edit_suggestions, 
                audio_h5_path, video_h5_path, youtube_url
            )
            print("✅ JSON 결과 생성 완료")
            
            # 5. 결과 저장
            print("💾 결과 저장 중...")
            self._save_tension_analysis(result, filename_pattern)
            print("✅ 결과 저장 완료")
            
            return result
            
        except Exception as e:
            print(f"❌ 텐션 계산 실패: {e}")
            import traceback
            print("📍 상세 오류:")
            traceback.print_exc()
            return None
    
    def _load_and_sync_data(self, audio_h5_path: str, video_h5_path: str) -> Optional[Dict]:
        """데이터 로드 및 시간 동기화"""
        try:
            print(f"📂 오디오 데이터 로드: {os.path.basename(audio_h5_path)}")
            # 오디오 데이터 로드
            with h5py.File(audio_h5_path, 'r') as f:
                print(f"   오디오 HDF5 키들: {list(f.keys())}")
                audio_data = {
                    'rms_values': f['sequences/rms_values'][:],
                    'vad_labels': f['sequences/vad_labels'][:],
                    'audio_timestamps': f['sequences/timestamps'][:],
                    'audio_interval': f.attrs['analysis_interval']  # 0.05초
                }
                print(f"   RMS 데이터: {len(audio_data['rms_values'])}개")
                print(f"   VAD 데이터: {len(audio_data['vad_labels'])}개")
            
            print(f"📂 비디오 데이터 로드: {os.path.basename(video_h5_path)}")
            # 비디오 데이터 로드
            with h5py.File(video_h5_path, 'r') as f:
                print(f"   비디오 HDF5 키들: {list(f.keys())}")
                video_data = {
                    'emotions': f['sequences/emotions'][:],      # [N, 10] - 8감정 + VA
                    'face_detected': f['sequences/face_detected'][:],
                    'video_timestamps': f['sequences/timestamps'][:],
                    'video_name': f.attrs['video_name']
                }
                print(f"   감정 데이터: {video_data['emotions'].shape}")
                print(f"   얼굴 탐지: {len(video_data['face_detected'])}개")
            
            # 시간 해상도 확인
            print(f"⏱️ 시간 해상도:")
            print(f"   오디오: {len(audio_data['rms_values'])}프레임 ({audio_data['audio_interval']:.3f}초 간격)")
            print(f"   비디오: {len(video_data['emotions'])}프레임")
            
            # 데이터 동기화 (오디오 0.05초 기준)
            print("🔄 데이터 동기화 중...")
            synced_data = self._synchronize_data(audio_data, video_data)
            print("✅ 데이터 동기화 완료")
            
            return synced_data
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _synchronize_data(self, audio_data: Dict, video_data: Dict) -> Dict:
        """오디오(0.05초)와 비디오(~0.25초) 데이터 동기화"""
        print("🔄 시간 동기화 처리 중...")
        
        audio_timestamps = audio_data['audio_timestamps']
        video_timestamps = video_data['video_timestamps']
        
        print(f"   오디오 시간 범위: {audio_timestamps[0]:.2f} ~ {audio_timestamps[-1]:.2f}초")
        print(f"   비디오 시간 범위: {video_timestamps[0]:.2f} ~ {video_timestamps[-1]:.2f}초")
        
        # 비디오 데이터를 오디오 타임스탬프에 맞춰 보간
        synced_emotions = []
        synced_face_detected = []
        
        for i, audio_ts in enumerate(audio_timestamps):
            # 가장 가까운 비디오 프레임 찾기
            video_idx = np.argmin(np.abs(video_timestamps - audio_ts))
            
            # 감정 데이터 (10차원: 8감정 + Valence + Arousal)
            emotion_frame = video_data['emotions'][video_idx]
            if np.isnan(emotion_frame).any():  # NaN 처리
                emotion_frame = np.zeros(10)
            synced_emotions.append(emotion_frame)
            
            # 얼굴 탐지 데이터
            synced_face_detected.append(video_data['face_detected'][video_idx])
            
            if i % 1000 == 0:  # 진행 상황 출력
                print(f"   동기화 진행: {i}/{len(audio_timestamps)} ({i/len(audio_timestamps)*100:.1f}%)")
        
        result = {
            'timestamps': audio_timestamps,
            'rms_values': audio_data['rms_values'],
            'vad_labels': audio_data['vad_labels'],
            'emotions': np.array(synced_emotions),
            'face_detected': np.array(synced_face_detected),
            'interval': audio_data['audio_interval'],  # 0.05초
            'video_name': video_data['video_name']
        }
        
        print(f"✅ 동기화 완료: {len(result['timestamps'])}개 프레임")
        return result
    
    def _calculate_windowed_tension(self, data: Dict) -> Dict:
        """윈도우별 텐션 계산 (멀티감정 기반)"""
        print("⚡ 윈도우별 텐션 계산 시작...")
        
        timestamps = data['timestamps']
        interval = data['interval']
        
        # 윈도우 설정
        window_frames = int(self.window_duration / interval)  # 0.5초 / 0.05초 = 10프레임
        step_frames = window_frames // 2  # 50% 겹침 (0.25초 간격)
        
        print(f"   윈도우 크기: {window_frames}프레임 ({self.window_duration}초)")
        print(f"   스텝 크기: {step_frames}프레임 (50% 겹침)")
        
        tension_timestamps = []
        emotion_tensions = []
        audio_tensions = []
        combined_tensions = []
        
        # 이전 값 추적용
        prev_emotion_tension = 0.0
        silence_count = 0
        
        total_windows = (len(timestamps) - window_frames) // step_frames + 1
        print(f"   총 윈도우 수: {total_windows}")
        
        # 윈도우별 처리
        for i in range(0, len(timestamps) - window_frames + 1, step_frames):
            window_end = min(i + window_frames, len(timestamps))
            
            # 윈도우 중앙 시간
            center_time = timestamps[i + window_frames // 2]
            tension_timestamps.append(center_time)
            
            # 윈도우 데이터 추출
            window_rms = data['rms_values'][i:window_end]
            window_vad = data['vad_labels'][i:window_end]
            window_emotions = data['emotions'][i:window_end]
            window_face = data['face_detected'][i:window_end]
            
            # 텐션 계산
            emotion_tension, audio_tension, combined_tension, prev_emotion_tension, silence_count = \
                self._calculate_single_window_tension(
                    window_rms, window_vad, window_emotions, window_face,
                    prev_emotion_tension, silence_count
                )
            
            emotion_tensions.append(emotion_tension)
            audio_tensions.append(audio_tension)
            combined_tensions.append(combined_tension)
            
            # 진행 상황 출력
            current_window = len(tension_timestamps)
            if current_window % 100 == 0:
                print(f"   윈도우 처리: {current_window}/{total_windows} ({current_window/total_windows*100:.1f}%)")
        
        result = {
            'timestamps': tension_timestamps,
            'emotion_tension': emotion_tensions,
            'audio_tension': audio_tensions,
            'combined_tension': combined_tensions
        }
        
        print(f"✅ 텐션 계산 완료: {len(tension_timestamps)}개 윈도우")
        print(f"   평균 결합 텐션: {np.mean(combined_tensions):.2f}")
        print(f"   최대 결합 텐션: {np.max(combined_tensions):.2f}")
        
        return result
    
    def _calculate_single_window_tension(self, rms_values: np.ndarray, vad_labels: np.ndarray,
                                       emotions: np.ndarray, face_detected: np.ndarray,
                                       prev_emotion_tension: float, silence_count: int) -> Tuple:
        """단일 윈도우 텐션 계산 (개선된 로직)"""
        
        # 1. 얼굴 탐지 비율
        face_ratio = np.mean(face_detected)
        
        # 2. VAD 기반 Voice RMS 계산
        voice_frames = rms_values[vad_labels == 1]
        voice_rms = np.mean(voice_frames) if len(voice_frames) > 0 else 0.0
        voice_rms_norm = min(voice_rms / self.voice_rms_max, 1.0) * 5  # 0~5 스케일
        
        # 3. VAD 활동 비율
        vad_activity = np.mean(vad_labels)
        
        # 4. 감정 텐션 계산 (얼굴 있을 때만)
        if face_ratio >= 0.5:  # 얼굴이 충분히 보이는 경우
            emotion_tension = self._calculate_multi_emotion_score(emotions)
            prev_emotion_tension = emotion_tension  # 이전 값 업데이트
            silence_count = 0  # 침묵 카운트 리셋
        else:
            emotion_tension = 0.0  # 얼굴 없으면 감정 텐션 0
        
        # 5. 오디오 텐션
        audio_tension = voice_rms_norm
        
        # 6. 결합 텐션 (로직 처리 포함)
        if face_ratio >= 0.5:  # 얼굴 보임
            combined_tension = (self.emotion_weight * emotion_tension + 
                              self.audio_weight * audio_tension)
        elif vad_activity > self.config['audio']['vad_activity_threshold']:  # 얼굴 없어도 발화 있음
            # 이전 감정값 활용 + 현재 오디오
            combined_tension = (self.emotion_weight * prev_emotion_tension + 
                              self.audio_weight * audio_tension)
            silence_count = 0  # 침묵 카운트 리셋
        else:  # 둘 다 없음 - Decay 적용
            silence_count += 1
            
            # Decay 적용
            if silence_count >= (3.0 / self.window_duration):  # 3초 이상 침묵
                decay_rate = self.silence_3sec_decay
            else:
                decay_rate = self.decay_rate
            
            # 이전 결합 텐션에 decay 적용
            prev_combined = (self.emotion_weight * prev_emotion_tension + 
                           self.audio_weight * audio_tension)
            combined_tension = prev_combined * decay_rate
            prev_emotion_tension = prev_emotion_tension * decay_rate
        
        return emotion_tension, audio_tension, combined_tension, prev_emotion_tension, silence_count
    
    def _calculate_multi_emotion_score(self, emotions: np.ndarray) -> float:
        """멀티감정 점수 계산 (중립 제외 7감정 + Arousal*10)"""
        if len(emotions) == 0 or np.isnan(emotions).all():
            return 0.0
        
        # 유효한 감정 프레임만 사용
        valid_emotions = emotions[~np.isnan(emotions).any(axis=1)]
        if len(valid_emotions) == 0:
            return 0.0
        
        # 평균 감정 벡터
        avg_emotions = np.mean(valid_emotions, axis=0)
        
        # 1. 중립 제외 7가지 감정 (양수만)
        emotion_sum = 0.0
        for i in range(8):  # 0~7 감정 인덱스
            if i != self.neutral_idx:  # 중립(5) 제외
                emotion_val = max(avg_emotions[i], 0.0)  # 양수만
                emotion_sum += emotion_val
        
        # 2. Arousal (9번째 인덱스) - 양수만, 10배
        arousal_val = max(avg_emotions[9], 0.0) * self.arousal_multiplier
        
        # 3. 총 멀티감정 점수
        multi_emotion_score = emotion_sum + arousal_val
        
        return multi_emotion_score
    
    def _detect_edit_opportunities(self, tension_results: Dict, data: Dict) -> Dict:
        """편집 포인트 탐지"""
        print("✂️ 편집 포인트 탐지 중...")
        
        combined_tension = np.array(tension_results['combined_tension'])
        timestamps = tension_results['timestamps']
        
        edit_suggestions = {
            'highlights': [],
            'cut_points': [],
            'low_energy_periods': []
        }
        
        # 통계 계산
        tension_mean = np.mean(combined_tension)
        tension_std = np.std(combined_tension)
        highlight_threshold = tension_mean + self.highlight_sensitivity * tension_std
        
        print(f"   텐션 평균: {tension_mean:.2f}")
        print(f"   텐션 표준편차: {tension_std:.2f}")
        print(f"   하이라이트 임계값: {highlight_threshold:.2f}")
        
        # 편집 포인트 탐지
        for i in range(len(combined_tension)):
            current_tension = combined_tension[i]
            current_time = timestamps[i]
            
            # 하이라이트 (높은 텐션)
            if current_tension > highlight_threshold:
                edit_suggestions['highlights'].append({
                    'timestamp': float(current_time),
                    'tension': float(current_tension),
                    'type': 'peak'
                })
            
            # 급격한 변화 (Cut 포인트)
            if i > 0:
                change_rate = abs(current_tension - combined_tension[i-1])
                if change_rate > self.change_threshold:
                    cut_type = 'cut_in' if current_tension > combined_tension[i-1] else 'cut_out'
                    edit_suggestions['cut_points'].append({
                        'timestamp': float(current_time),
                        'change_rate': float(change_rate),
                        'type': cut_type
                    })
            
            # 낮은 에너지 구간
            if current_tension < self.low_tension_threshold:
                edit_suggestions['low_energy_periods'].append({
                    'timestamp': float(current_time),
                    'tension': float(current_tension)
                })
        
        print(f"✅ 편집 포인트 탐지 완료:")
        print(f"   하이라이트: {len(edit_suggestions['highlights'])}개")
        print(f"   컷 포인트: {len(edit_suggestions['cut_points'])}개")
        print(f"   저에너지 구간: {len(edit_suggestions['low_energy_periods'])}개")
        
        return edit_suggestions
    
    def _generate_json_result(self, data: Dict, tension_results: Dict, 
                            edit_suggestions: Dict, audio_h5_path: str, 
                            video_h5_path: str, youtube_url: str = None) -> Dict:
        """JSON 결과 생성"""
        print("📝 JSON 결과 생성 중...")
        
        # 기본 통계
        combined_tension = np.array(tension_results['combined_tension'])
        
        result = {
            'metadata': {
                'video_name': data['video_name'],
                'duration': float(data['timestamps'][-1]),
                'youtube_url': youtube_url,
                'processed_at': datetime.now().isoformat(),
                'audio_source': os.path.basename(audio_h5_path),
                'video_source': os.path.basename(video_h5_path)
            },
            'tension_timeline': {
                'timestamps': [float(t) for t in tension_results['timestamps']],
                'emotion_tension': [float(t) for t in tension_results['emotion_tension']],
                'audio_tension': [float(t) for t in tension_results['audio_tension']],
                'combined_tension': [float(t) for t in tension_results['combined_tension']]
            },
            'edit_suggestions': edit_suggestions,
            'statistics': {
                'avg_tension': float(np.mean(combined_tension)),
                'max_tension': float(np.max(combined_tension)),
                'min_tension': float(np.min(combined_tension)),
                'std_tension': float(np.std(combined_tension)),
                'highlight_count': len(edit_suggestions['highlights']),
                'cut_point_count': len(edit_suggestions['cut_points']),
                'low_energy_count': len(edit_suggestions['low_energy_periods']),
                'voice_activity_ratio': float(np.mean(data['vad_labels']))
            },
            'config_used': {
                'emotion_weight': self.emotion_weight,
                'audio_weight': self.audio_weight,
                'arousal_multiplier': self.arousal_multiplier,
                'window_duration': self.window_duration
            }
        }
        
        print(f"✅ JSON 결과 생성 완료")
        print(f"   메타데이터: {len(result['metadata'])}개 항목")
        print(f"   타임라인: {len(result['tension_timeline']['timestamps'])}개 포인트")
        
        return result
    
    def _save_tension_analysis(self, result: Dict, filename_pattern: str):
        """텐션 분석 결과 저장"""
        try:
            print("💾 결과 저장 시작...")
            
            # 파일명 생성
            safe_name = filename_pattern.replace('/', '_').replace('\\', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON 파일로 저장
            json_filename = f"tension_{safe_name}_{timestamp}.json"
            json_path = os.path.join(self.tension_output_dir, json_filename)
            
            print(f"   저장 경로: {json_path}")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 텐션 분석 결과 저장 완료: {json_filename}")
            
            # 요약 정보 출력
            stats = result['statistics']
            print(f"📊 최종 통계:")
            print(f"   평균 텐션: {stats['avg_tension']:.2f}")
            print(f"   최대 텐션: {stats['max_tension']:.2f}")
            print(f"   하이라이트: {stats['highlight_count']}개")
            print(f"   컷 포인트: {stats['cut_point_count']}개")
            print(f"   음성 활동 비율: {stats['voice_activity_ratio']:.1%}")
            
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            import traceback
            traceback.print_exc()


def main():
    """메인 실행 함수"""
    print("\n" + "="*50)
    print("멀티감정 텐션 분석기 실행")
    print("="*50)
    
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='멀티감정 기반 텐션 계산')
    parser.add_argument('filename_pattern', help='파일명 패턴 (오디오+비디오 자동 매칭)')
    parser.add_argument('--youtube_url', help='유튜브 URL (옵션)')
    parser.add_argument('--config', default='tension_analyzer/configs/tension_config.yaml', 
                       help='텐션 설정 파일 경로')
    
    try:
        args = parser.parse_args()
        print(f"📋 인자 파싱 완료:")
        print(f"   파일명 패턴: {args.filename_pattern}")
        print(f"   설정 파일: {args.config}")
        if args.youtube_url:
            print(f"   유튜브 URL: {args.youtube_url}")
        
    except SystemExit:
        print("❌ 인자 파싱 실패 또는 --help 요청")
        return
    
    try:
        # 텐션 계산기 실행
        print("\n🚀 텐션 계산기 초기화...")
        calculator = MultiEmotionTensionCalculator(args.config)
        
        # 텐션 계산
        print("\n⚡ 텐션 계산 시작...")
        result = calculator.calculate_tension(args.filename_pattern, args.youtube_url)
        
        if result:
            print("\n" + "="*50)
            print("✅ 멀티감정 텐션 분석 완료!")
            print("="*50)
            
            stats = result['statistics']
            print(f"📊 최종 결과:")
            print(f"   평균 텐션: {stats['avg_tension']:.2f}")
            print(f"   최대 텐션: {stats['max_tension']:.2f}")
            print(f"   최소 텐션: {stats['min_tension']:.2f}")
            print(f"   하이라이트: {stats['highlight_count']}개")
            print(f"   컷 포인트: {stats['cut_point_count']}개")
            
            # 상위 하이라이트 출력
            highlights = sorted(result['edit_suggestions']['highlights'], 
                              key=lambda x: x['tension'], reverse=True)[:3]
            if highlights:
                print(f"\n🎯 주요 하이라이트:")
                for i, hl in enumerate(highlights, 1):
                    timestamp_str = str(timedelta(seconds=int(hl['timestamp'])))
                    print(f"   {i}. {timestamp_str} (텐션: {hl['tension']:.2f})")
            
            print(f"\n💾 결과 파일: tension_analyzer/outputs/tension_data/")
            
        else:
            print("❌ 텐션 분석 실패")
            
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류: {e}")
        import traceback
        print("📍 상세 오류:")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("프로그램 종료")
    print("="*50)


if __name__ == "__main__":
    main()