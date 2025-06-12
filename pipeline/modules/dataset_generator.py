#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from scipy import stats

# 프로젝트 루트로 이동
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_root = os.path.dirname(current_dir)  # pipeline/
project_root = os.path.dirname(pipeline_root)  # project_root/
os.chdir(project_root)

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils

class ChimchakmanDatasetGenerator:
    """
    침착맨 재미도 데이터셋 생성기
    - HDF5 파일들에서 특징 추출
    - 3가지 실험 설정 지원 (104/78/92차원)
    - 엄격한 검증 정책 적용
    - PipelineUtils 활용으로 기존 코드 재사용
    """
    
    def __init__(self, config_path: str = "pipeline/configs/dataset_config.yaml"):
        print(f"🏗️ 데이터셋 생성기 초기화")
        print(f"   설정 파일: {config_path}")
        
        # PipelineUtils로 설정 로드
        self.config = PipelineUtils.load_config(config_path)
        
        # 출력 디렉토리 설정
        self.output_dirs = PipelineUtils.setup_output_directories(self.config)
        
        # 로깅 설정
        self.logger = PipelineUtils.setup_logging(self.config, self.output_dirs)
        
        # 경로 설정
        self.input_base_dir = self.config['dataset']['input_base_dir']
        
        # 데이터셋 출력 경로
        self.dataset_output_dir = self.config['dataset']['dataset_output_dir']
        self.hdf5_filename = self.config['dataset']['hdf5_filename']
        self.dataset_path = os.path.join(self.dataset_output_dir, self.hdf5_filename)
        
        # 검증 설정
        self.min_frames_per_segment = self.config['dataset']['validation']['min_frames_per_segment']
        self.min_duration = self.config['dataset']['validation']['min_clip_duration']
        self.max_duration = self.config['dataset']['validation']['max_clip_duration']
        
        # 출력 디렉토리 생성
        os.makedirs(self.dataset_output_dir, exist_ok=True)
        
        self.logger.info("✅ 데이터셋 생성기 초기화 완료")
        self.logger.info(f"   입력 경로: {self.input_base_dir}")
        self.logger.info(f"   출력 경로: {self.dataset_path}")
        self.logger.info(f"   검증 정책: 구간별 최소 {self.min_frames_per_segment}프레임")
    
    def is_clip_already_processed(self, clip_id: str) -> bool:
        """클립이 이미 데이터셋에 있는지 확인"""
        if not os.path.exists(self.dataset_path):
            return False
        
        try:
            import h5py
            with h5py.File(self.dataset_path, 'r') as f:
                current_size = f.attrs.get('current_size', 0)
                if current_size == 0:
                    return False
                
                existing_clip_ids = f['clip_ids'][:current_size]
                # 문자열 비교 (bytes to str 변환 필요할 수 있음)
                for existing_id in existing_clip_ids:
                    if isinstance(existing_id, bytes):
                        existing_id = existing_id.decode('utf-8')
                    if existing_id == clip_id:
                        return True
                return False
        except Exception as e:
            self.logger.warning(f"클립 중복 검사 실패: {e}")
            return False
        """클립 폴더 스캔 및 자동 라벨링"""
    def scan_clips(self) -> List[Dict]:
        """클립 폴더 스캔 및 자동 라벨링"""
        self.logger.info(f"📁 클립 폴더 스캔: {self.input_base_dir}")
        
        clips = []
        label_mapping = self.config['dataset']['label_mapping']
        extensions = self.config['dataset']['clip_extensions']
        
        for label_dir, label_value in label_mapping.items():
            folder_path = os.path.join(self.input_base_dir, label_dir)
            
            if not os.path.exists(folder_path):
                self.logger.warning(f"폴더 없음: {folder_path}")
                continue
            
            # 클립 파일 찾기
            clip_files = []
            for ext in extensions:
                pattern = f"*{ext}"
                clip_files.extend(Path(folder_path).glob(pattern))
            
            for clip_path in clip_files:
                clip_id = clip_path.stem
                clips.append({
                    'clip_id': clip_id,
                    'clip_path': str(clip_path),
                    'label': label_value,
                    'label_name': label_dir
                })
            
            self.logger.info(f"   {label_dir}: {len(clip_files)}개 클립 (label={label_value})")
        
        self.logger.info(f"✅ 총 {len(clips)}개 클립 발견")
        return clips
    
    def find_pipeline_files(self, clip_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """클립 ID로 파이프라인 결과 파일들 찾기"""
        video_file = None
        audio_file = None
        tension_file = None
        
        base_dir = self.config['output']['base_dir']  # "dataset/preprocessed"
        preprocessed_dir = self.config['output']['preprocessed_dir']  # "preprocessed_data"
        
        # 비디오 HDF5 찾기
        video_seq_dir = os.path.join(base_dir, preprocessed_dir, 
                                   self.config['output']['video_sequence_dir'])
        if os.path.exists(video_seq_dir):
            for file in os.listdir(video_seq_dir):
                if clip_id in file and file.endswith('.h5'):
                    video_file = os.path.join(video_seq_dir, file)
                    break
        
        # 오디오 HDF5 찾기  
        audio_seq_dir = os.path.join(base_dir, preprocessed_dir,
                                   self.config['output']['audio_sequence_dir'])
        if os.path.exists(audio_seq_dir):
            for file in os.listdir(audio_seq_dir):
                if clip_id in file and file.endswith('.h5'):
                    audio_file = os.path.join(audio_seq_dir, file)
                    break
        
        # 텐션 JSON 찾기
        tension_dir = os.path.join(base_dir, self.config['output']['tension_analysis_dir'])
        if os.path.exists(tension_dir):
            for file in os.listdir(tension_dir):
                if clip_id in file and file.endswith('.json'):
                    tension_file = os.path.join(tension_dir, file)
                    break
        
        return video_file, audio_file, tension_file
    
    def load_tension_json(self, json_path: str) -> Optional[Dict]:
        """텐션 분석 JSON 파일 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                tension_data = json.load(f)
            
            return {
                'timestamps': np.array(tension_data['tension_timeline']['timestamps']),
                'combined_tension': np.array(tension_data['tension_timeline']['combined_tension'])
            }
        except Exception as e:
            self.logger.error(f"텐션 JSON 로드 실패: {e}")
            return None
    
    def load_pipeline_data(self, video_h5_path: str, audio_h5_path: str, tension_json_path: str) -> Optional[Dict]:
        """파이프라인 결과 파일들에서 데이터 로드 (PipelineUtils 활용)"""
        try:
            # PipelineUtils로 HDF5 로드
            video_data = PipelineUtils.load_video_hdf5(video_h5_path)
            audio_data = PipelineUtils.load_audio_hdf5(audio_h5_path)
            tension_data = self.load_tension_json(tension_json_path)
            
            if not video_data or not audio_data or not tension_data:
                return None
            
            # 데이터 동기화 (오디오 타임스탬프 기준)
            synced_data = self._synchronize_pipeline_data(video_data, audio_data, tension_data)
            return synced_data
            
        except Exception as e:
            self.logger.error(f"파이프라인 데이터 로드 실패: {e}")
            return None
    
    def _synchronize_pipeline_data(self, video_data: Dict, audio_data: Dict, tension_data: Dict) -> Dict:
        """비디오, 오디오, 텐션 데이터 동기화"""
        audio_timestamps = audio_data['sequences']['timestamps']
        video_timestamps = video_data['sequences']['timestamps']
        tension_timestamps = tension_data['timestamps']
        
        # 비디오 데이터를 오디오 타임스탬프에 맞춰 보간
        synced_emotions = []
        synced_face_detected = []
        synced_tension = []
        
        for audio_ts in audio_timestamps:
            # 가장 가까운 비디오 프레임 찾기
            video_idx = np.argmin(np.abs(video_timestamps - audio_ts))
            
            emotion_frame = video_data['sequences']['emotions'][video_idx]
            if np.isnan(emotion_frame).any():
                emotion_frame = np.zeros(10)
            synced_emotions.append(emotion_frame)
            synced_face_detected.append(video_data['sequences']['face_detected'][video_idx])
            
            # 가장 가까운 텐션 값 찾기
            tension_idx = np.argmin(np.abs(tension_timestamps - audio_ts))
            synced_tension.append(tension_data['combined_tension'][tension_idx])
        
        return {
            'timestamps': audio_timestamps,
            'emotions': np.array(synced_emotions),
            'face_detected': np.array(synced_face_detected),
            'rms_values': audio_data['sequences']['rms_values'],
            'vad_labels': audio_data['sequences']['vad_labels'],
            'tension_values': np.array(synced_tension),
            'analysis_interval': audio_data['metadata']['analysis_interval'],
            'duration': video_data['metadata']['duration']
        }
    
    def validate_clip(self, data: Dict, config_name: str) -> Tuple[bool, str]:
        """클립 검증 (엄격한 정책)"""
        duration = data['duration']
        
        # 길이 검증
        if duration < self.min_duration or duration > self.max_duration:
            return False, f"길이 부적합: {duration:.1f}초 ({self.min_duration}-{self.max_duration}초 범위 벗어남)"
        
        # 구간별 얼굴 프레임 수 검증
        config = self.config['dataset']['feature_configs'][config_name]
        num_segments = config['segments']
        
        total_frames = len(data['face_detected'])
        frames_per_segment = total_frames // num_segments
        
        for i in range(num_segments):
            start_idx = i * frames_per_segment
            end_idx = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
            
            segment_faces = data['face_detected'][start_idx:end_idx]
            valid_frames = np.sum(segment_faces)
            
            if valid_frames < self.min_frames_per_segment:
                return False, f"구간 {i+1}: {valid_frames}프레임 < {self.min_frames_per_segment}프레임"
        
        return True, "통과"
    
    def extract_features(self, data: Dict, config_name: str) -> np.ndarray:
        """특징 추출 (설정별) - RMS 회귀 특성 추가"""
        config = self.config['dataset']['feature_configs'][config_name]
        num_segments = config['segments']
        use_regression = config['use_regression']
        
        total_frames = len(data['emotions'])
        frames_per_segment = total_frames // num_segments
        
        features = []
        
        for i in range(num_segments):
            start_idx = i * frames_per_segment
            end_idx = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
            
            # 구간 데이터 추출
            segment_emotions = data['emotions'][start_idx:end_idx]
            segment_rms = data['rms_values'][start_idx:end_idx]
            segment_vad = data['vad_labels'][start_idx:end_idx]
            segment_face = data['face_detected'][start_idx:end_idx]
            
            # 얼굴 있는 프레임만 사용
            valid_mask = segment_face > 0
            if np.sum(valid_mask) > 0:
                valid_emotions = segment_emotions[valid_mask]
                
                # 양수 변환 후 감정 특징 계산
                positive_emotions = np.maximum(valid_emotions, 0)
                
                # 감정 특징 (20차원: 평균 10 + 표준편차 10)
                emotion_mean = np.mean(positive_emotions, axis=0)
                emotion_std = np.std(positive_emotions, axis=0)
                
                # 회귀 특징 (20차원 추가)
                if use_regression:
                    emotion_slope = []
                    emotion_r2 = []
                    
                    for dim in range(10):
                        if len(positive_emotions) >= 3:  # 최소 3개 점 필요
                            y = positive_emotions[:, dim]
                            x = np.arange(len(y))
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            emotion_slope.append(slope)
                            emotion_r2.append(max(0, r_value ** 2))  # 음수 방지
                        else:
                            emotion_slope.append(0.0)
                            emotion_r2.append(0.0)
                    
                    emotion_features = np.concatenate([
                        emotion_mean,      # 10차원
                        emotion_std,       # 10차원  
                        emotion_slope,     # 10차원
                        emotion_r2         # 10차원
                    ])  # 총 40차원
                else:
                    emotion_features = np.concatenate([
                        emotion_mean,      # 10차원
                        emotion_std        # 10차원
                    ])  # 총 20차원
            else:
                # 얼굴 없으면 0으로 채움
                if use_regression:
                    emotion_features = np.zeros(40)
                else:
                    emotion_features = np.zeros(20)
            
            # 🆕 오디오 특징 확장 (RMS 회귀 추가)
            rms_mean = np.mean(segment_rms)
            rms_std = np.std(segment_rms)
            
            # RMS 회귀 분석
            if len(segment_rms) >= 3 and np.std(segment_rms) > 1e-8:
                x = np.arange(len(segment_rms))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, segment_rms)
                rms_slope = slope
                rms_r2 = max(0, r_value ** 2)  # 음수 방지
            else:
                rms_slope = 0.0
                rms_r2 = 0.0
            
            # 오디오 특징 (4차원: 평균 + 표준편차 + 기울기 + R²)
            audio_features = np.array([
                rms_mean,
                rms_std,
                rms_slope,
                rms_r2
            ])
            
            # VAD 특징 (1차원: 발화 비율)
            vad_features = np.array([
                np.mean(segment_vad)
            ])
            
            # 텐션 특징 (3차원: 평균 + 표준편차 + 최대값)
            segment_tension = data['tension_values'][start_idx:end_idx]
            
            tension_features = np.array([
                np.mean(segment_tension),
                np.std(segment_tension),
                np.max(segment_tension)
            ])
            
            # 구간 특징 결합
            if use_regression:
                segment_features = np.concatenate([
                    emotion_features,  # 40차원
                    audio_features,    # 4차원 (기존 2차원 → 4차원)
                    vad_features,      # 1차원
                    tension_features   # 3차원
                ])  # 총 48차원 (기존 46차원 → 48차원)
            else:
                segment_features = np.concatenate([
                    emotion_features,  # 20차원
                    audio_features,    # 4차원 (기존 2차원 → 4차원)
                    vad_features,      # 1차원
                    tension_features   # 3차원
                ])  # 총 28차원 (기존 26차원 → 28차원)
            
            features.extend(segment_features)
        
        return np.array(features)
    
    def create_or_append_dataset(self, features_dict: Dict, label: int, clip_id: str):
        """HDF5 데이터셋 생성 또는 추가"""
        import h5py
        
        if not os.path.exists(self.dataset_path):
            # 새 데이터셋 생성
            self._create_new_dataset(features_dict, label, clip_id)
        else:
            # 기존 데이터셋에 추가
            self._append_to_dataset(features_dict, label, clip_id)
    
    def _create_new_dataset(self, features_dict: Dict, label: int, clip_id: str):
        """새 HDF5 데이터셋 생성"""
        import h5py
        
        self.logger.info(f"📦 새 데이터셋 생성: {self.dataset_path}")
        
        initial_size = self.config['dataset']['hdf5']['initial_size']
        
        with h5py.File(self.dataset_path, 'w') as f:
            # 메타데이터
            f.attrs['dataset_name'] = self.config['dataset']['dataset_name']
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['version'] = '1.0'
            
            # 각 설정별 특징 데이터셋
            for config_name, features in features_dict.items():
                config = self.config['dataset']['feature_configs'][config_name]
                dims = config['dimensions']
                
                # 특징 데이터셋
                f.create_dataset(f'features_{config_name}', 
                               shape=(initial_size, dims),
                               dtype='float32',
                               compression=None)
                f[f'features_{config_name}'][0] = features
            
            # 라벨 데이터셋
            f.create_dataset('labels', 
                           shape=(initial_size,),
                           dtype='int32',
                           compression=None)
            f['labels'][0] = label
            
            # 클립 ID 데이터셋
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('clip_ids',
                           shape=(initial_size,),
                           dtype=dt,
                           compression=None)
            f['clip_ids'][0] = clip_id
            
            # 현재 크기 추적
            f.attrs['current_size'] = 1
    
    def _append_to_dataset(self, features_dict: Dict, label: int, clip_id: str):
        """기존 데이터셋에 추가"""
        import h5py
        
        with h5py.File(self.dataset_path, 'a') as f:
            current_size = f.attrs['current_size']
            
            # 각 설정별 특징 추가
            for config_name, features in features_dict.items():
                dataset_name = f'features_{config_name}'
                
                # 필요시 크기 확장
                if current_size >= f[dataset_name].shape[0]:
                    new_size = f[dataset_name].shape[0] * 2
                    f[dataset_name].resize((new_size, f[dataset_name].shape[1]))
                
                f[dataset_name][current_size] = features
            
            # 라벨 추가
            if current_size >= f['labels'].shape[0]:
                f['labels'].resize((f['labels'].shape[0] * 2,))
            f['labels'][current_size] = label
            
            # 클립 ID 추가
            if current_size >= f['clip_ids'].shape[0]:
                f['clip_ids'].resize((f['clip_ids'].shape[0] * 2,))
            f['clip_ids'][current_size] = clip_id
            
            # 크기 업데이트
            f.attrs['current_size'] = current_size + 1
    
    def generate_dataset(self) -> Dict:
        """전체 데이터셋 생성"""
        PipelineUtils.print_step_banner(6, "데이터셋 생성", "파이프라인 결과를 통합 데이터셋으로 변환")
        
        # 클립 스캔
        clips = self.scan_clips()
        if not clips:
            raise ValueError("처리할 클립이 없습니다")
        
        stats = {
            'total_clips': len(clips),
            'processed_clips': 0,
            'failed_clips': 0,
            'label_distribution': {},
            'config_results': {}
        }
        
        # 각 클립 처리
        for i, clip in enumerate(clips):
            self.logger.info(f"\n📹 클립 처리 {i+1}/{len(clips)}: {clip['clip_id']}")
            
            # 중복 검사
            if self.is_clip_already_processed(clip['clip_id']):
                self.logger.info(f"⏩ 이미 처리된 클립, 건너뛰기: {clip['clip_id']}")
                stats['processed_clips'] += 1  # 이미 처리된 것으로 간주
                label_name = clip['label_name']
                stats['label_distribution'][label_name] = stats['label_distribution'].get(label_name, 0) + 1
                continue
            
            try:
                # 파이프라인 결과 파일들 찾기
                video_h5, audio_h5, tension_json = self.find_pipeline_files(clip['clip_id'])
                if not video_h5 or not audio_h5 or not tension_json:
                    missing = []
                    if not video_h5: missing.append("video_h5")
                    if not audio_h5: missing.append("audio_h5") 
                    if not tension_json: missing.append("tension_json")
                    self.logger.warning(f"파일 없음: {missing}")
                    stats['failed_clips'] += 1
                    continue
                
                # 데이터 로드
                data = self.load_pipeline_data(video_h5, audio_h5, tension_json)
                if data is None:
                    self.logger.error(f"데이터 로드 실패")
                    stats['failed_clips'] += 1
                    continue
                
                # 각 설정별 특징 추출
                features_dict = {}
                valid_configs = []
                
                for config_name in self.config['dataset']['feature_configs'].keys():
                    # 검증
                    is_valid, reason = self.validate_clip(data, config_name)
                    if not is_valid:
                        self.logger.warning(f"{config_name} 검증 실패: {reason}")
                        continue
                    
                    # 특징 추출
                    features = self.extract_features(data, config_name)
                    features_dict[config_name] = features
                    valid_configs.append(config_name)
                    self.logger.info(f"✅ {config_name}: {len(features)}차원 특징 추출")
                
                # 유효한 설정이 있으면 데이터셋에 추가
                if features_dict:
                    self.create_or_append_dataset(features_dict, clip['label'], clip['clip_id'])
                    stats['processed_clips'] += 1
                    
                    # 라벨 분포 업데이트
                    label_name = clip['label_name']
                    stats['label_distribution'][label_name] = stats['label_distribution'].get(label_name, 0) + 1
                    
                    self.logger.info(f"✅ 데이터셋 추가 완료")
                else:
                    self.logger.error(f"모든 설정 검증 실패")
                    stats['failed_clips'] += 1
                
            except Exception as e:
                self.logger.error(f"클립 처리 실패: {e}")
                stats['failed_clips'] += 1
        
        PipelineUtils.print_completion_banner(6, "데이터셋 생성", 
            f"처리: {stats['processed_clips']}개, 실패: {stats['failed_clips']}개, 분포: {stats['label_distribution']}")
        
        return stats


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='침착맨 재미도 데이터셋 생성')
    parser.add_argument('--config', default='pipeline/configs/dataset_config.yaml',
                       help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        generator = ChimchakmanDatasetGenerator(args.config)
        stats = generator.generate_dataset()
        
        print(f"\n✅ 데이터셋 생성 성공!")
        print(f"📄 저장 위치: {generator.dataset_path}")
        
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()