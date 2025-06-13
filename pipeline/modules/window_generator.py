#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/modules/window_generator.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 및 기존 모듈 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class WindowGenerator:
    """
    클러스터 기반 윈도우 생성 및 재미도 점수 계산
    - 클러스터별 다양한 길이 윈도우 생성 (3/4 지점 배치)
    - dataset_generator 특징 추출 로직 재활용
    - XGBoost 모델로 재미도 점수 예측
    """
    
    def __init__(self, config_path: str = None):
        """
        윈도우 생성기 초기화
        
        Args:
            config_path (str): config 파일 경로
        """
        # 프로젝트 루트로 작업 디렉토리 변경
        os.chdir(project_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Config 로드
        if config_path is None:
            config_path = "pipeline/configs/funclip_extraction_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 윈도우 생성 파라미터
        window_config = config['window_generation']
        self.window_lengths = self._get_window_lengths(window_config)
        self.step_size = window_config['step_size']  # 1초
        self.position_ratio = window_config['position_ratio']  # 0.75 (3/4 지점)
        
        # 배치 처리 설정
        self.feature_batch_size = 100   # 특징 추출 배치 크기
        self.prediction_batch_size = 500  # 예측 배치 크기
        
        # XGBoost 모델 로드
        model_path = config['model']['path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)
        
        self.model = joblib.load(model_path)
        
        self.logger.info(f"✅ 윈도우 생성기 초기화 완료")
        self.logger.info(f"   윈도우 길이: {len(self.window_lengths)}개 ({min(self.window_lengths)}~{max(self.window_lengths)}초)")
        self.logger.info(f"   위치 비율: {self.position_ratio} (3/4 지점 배치)")
        self.logger.info(f"   XGBoost 모델: {os.path.relpath(model_path)}")
        self.logger.info(f"   배치 크기: 특징추출 {self.feature_batch_size}, 예측 {self.prediction_batch_size}")
    
    def _get_window_lengths(self, window_config: Dict) -> List[int]:
        """윈도우 길이 리스트 생성"""
        # 방법 1: length_range 사용
        if 'length_range' in window_config:
            range_config = window_config['length_range']
            return list(range(
                int(range_config['min']),
                int(range_config['max']) + 1,
                int(range_config.get('step', 1))
            ))
        
        # 방법 2: 명시적 lengths 리스트
        elif 'lengths' in window_config:
            # 문자열을 정수로 변환
            lengths = window_config['lengths']
            return [int(length) for length in lengths]
        
        # 방법 3: recommended + additional (기본값)
        else:
            recommended = window_config.get('recommended_lengths', [20, 25, 30])
            additional = window_config.get('additional_lengths', [18, 22, 28, 35, 40])
            
            # 문자열을 정수로 변환 후 중복 제거 및 정렬
            recommended = [int(x) for x in recommended]
            additional = [int(x) for x in additional]
            
            all_lengths = list(set(recommended + additional))
            all_lengths.sort()
            
            return all_lengths
    
    def load_clusters(self, clusters_json_path: str) -> Dict:
        """클러스터 JSON 파일 로드"""
        try:
            with open(clusters_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            clusters = data['clusters']
            metadata = data['metadata']
            
            self.logger.info(f"📊 클러스터 로드 완료: {len(clusters)}개")
            self.logger.info(f"   원본 하이라이트: {metadata['total_highlights']}개")
            self.logger.info(f"   확장된 클러스터: {metadata['single_expanded_count']}개")
            if 'video_name' in metadata:
                self.logger.info(f"   비디오: {metadata['video_name']}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ 클러스터 로드 실패: {e}")
            raise
    
    def generate_cluster_windows(self, cluster_data: Dict, video_duration: float) -> List[Dict]:
        """
        클러스터 기반 윈도우 생성
        
        Args:
            cluster_data (Dict): 클러스터 데이터 (전체 데이터)
            video_duration (float): 영상 전체 길이 (초)
            
        Returns:
            List[Dict]: 생성된 윈도우 리스트
        """
        self.logger.info("🔍 클러스터 기반 윈도우 생성 시작...")
        
        windows = []
        window_id = 0
        
        # 클러스터 리스트 추출
        clusters = cluster_data['clusters']
        
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            
            # 클러스터 span 계산 (또는 이미 계산된 span 사용)
            if 'span' in cluster:
                cluster_span = cluster['span']
                span_start = cluster_span['start']
                span_end = cluster_span['end']
            else:
                # span이 없는 경우 직접 계산
                timestamps = [p['timestamp'] for p in cluster['points']]
                if not timestamps:
                    self.logger.warning(f"⚠️ 클러스터 {cluster_id}: 포인트 없음")
                    continue
                span_start = min(timestamps)
                span_end = max(timestamps)
            
            self.logger.info(f"   클러스터 {cluster_id} 범위: {span_start:.1f}초 ~ {span_end:.1f}초")
            
            # 각 윈도우 길이별 윈도우 생성
            cluster_windows = 0
            for length in self.window_lengths:
                # 윈도우 시작 범위 계산 (하이라이트가 3/4 지점에 오도록)
                start_min = span_start - (length * self.position_ratio)
                start_max = span_end - (length * self.position_ratio)
                
                # 슬라이딩 윈도우 생성 (1초 간격)
                start = int(start_min) if start_min >= 0 else 0
                while start <= start_max and start <= video_duration - length:
                    end_time = start + length
                    
                    # 영상 범위 체크
                    if end_time <= video_duration:
                        # 윈도우에 해당하는 하이라이트 시간 (클러스터의 중간점 사용)
                        highlight_time = (span_start + span_end) / 2
                        
                        # 텐션 값 (클러스터 내 최대 텐션 또는 평균 텐션)
                        tensions = [float(p.get('tension', 0.0)) for p in cluster['points']]
                        highlight_tension = max(tensions) if tensions else 0.0
                        
                        windows.append({
                            'id': window_id,
                            'start_time': float(start),
                            'end_time': float(end_time),
                            'duration': length,
                            'cluster_id': cluster_id,
                            'highlight_time': float(highlight_time),
                            'highlight_tension': highlight_tension
                        })
                        window_id += 1
                        cluster_windows += 1
                    
                    # 다음 시작점 (1초 간격)
                    start += self.step_size
            
            self.logger.debug(f"   클러스터 {cluster_id}: {cluster_windows}개 윈도우")
        
        self.logger.info(f"✅ 윈도우 생성 완료: {len(windows)}개")
        if clusters:
            self.logger.info(f"   평균 윈도우/클러스터: {len(windows)/len(clusters):.1f}개")
        
        return windows
    
    def load_pipeline_data(self, video_h5_path: str, audio_h5_path: str, tension_json_path: str) -> Dict:
        """파이프라인 결과 데이터 로드 및 동기화"""
        self.logger.info("📂 파이프라인 데이터 로드 중...")
        
        try:
            # HDF5 파일들 로드
            video_data = PipelineUtils.load_video_hdf5(video_h5_path)
            audio_data = PipelineUtils.load_audio_hdf5(audio_h5_path)
            
            # 텐션 JSON 로드
            with open(tension_json_path, 'r', encoding='utf-8') as f:
                tension_data = json.load(f)
            
            # 데이터 동기화 (오디오 0.05초 기준)
            synced_data = self._synchronize_pipeline_data(video_data, audio_data, tension_data)
            
            self.logger.info(f"✅ 데이터 동기화 완료: {len(synced_data['timestamps'])}개 프레임")
            self.logger.info(f"   영상 길이: {synced_data['duration']:.1f}초")
            
            return synced_data
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 데이터 로드 실패: {e}")
            raise
    
    def _synchronize_pipeline_data(self, video_data: Dict, audio_data: Dict, tension_data: Dict) -> Dict:
        """비디오, 오디오, 텐션 데이터 동기화 (dataset_generator 로직 재사용)"""
        audio_timestamps = audio_data['sequences']['timestamps']
        video_timestamps = video_data['sequences']['timestamps']
        tension_timestamps = np.array(tension_data['tension_timeline']['timestamps'])
        
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
            synced_tension.append(tension_data['tension_timeline']['combined_tension'][tension_idx])
        
        return {
            'timestamps': audio_timestamps,
            'emotions': np.array(synced_emotions),
            'face_detected': np.array(synced_face_detected),
            'rms_values': audio_data['sequences']['rms_values'],
            'vad_labels': audio_data['sequences']['vad_labels'],
            'tension_values': np.array(synced_tension),
            'duration': video_data['metadata']['duration']
        }
    
    def extract_features_for_windows(self, windows: List[Dict], synced_data: Dict) -> np.ndarray:
        """
        윈도우들에 대한 특징 추출 (배치 처리)
        
        Args:
            windows (List[Dict]): 윈도우 리스트
            synced_data (Dict): 동기화된 파이프라인 데이터
            
        Returns:
            np.ndarray: [num_windows, 112] 특징 배열
        """
        self.logger.info(f"🧩 윈도우 특징 추출 시작: {len(windows)}개")
        
        all_features = []
        
        # 배치 처리
        for i in range(0, len(windows), self.feature_batch_size):
            batch_end = min(i + self.feature_batch_size, len(windows))
            batch_windows = windows[i:batch_end]
            
            # 배치 특징 추출
            batch_features = []
            for window in batch_windows:
                try:
                    features = self._extract_window_features(window, synced_data)
                    batch_features.append(features)
                except Exception as e:
                    # 실패한 윈도우는 0벡터로 처리
                    self.logger.warning(f"⚠️ 윈도우 {window['id']} 특징 추출 실패: {e}")
                    batch_features.append(np.zeros(112))
            
            all_features.extend(batch_features)
            
            # 진행 상황 출력
            progress = batch_end / len(windows) * 100
            self.logger.info(f"   특징 추출 진행: {batch_end}/{len(windows)} ({progress:.1f}%)")
        
        features_array = np.array(all_features)
        self.logger.info(f"✅ 특징 추출 완료: {features_array.shape}")
        
        return features_array
    
    def _extract_window_features(self, window: Dict, synced_data: Dict) -> np.ndarray:
        """단일 윈도우 특징 추출 (dataset_generator 로직 재사용)"""
        # 윈도우 시간 범위에 해당하는 인덱스 찾기
        start_time = window['start_time']
        end_time = window['end_time']
        
        timestamps = synced_data['timestamps']
        start_idx = np.searchsorted(timestamps, start_time)
        end_idx = np.searchsorted(timestamps, end_time)
        
        # 윈도우 데이터 추출
        window_data = {
            'emotions': synced_data['emotions'][start_idx:end_idx],
            'face_detected': synced_data['face_detected'][start_idx:end_idx],
            'rms_values': synced_data['rms_values'][start_idx:end_idx],
            'vad_labels': synced_data['vad_labels'][start_idx:end_idx],
            'tension_values': synced_data['tension_values'][start_idx:end_idx],
            'duration': end_time - start_time
        }
        
        # dataset_generator의 extract_features 로직 재사용 (config1 = 4구간)
        features = self._extract_features_config1(window_data)
        
        return features
    
    def _extract_features_config1(self, data: Dict) -> np.ndarray:
        """4구간 특징 추출 (dataset_generator 로직 재사용)"""
        num_segments = 4  # config1
        
        total_frames = len(data['emotions'])
        if total_frames == 0:
            return np.zeros(112)  # 28차원 × 4구간
        
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
            segment_tension = data['tension_values'][start_idx:end_idx]
            
            # 감정 특징 (20차원)
            valid_mask = segment_face > 0
            if np.sum(valid_mask) > 0:
                valid_emotions = segment_emotions[valid_mask]
                positive_emotions = np.maximum(valid_emotions, 0)
                emotion_mean = np.mean(positive_emotions, axis=0)
                emotion_std = np.std(positive_emotions, axis=0)
                emotion_features = np.concatenate([emotion_mean, emotion_std])
            else:
                emotion_features = np.zeros(20)
            
            # VAD 필터링된 오디오 특징 (4차원)
            voice_mask = segment_vad > 0
            non_voice_mask = segment_vad == 0
            
            if np.sum(voice_mask) > 0:
                voice_rms_values = segment_rms[voice_mask]
                voice_rms_mean = np.mean(voice_rms_values)
                voice_rms_max = np.max(voice_rms_values)
            else:
                voice_rms_mean = 0.0
                voice_rms_max = 0.0
            
            if np.sum(non_voice_mask) > 0:
                background_rms_mean = np.mean(segment_rms[non_voice_mask])
            else:
                background_rms_mean = 0.0
            
            total_rms_std = np.std(segment_rms) if len(segment_rms) > 0 else 0.0
            
            audio_features = np.array([
                voice_rms_mean,
                voice_rms_max,
                background_rms_mean,
                total_rms_std
            ])
            
            # VAD 특징 (1차원)
            vad_features = np.array([
                np.mean(segment_vad) if len(segment_vad) > 0 else 0.0
            ])
            
            # 텐션 특징 (3차원)
            if len(segment_tension) > 0:
                tension_features = np.array([
                    np.mean(segment_tension),
                    np.std(segment_tension),
                    np.max(segment_tension)
                ])
            else:
                tension_features = np.zeros(3)
            
            # 구간 특징 결합 (28차원)
            segment_features = np.concatenate([
                emotion_features,  # 20차원
                audio_features,    # 4차원
                vad_features,      # 1차원
                tension_features   # 3차원
            ])
            
            features.extend(segment_features)
        
        return np.array(features)  # 112차원 (28 × 4)
    
    def evaluate_with_xgb(self, features: np.ndarray) -> List[float]:
        """
        XGBoost 모델로 재미도 점수 예측 (배치 처리)
        
        Args:
            features (np.ndarray): [num_windows, 112] 특징 배열
            
        Returns:
            List[float]: 재미도 점수 리스트 (0~1)
        """
        self.logger.info(f"🤖 XGBoost 재미도 예측 시작: {len(features)}개")
        
        all_scores = []
        
        # 배치 처리
        for i in range(0, len(features), self.prediction_batch_size):
            batch_end = min(i + self.prediction_batch_size, len(features))
            batch_features = features[i:batch_end]
            
            # 예측 (Funny 클래스 확률)
            probabilities = self.model.predict_proba(batch_features)
            fun_scores = probabilities[:, 1]  # 클래스 1 (Funny) 확률
            
            all_scores.extend(fun_scores.tolist())
            
            # 진행 상황 출력
            progress = batch_end / len(features) * 100
            self.logger.info(f"   예측 진행: {batch_end}/{len(features)} ({progress:.1f}%)")
        
        self.logger.info(f"✅ 재미도 예측 완료")
        
        # 점수 통계
        scores_array = np.array(all_scores)
        self.logger.info(f"   점수 범위: {np.min(scores_array):.3f} ~ {np.max(scores_array):.3f}")
        self.logger.info(f"   평균 점수: {np.mean(scores_array):.3f}")
        
        return all_scores
    
    def generate_and_score_windows(self, clusters_json_path: str, video_h5_path: str, 
                                  audio_h5_path: str, tension_json_path: str) -> Dict:
        """
        전체 윈도우 생성 및 점수 계산 프로세스
        
        Args:
            clusters_json_path (str): 클러스터 JSON 파일 경로
            video_h5_path (str): 비디오 HDF5 파일 경로
            audio_h5_path (str): 오디오 HDF5 파일 경로
            tension_json_path (str): 텐션 JSON 파일 경로
            
        Returns:
            Dict: scored_windows.json 데이터
        """
        self.logger.info("🔍 윈도우 생성 및 점수 계산 프로세스 시작")
        
        # 1. 클러스터 로드
        cluster_data = self.load_clusters(clusters_json_path)
        
        # 2. 파이프라인 데이터 로드
        synced_data = self.load_pipeline_data(video_h5_path, audio_h5_path, tension_json_path)
        
        # 3. 윈도우 생성
        windows = self.generate_cluster_windows(cluster_data, synced_data['duration'])
        
        if not windows:
            raise ValueError("생성된 윈도우가 없습니다. 클러스터나 영상 길이를 확인하세요.")
        
        # 4. 특징 추출
        features = self.extract_features_for_windows(windows, synced_data)
        
        # 5. XGBoost 예측
        fun_scores = self.evaluate_with_xgb(features)
        
        # 6. 결과 조합
        for i, window in enumerate(windows):
            window['fun_score'] = fun_scores[i]
        
        # 비디오 이름 가져오기 (클러스터 메타데이터에서)
        video_name = cluster_data.get('metadata', {}).get('video_name', 'unknown')
        
        # 7. 결과 구성
        result = {
            'metadata': {
                'video_name': video_name,
                'total_windows': len(windows),
                'video_duration': synced_data['duration'],
                'score_statistics': {
                    'mean': float(np.mean(fun_scores)),
                    'std': float(np.std(fun_scores)),
                    'min': float(np.min(fun_scores)),
                    'max': float(np.max(fun_scores))
                },
                'source_files': {
                    'clusters': os.path.basename(clusters_json_path),
                    'video_h5': os.path.basename(video_h5_path),
                    'audio_h5': os.path.basename(audio_h5_path),
                    'tension': os.path.basename(tension_json_path)
                },
                'generated_at': datetime.now().isoformat()
            },
            'generation_config': {
                'window_lengths': self.window_lengths,
                'position_ratio': self.position_ratio,
                'step_size': self.step_size,
                'model_path': os.path.relpath(self.model.get_booster().save_config())
                if hasattr(self.model, 'get_booster') else 'xgb_model'
            },
            'windows': windows
        }
        
        self.logger.info("🔍 윈도우 생성 및 점수 계산 완료!")
        self.logger.info(f"   비디오: {video_name}")
        self.logger.info(f"   최종 윈도우: {len(windows)}개")
        self.logger.info(f"   평균 재미도: {np.mean(fun_scores):.3f}")
        
        return result
    
    def save_scored_windows(self, scored_windows: Dict, output_path: str) -> None:
        """
        점수가 매겨진 윈도우들을 JSON 파일로 저장
        
        Args:
            scored_windows (Dict): 점수 매겨진 윈도우 데이터
            output_path (str): 저장할 파일 경로
        """
        try:
            # 프로젝트 루트 기준 절대 경로 생성
            if not os.path.isabs(output_path):
                output_path = os.path.join(project_root, output_path)
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scored_windows, f, indent=2, ensure_ascii=False)
            
            # 상대 경로로 로그 출력
            relative_path = os.path.relpath(output_path, project_root)
            self.logger.info(f"💾 점수 윈도우 저장 완료: {relative_path}")
            
            # 요약 정보 출력
            metadata = scored_windows['metadata']
            self.logger.info(f"   총 윈도우: {metadata['total_windows']}개")
            self.logger.info(f"   평균 점수: {metadata['score_statistics']['mean']:.3f}")
            if 'video_name' in metadata:
                self.logger.info(f"   비디오: {metadata['video_name']}")
            
        except Exception as e:
            self.logger.error(f"❌ 점수 윈도우 저장 실패: {e}")
            raise


def main():
    """테스트 실행"""
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='윈도우 생성 및 재미도 점수 계산')
    parser.add_argument('clusters_json', help='클러스터 JSON 파일 경로')
    parser.add_argument('video_h5', help='비디오 HDF5 파일 경로')
    parser.add_argument('audio_h5', help='오디오 HDF5 파일 경로')
    parser.add_argument('tension_json', help='텐션 JSON 파일 경로')
    parser.add_argument('--output', help='출력 JSON 경로')
    parser.add_argument('--config', help='Config 파일 경로')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 윈도우 생성기 실행
        generator = WindowGenerator(config_path=args.config)
        
        # 윈도우 생성 및 점수 계산
        result = generator.generate_and_score_windows(
            args.clusters_json,
            args.video_h5,
            args.audio_h5,
            args.tension_json
        )
        
        # 결과 저장
        if args.output:
            output_path = args.output
        else:
            # 기본 출력 경로 생성
            # 클러스터 경로에서 디렉토리 구조 유지하여 저장
            clusters_dir = os.path.dirname(args.clusters_json)
            
            # 비디오 이름과 타임스탬프로 고유한 파일명 생성
            video_name = result['metadata']['video_name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(clusters_dir, f'scored_windows_{video_name}_{timestamp}.json')
        
        generator.save_scored_windows(result, output_path)
        
        print(f"\n✅ 윈도우 생성 및 점수 계산 완료!")
        print(f"📊 {result['metadata']['total_windows']}개 윈도우 생성")
        print(f"🎯 평균 재미도: {result['metadata']['score_statistics']['mean']:.3f}")
        print(f"💾 결과 저장: {os.path.relpath(output_path if os.path.isabs(output_path) else os.path.join(project_root, output_path), project_root)}")
        
    except Exception as e:
        print(f"❌ 윈도우 생성 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()