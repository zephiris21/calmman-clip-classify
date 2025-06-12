#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# 🎯 기존 패턴 그대로 사용 - 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent  # clips_training/src/utils → project_root
sys.path.insert(0, str(project_root))

# 기존 모듈 재사용
from pipeline.utils.pipeline_utils import PipelineUtils


class TrainingUtils(PipelineUtils):
    """
    학습용 유틸리티 클래스 (PipelineUtils 확장)
    - 기존 PipelineUtils의 모든 기능 상속
    - 학습 특화 기능 추가
    """
    
    @staticmethod
    def load_training_config(config_path: str = "clips_training/configs/training_config.yaml") -> Dict:
        """
        학습 설정 파일 로드
        
        Args:
            config_path (str): 설정 파일 경로
            
        Returns:
            Dict: 로드된 학습 설정
        """
        try:
            # PipelineUtils의 load_config 재사용
            config = PipelineUtils.load_config(config_path)
            print(f"✅ 학습 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 학습 설정 파일 로드 실패: {e}")
            raise
    
    @staticmethod
    def setup_training_directories(config: Dict) -> Dict:
        """
        학습용 출력 디렉토리 생성
        
        Args:
            config (Dict): 학습 설정
            
        Returns:
            Dict: 생성된 디렉토리 경로들
        """
        base_dir = config['output']['base_dir']
        
        dirs = {
            'base': base_dir,
            'models': os.path.join(base_dir, config['output']['models_dir']),
            'results': os.path.join(base_dir, config['output']['results_dir']),
            'logs': os.path.join(base_dir, config['output']['logs_dir'])
        }
        
        # Config별 모델 디렉토리 추가 생성
        target_config = config['data']['target_config']
        dirs['target_models'] = os.path.join(dirs['models'], f'config{target_config}')
        
        # 모든 디렉토리 생성
        for dir_name, dir_path in dirs.items():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 학습용 출력 디렉토리 생성 완료: {base_dir}")
        print(f"   타겟 모델 디렉토리: config{target_config}")
        return dirs
    
    @staticmethod
    def setup_training_logging(config: Dict, output_dirs: Dict) -> 'logging.Logger':
        """
        학습용 로깅 시스템 설정
        
        Args:
            config (Dict): 학습 설정
            output_dirs (Dict): 출력 디렉토리 정보
            
        Returns:
            logging.Logger: 설정된 로거
        """
        import logging
        
        logger = logging.getLogger('ClipsTraining')
        logger.setLevel(getattr(logging, config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        if config['logging']['console_output']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 파일 핸들러
        if config['logging']['file_output']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = config['logging']['log_filename'].format(timestamp=timestamp)
            log_path = os.path.join(output_dirs['logs'], log_filename)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"📄 학습 로그 파일: {log_path}")
        
        return logger
    
    @staticmethod
    def load_dataset_hdf5(dataset_path: str, target_config: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        dataset.h5에서 데이터 로드 (동적 차원 지원)
        
        Args:
            dataset_path (str): dataset.h5 파일 경로
            target_config (int): 타겟 설정 (1, 2, 3 중 선택)
            
        Returns:
            Tuple: (X, y, clip_ids, metadata)
                - X: 특징 배열 (n_samples, n_features)
                - y: 라벨 배열 (n_samples,)
                - clip_ids: 클립 ID 리스트
                - metadata: 데이터셋 메타데이터
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        
        try:
            with h5py.File(dataset_path, 'r') as f:
                # 현재 데이터 크기 확인
                current_size = f.attrs['current_size']
                
                # 특징 데이터 로드 (실제 차원 사용)
                features_key = f'features_config_{target_config}'
                if features_key not in f:
                    available_configs = [key for key in f.keys() if key.startswith('features_config')]
                    raise KeyError(f"config{target_config}를 찾을 수 없습니다. 사용 가능한 설정: {available_configs}")
                
                X = f[features_key][:current_size]
                y = f['labels'][:current_size] 
                
                # 클립 ID 로드 (문자열 처리)
                clip_ids_raw = f['clip_ids'][:current_size]
                clip_ids = []
                for clip_id in clip_ids_raw:
                    if isinstance(clip_id, bytes):
                        clip_ids.append(clip_id.decode('utf-8'))
                    else:
                        clip_ids.append(str(clip_id))
                
                # 메타데이터 수집
                metadata = {
                    'dataset_name': f.attrs.get('dataset_name', ''),
                    'created_at': f.attrs.get('created_at', ''),
                    'version': f.attrs.get('version', ''),
                    'features': f.attrs.get('features', ''),
                    'total_samples': current_size,
                    'n_features': X.shape[1],
                    'target_config': target_config
                }
                
                # 실제 차원 정보 추가
                if f'{target_config}_dimensions' in f.attrs:
                    metadata['actual_dimensions'] = f.attrs[f'{target_config}_dimensions']
                
                print(f"✅ 데이터셋 로드 완료:")
                print(f"   설정: config{target_config}")
                print(f"   샘플 수: {current_size}")
                print(f"   특징 차원: {X.shape[1]}")
                print(f"   클래스 분포: {np.bincount(y)}")
                
                return X, y, clip_ids, metadata
                
        except Exception as e:
            print(f"❌ 데이터셋 로드 실패: {e}")
            raise
    
    @staticmethod
    def get_feature_names(target_config: int = 1) -> List[str]:
        """
        동적 특징명 생성 (실제 차원 기반)
        
        Args:
            target_config (int): 타겟 설정 (1, 2, 3)
            
        Returns:
            List[str]: 특징명 리스트
        """
        # 28차원 기본 특징 정의 (구간별 반복)
        base_features = []
        
        # 감정 특징 (20차원: 평균 10 + 표준편차 10)
        emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 
                   'neutral', 'sadness', 'surprise', 'valence', 'arousal']
        for emotion in emotions:
            base_features.append(f'emotion_{emotion}_mean')
        for emotion in emotions:
            base_features.append(f'emotion_{emotion}_std')
        
        # 오디오 특징 (4차원: VAD 필터링)
        base_features.extend([
            'voice_rms_mean',       # 발화 평균 음량
            'voice_rms_max',        # 발화 최대 음량 (핵심!)
            'background_rms_mean',  # 배경음 평균
            'total_rms_std'         # 전체 변동성
        ])
        
        # VAD 특징 (1차원)
        base_features.append('vad_ratio')
        
        # 텐션 특징 (3차원)
        base_features.extend([
            'tension_mean',
            'tension_std', 
            'tension_max'
        ])
        
        # 구간별 특징명 생성
        segments = 4 if target_config == 1 else (3 if target_config == 2 else 2)
        feature_names = []
        
        for seg in range(1, segments + 1):
            for feat in base_features:
                feature_names.append(f'segment{seg}_{feat}')
        
        expected_dims = len(base_features) * segments
        print(f"✅ 특징명 생성 완료:")
        print(f"   config{target_config}: {segments}구간 × {len(base_features)}차원 = {expected_dims}차원")
        
        return feature_names
    
    @staticmethod
    def get_feature_blocks(target_config: int = 1) -> Dict[str, List[int]]:
        """
        특징 블록별 인덱스 반환 (EDA용)
        
        Args:
            target_config (int): 타겟 설정
            
        Returns:
            Dict: 블록별 특징 인덱스 딕셔너리
        """
        segments = 4 if target_config == 1 else (3 if target_config == 2 else 2)
        block_size = 28  # 구간별 특징 수
        
        blocks = {}
        
        for seg in range(segments):
            seg_start = seg * block_size
            
            # 각 구간별 블록 인덱스
            blocks[f'segment{seg+1}_emotion'] = list(range(seg_start, seg_start + 20))
            blocks[f'segment{seg+1}_audio'] = list(range(seg_start + 20, seg_start + 24))
            blocks[f'segment{seg+1}_vad'] = [seg_start + 24]
            blocks[f'segment{seg+1}_tension'] = list(range(seg_start + 25, seg_start + 28))
        
        # 전체 블록별 인덱스 (구간 통합)
        blocks['all_emotion'] = []
        blocks['all_audio'] = []
        blocks['all_vad'] = []
        blocks['all_tension'] = []
        
        for seg in range(segments):
            seg_start = seg * block_size
            blocks['all_emotion'].extend(range(seg_start, seg_start + 20))
            blocks['all_audio'].extend(range(seg_start + 20, seg_start + 24))
            blocks['all_vad'].append(seg_start + 24)
            blocks['all_tension'].extend(range(seg_start + 25, seg_start + 28))
        
        return blocks
    
    @staticmethod
    def get_key_feature_indices(feature_names: List[str], config: Dict) -> Dict[str, List[int]]:
        """
        핵심 특징들의 인덱스 반환 (설정 기반)
        
        Args:
            feature_names (List[str]): 전체 특징명 리스트
            config (Dict): 학습 설정
            
        Returns:
            Dict: 핵심 특징 그룹별 인덱스
        """
        key_features = config['features']['key_features']
        key_indices = {}
        
        for group_name, feature_patterns in key_features.items():
            indices = []
            for pattern in feature_patterns:
                # 패턴과 매치되는 특징 찾기
                matching_indices = [i for i, name in enumerate(feature_names) 
                                  if pattern in name]
                indices.extend(matching_indices)
            
            key_indices[group_name] = sorted(list(set(indices)))  # 중복 제거 및 정렬
        
        return key_indices
    
    @staticmethod
    def analyze_dataset_info(X: np.ndarray, y: np.ndarray, clip_ids: List[str], 
                           feature_names: List[str], logger=None) -> Dict:
        """
        데이터셋 기본 정보 분석
        
        Args:
            X: 특징 배열
            y: 라벨 배열  
            clip_ids: 클립 ID 리스트
            feature_names: 특징명 리스트
            logger: 로거 (선택적)
            
        Returns:
            Dict: 분석 결과
        """
        # 기본 통계
        n_samples, n_features = X.shape
        class_counts = np.bincount(y)
        class_ratio = class_counts[1] / (class_counts[0] + class_counts[1])
        
        # 특징 통계
        feature_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0),
            'min': np.min(X, axis=0),
            'max': np.max(X, axis=0),
            'nan_count': np.sum(np.isnan(X), axis=0),
            'inf_count': np.sum(np.isinf(X), axis=0)
        }
        
        analysis = {
            'basic_info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'class_counts': class_counts.tolist(),
                'class_ratio': class_ratio,
                'is_balanced': abs(class_ratio - 0.5) < 0.1
            },
            'feature_stats': feature_stats,
            'data_quality': {
                'has_nan': np.any(feature_stats['nan_count'] > 0),
                'has_inf': np.any(feature_stats['inf_count'] > 0),
                'features_with_zero_var': np.sum(feature_stats['std'] == 0)
            }
        }
        
        # 로깅
        log_func = logger.info if logger else print
        log_func(f"📊 데이터셋 기본 분석:")
        log_func(f"   샘플 수: {n_samples} (Boring: {class_counts[0]}, Funny: {class_counts[1]})")
        log_func(f"   특징 수: {n_features}")
        log_func(f"   클래스 균형: {'✅' if analysis['basic_info']['is_balanced'] else '⚠️'} (Funny 비율: {class_ratio:.1%})")
        
        if analysis['data_quality']['has_nan']:
            log_func(f"   ⚠️ NaN 값 발견")
        if analysis['data_quality']['has_inf']:
            log_func(f"   ⚠️ Inf 값 발견") 
        if analysis['data_quality']['features_with_zero_var'] > 0:
            log_func(f"   ⚠️ 분산이 0인 특징: {analysis['data_quality']['features_with_zero_var']}개")
        
        return analysis
    
    @staticmethod
    def save_model_artifacts(model, scaler, feature_names: List[str], 
                           config: Dict, output_dirs: Dict, 
                           metrics: Dict = None) -> Dict:
        """
        모델 및 관련 파일들 저장
        
        Args:
            model: 학습된 모델
            scaler: 전처리 스케일러
            feature_names: 특징명 리스트
            config: 학습 설정
            output_dirs: 출력 디렉토리
            metrics: 성능 지표 (선택적)
            
        Returns:
            Dict: 저장된 파일 경로들
        """
        import pickle
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_config = config['data']['target_config']
        
        saved_files = {}
        
        # 모델 저장
        if config['output']['save_model']:
            model_filename = f"xgboost_config{target_config}_{timestamp}.pkl"
            model_path = os.path.join(output_dirs['target_models'], model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            saved_files['model'] = model_path
        
        # 스케일러 저장  
        if config['output']['save_scaler'] and scaler is not None:
            scaler_filename = f"scaler_config{target_config}_{timestamp}.pkl"
            scaler_path = os.path.join(output_dirs['target_models'], scaler_filename)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            saved_files['scaler'] = scaler_path
        
        # 특징명 저장
        if config['output']['save_feature_names']:
            features_filename = f"feature_names_config{target_config}_{timestamp}.json"
            features_path = os.path.join(output_dirs['target_models'], features_filename)
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(feature_names, f, ensure_ascii=False, indent=2)
            saved_files['feature_names'] = features_path
        
        # 성능 지표 저장
        if config['output']['save_metrics'] and metrics:
            metrics_filename = f"metrics_config{target_config}_{timestamp}.json"
            metrics_path = os.path.join(output_dirs['results'], metrics_filename)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            saved_files['metrics'] = metrics_path
        
        print(f"💾 모델 아티팩트 저장 완료:")
        for artifact_type, file_path in saved_files.items():
            print(f"   {artifact_type}: {file_path}")
        
        return saved_files
    
    @staticmethod
    def print_training_banner(step_name: str, description: str):
        """
        학습 단계별 배너 출력 (PipelineUtils 스타일)
        
        Args:
            step_name (str): 단계 이름
            description (str): 단계 설명
        """
        print(f"\n{'='*60}")
        print(f"🤖 {step_name}")
        print(f"📋 {description}")
        print(f"{'='*60}")
    
    @staticmethod
    def print_training_completion(step_name: str, result_info: str = ""):
        """
        학습 단계 완료 배너 출력
        
        Args:
            step_name (str): 단계 이름
            result_info (str): 결과 정보
        """
        print(f"\n✅ {step_name} 완료!")
        if result_info:
            print(f"📊 {result_info}")


def main():
    """테스트 실행"""
    try:
        # 설정 로드 테스트
        config = TrainingUtils.load_training_config()
        print("✅ 학습 설정 로드 테스트 성공")
        
        # 출력 디렉토리 생성 테스트
        output_dirs = TrainingUtils.setup_training_directories(config)
        print("✅ 출력 디렉토리 생성 테스트 성공")
        
        # 특징명 생성 테스트
        feature_names = TrainingUtils.get_feature_names(target_config=1)
        print(f"✅ 특징명 생성 테스트 성공: {len(feature_names)}개 특징")
        
        # 특징 블록 테스트
        blocks = TrainingUtils.get_feature_blocks(target_config=1)
        print(f"✅ 특징 블록 생성 테스트 성공: {len(blocks)}개 블록")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()