#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Any
from datetime import datetime

# 머신러닝 라이브러리
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 기존 유틸리티 재사용
from clips_training.src.utils.training_utils import TrainingUtils


class XGBoostTrainer:
    """
    XGBoost 베이스라인 학습기
    - 빠른 베이스라인 구현
    - 5-fold CV 검증
    - 피처 중요도 분석
    """
    
    def __init__(self, config_path: str = "clips_training/configs/training_config.yaml"):
        """
        XGBoost 학습기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 및 디렉토리 설정
        self.config = TrainingUtils.load_training_config(config_path)
        self.output_dirs = TrainingUtils.setup_training_directories(self.config)
        self.logger = TrainingUtils.setup_training_logging(self.config, self.output_dirs)
        
        # 모델 및 데이터 초기화
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.logger.info("✅ XGBoost 학습기 초기화 완료")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        데이터셋 로드 및 전처리
        
        Returns:
            Tuple: (X, y, clip_ids)
        """
        TrainingUtils.print_training_banner("데이터 로드", "dataset.h5에서 학습 데이터 로드")
        
        # 데이터 로드
        dataset_path = self.config['data']['dataset_path']
        target_config = self.config['data']['target_config']
        
        X, y, clip_ids, metadata = TrainingUtils.load_dataset_hdf5(dataset_path, target_config)
        
        # 특징명 생성
        self.feature_names = TrainingUtils.get_feature_names(target_config)
        
        # 데이터 분석
        analysis = TrainingUtils.analyze_dataset_info(X, y, clip_ids, self.feature_names, self.logger)
        
        TrainingUtils.print_training_completion("데이터 로드", 
                                              f"샘플: {len(X)}, 특징: {X.shape[1]}, 균형: {analysis['basic_info']['is_balanced']}")
        
        return X, y, clip_ids
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        데이터 분할 (train/test)
        
        Args:
            X: 특징 배열
            y: 라벨 배열
        """
        TrainingUtils.print_training_banner("데이터 분할", "학습/테스트 셋 분할 및 전처리")
        
        # 설정값 읽기
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        stratify = y if self.config['data']['stratify'] else None
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # 스케일링 (설정에 따라)
        scaler_type = self.config['preprocessing']['scaler']
        if scaler_type != 'none':
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                self.logger.warning(f"지원하지 않는 스케일러: {scaler_type}, StandardScaler 사용")
                self.scaler = StandardScaler()
            
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            self.logger.info(f"✅ 데이터 스케일링 완료: {scaler_type}")
        
        # 분할 결과 로깅
        train_class_counts = np.bincount(self.y_train)
        test_class_counts = np.bincount(self.y_test)
        
        self.logger.info(f"📊 데이터 분할 완료:")
        self.logger.info(f"   학습셋: {len(self.X_train)}개 (Boring: {train_class_counts[0]}, Funny: {train_class_counts[1]})")
        self.logger.info(f"   테스트셋: {len(self.X_test)}개 (Boring: {test_class_counts[0]}, Funny: {test_class_counts[1]})")
        
        TrainingUtils.print_training_completion("데이터 분할", 
                                              f"학습: {len(self.X_train)}, 테스트: {len(self.X_test)}")
    
    def train_model(self) -> Dict[str, float]:
        """
        XGBoost 모델 학습 및 교차 검증
        
        Returns:
            Dict: 교차 검증 결과
        """
        TrainingUtils.print_training_banner("모델 학습", "XGBoost 학습 및 5-fold 교차 검증")
        
        # XGBoost 파라미터 설정
        xgb_params = self.config['model']['xgboost'].copy()
        xgb_params['random_state'] = self.config['model']['random_state']
        
        # 모델 초기화
        self.model = xgb.XGBClassifier(**xgb_params)
        
        # 교차 검증 설정
        cv_folds = self.config['model']['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['model']['random_state'])
        
        # 교차 검증 실행
        self.logger.info(f"🔄 {cv_folds}-fold 교차 검증 시작...")
        start_time = time.time()
        
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                  cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_time = time.time() - start_time
        
        # CV 결과 정리
        cv_results = {
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist(),
            'cv_time': cv_time
        }
        
        # 최종 모델 학습
        self.logger.info("🎯 전체 학습 데이터로 최종 모델 학습...")
        final_start_time = time.time()
        
        self.model.fit(self.X_train, self.y_train)
        
        final_time = time.time() - final_start_time
        cv_results['final_training_time'] = final_time
        
        # 결과 로깅
        self.logger.info(f"✅ 교차 검증 완료:")
        self.logger.info(f"   평균 정확도: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")
        self.logger.info(f"   개별 점수: {[f'{score:.4f}' for score in cv_scores]}")
        self.logger.info(f"   CV 시간: {cv_time:.2f}초")
        self.logger.info(f"   최종 학습 시간: {final_time:.2f}초")
        
        TrainingUtils.print_training_completion("모델 학습", 
                                              f"CV 정확도: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")
        
        return cv_results
    
    def evaluate_model(self) -> Dict[str, Any]:
        """
        테스트 셋으로 모델 성능 평가
        
        Returns:
            Dict: 평가 결과
        """
        TrainingUtils.print_training_banner("모델 평가", "테스트 셋 성능 평가 및 피처 중요도 분석")
        
        # 예측 수행
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # 성능 지표 계산
        metrics = {
            'test_accuracy': float(accuracy_score(self.y_test, y_pred)),
            'test_precision': float(precision_score(self.y_test, y_pred)),
            'test_recall': float(recall_score(self.y_test, y_pred)),
            'test_f1': float(f1_score(self.y_test, y_pred)),
            'test_roc_auc': float(roc_auc_score(self.y_test, y_pred_proba))
        }
        
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 피처 중요도 (상위 20개)
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(20)
        metrics['top_features'] = {
            'names': top_features['feature'].tolist(),
            'scores': top_features['importance'].tolist()
        }
        
        # 결과 로깅
        self.logger.info(f"📊 테스트 셋 성능 평가:")
        self.logger.info(f"   정확도: {metrics['test_accuracy']:.4f}")
        self.logger.info(f"   정밀도: {metrics['test_precision']:.4f}")
        self.logger.info(f"   재현율: {metrics['test_recall']:.4f}")
        self.logger.info(f"   F1-score: {metrics['test_f1']:.4f}")
        self.logger.info(f"   ROC-AUC: {metrics['test_roc_auc']:.4f}")
        
        self.logger.info(f"🎯 상위 5개 중요 특징:")
        for i, (name, score) in enumerate(zip(top_features['feature'][:5], top_features['importance'][:5])):
            self.logger.info(f"   {i+1}. {name}: {score:.4f}")
        
        # 분류 보고서
        self.logger.info(f"📋 분류 보고서:")
        report = classification_report(self.y_test, y_pred, target_names=['Boring', 'Funny'])
        for line in report.split('\n'):
            if line.strip():
                self.logger.info(f"   {line}")
        
        TrainingUtils.print_training_completion("모델 평가", 
                                              f"정확도: {metrics['test_accuracy']:.4f}, F1: {metrics['test_f1']:.4f}")
        
        return metrics
    
    def save_results(self, cv_results: Dict, test_metrics: Dict) -> Dict[str, str]:
        """
        모델 및 결과 저장
        
        Args:
            cv_results: 교차 검증 결과
            test_metrics: 테스트 성능 지표
            
        Returns:
            Dict: 저장된 파일 경로들
        """
        TrainingUtils.print_training_banner("결과 저장", "모델 및 성능 지표 저장")
        
        # 전체 결과 통합
        all_metrics = {
            'cross_validation': cv_results,
            'test_performance': test_metrics,
            'model_config': self.config['model'],
            'data_config': self.config['data'],
            'timestamp': datetime.now().isoformat()
        }
        
        # 모델 저장
        saved_files = TrainingUtils.save_model_artifacts(
            model=self.model,
            scaler=self.scaler,
            feature_names=self.feature_names,
            config=self.config,
            output_dirs=self.output_dirs,
            metrics=all_metrics
        )
        
        TrainingUtils.print_training_completion("결과 저장", f"{len(saved_files)}개 파일 저장 완료")
        
        return saved_files
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        전체 학습 파이프라인 실행
        
        Returns:
            Dict: 전체 실행 결과
        """
        pipeline_start_time = time.time()
        
        try:
            # 1. 데이터 로드
            X, y, clip_ids = self.load_data()
            
            # 2. 데이터 분할
            self.split_data(X, y)
            
            # 3. 모델 학습
            cv_results = self.train_model()
            
            # 4. 모델 평가
            test_metrics = self.evaluate_model()
            
            # 5. 결과 저장
            saved_files = self.save_results(cv_results, test_metrics)
            
            # 전체 파이프라인 완료
            total_time = time.time() - pipeline_start_time
            
            final_results = {
                'success': True,
                'total_time': total_time,
                'cv_results': cv_results,
                'test_metrics': test_metrics,
                'saved_files': saved_files,
                'summary': {
                    'cv_accuracy': cv_results['cv_mean'],
                    'test_accuracy': test_metrics['test_accuracy'],
                    'test_f1': test_metrics['test_f1'],
                    'top_feature': test_metrics['top_features']['names'][0]
                }
            }
            
            self.logger.info(f"\n🎉 전체 파이프라인 완료! (총 {total_time:.1f}초)")
            self.logger.info(f"📊 최종 성능 요약:")
            self.logger.info(f"   CV 정확도: {cv_results['cv_mean']:.4f}")
            self.logger.info(f"   테스트 정확도: {test_metrics['test_accuracy']:.4f}")
            self.logger.info(f"   F1-Score: {test_metrics['test_f1']:.4f}")
            self.logger.info(f"   가장 중요한 특징: {test_metrics['top_features']['names'][0]}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 파이프라인 실행 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - pipeline_start_time
            }


def main():
    """테스트 실행"""
    print("🚀 XGBoost 베이스라인 학습기 테스트")
    
    try:
        # 학습기 초기화
        trainer = XGBoostTrainer()
        
        # 전체 파이프라인 실행
        results = trainer.run_full_pipeline()
        
        if results['success']:
            print("\n✅ 베이스라인 학습 성공!")
            print(f"📊 CV 정확도: {results['cv_results']['cv_mean']:.4f}")
            print(f"📊 테스트 정확도: {results['test_metrics']['test_accuracy']:.4f}")
            print(f"🏆 가장 중요한 특징: {results['test_metrics']['top_features']['names'][0]}")
        else:
            print(f"❌ 학습 실패: {results['error']}")
            
    except Exception as e:
        print(f"❌ 실행 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()