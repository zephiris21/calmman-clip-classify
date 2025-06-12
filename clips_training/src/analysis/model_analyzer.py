#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# 머신러닝 라이브러리
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 기존 유틸리티 재사용
from clips_training.src.utils.training_utils import TrainingUtils


class ModelAnalyzer:
    """
    모델 성능 분석 및 시각화 도구
    - 자동 차트 생성 및 저장
    - 노트북 친화적 인터페이스
    - 실행 즉시 결과 확인
    """
    
    def __init__(self, config_path: str = "clips_training/configs/training_config.yaml"):
        """
        모델 분석기 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = TrainingUtils.load_training_config(config_path)
        self.output_dirs = TrainingUtils.setup_training_directories(self.config)
        
        # 시각화 설정
        self.setup_visualization_style()
        
        # 분석 결과 저장용
        self.analysis_results = {}
        
        print("✅ 모델 분석기 초기화 완료")
    
    def setup_visualization_style(self):
        """시각화 스타일 설정"""
        # 한글 폰트 설정 (Windows 환경 대응)
        plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 기본 스타일
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 색상 팔레트 (config에서 가져오기)
        viz_config = self.config.get('visualization', {})
        self.colors = viz_config.get('colors', {
            'funny': '#FF6B6B',
            'boring': '#4ECDC4', 
            'highlight': '#FFD93D',
            'neutral': '#95A5A6'
        })
        
        # 기본 figure 크기
        self.figsize = viz_config.get('figure_size', [12, 8])
        self.dpi = viz_config.get('dpi', 150)
        
        print(f"📊 시각화 스타일 설정 완료 (크기: {self.figsize}, DPI: {self.dpi})")
    
    def load_latest_results(self) -> Tuple[Any, Any, Dict, List[str]]:
        """
        가장 최근 학습 결과 로드
        
        Returns:
            Tuple: (model, scaler, metrics, feature_names)
        """
        print("📂 최신 학습 결과 로드 중...")
        
        target_config = self.config['data']['target_config']
        models_dir = self.output_dirs['target_models']
        results_dir = self.output_dirs['results']
        
        # 모델 파일 찾기 (가장 최근)
        model_files = list(Path(models_dir).glob("xgboost_*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {models_dir}")
        
        latest_model_file = max(model_files, key=os.path.getctime)
        
        # 모델 로드
        with open(latest_model_file, 'rb') as f:
            model = pickle.load(f)
        
        # 스케일러 로드 (있는 경우)
        scaler_files = list(Path(models_dir).glob("scaler_*.pkl"))
        if scaler_files:
            latest_scaler_file = max(scaler_files, key=os.path.getctime)
            with open(latest_scaler_file, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        # 성능 지표 로드
        metrics_files = list(Path(results_dir).glob("metrics_*.json"))
        if metrics_files:
            latest_metrics_file = max(metrics_files, key=os.path.getctime)
            with open(latest_metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        # 특징명 로드
        feature_files = list(Path(models_dir).glob("feature_names_*.json"))
        if feature_files:
            latest_feature_file = max(feature_files, key=os.path.getctime)
            with open(latest_feature_file, 'r', encoding='utf-8') as f:
                feature_names = json.load(f)
        else:
            feature_names = TrainingUtils.get_feature_names(target_config)
        
        print(f"✅ 결과 로드 완료:")
        print(f"   모델: {latest_model_file.name}")
        print(f"   성능 지표: {latest_metrics_file.name if metrics_files else '기본값'}")
        print(f"   특징 수: {len(feature_names)}")
        
        return model, scaler, metrics, feature_names
    
    def create_results_directory(self) -> str:
        """분석 결과 저장용 디렉토리 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_config = self.config['data']['target_config']
        
        analysis_dir = os.path.join(
            self.output_dirs['results'], 
            f"analysis_config{target_config}_{timestamp}"
        )
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"📁 분석 결과 저장 디렉토리: {analysis_dir}")
        return analysis_dir
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str) -> plt.Figure:
        """
        혼동행렬 히트맵 생성
        
        Args:
            y_true: 실제 라벨
            y_pred: 예측 라벨
            save_path: 저장 경로
            
        Returns:
            matplotlib.Figure: 생성된 figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # 혼동행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        
        # 히트맵 생성
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Boring', 'Funny'],
                    yticklabels=['Boring', 'Funny'],
                    ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # 정확도 표시
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 혼동행렬 저장: {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, 
                      save_path: str) -> plt.Figure:
        """
        ROC 곡선 생성
        
        Args:
            y_true: 실제 라벨
            y_scores: 예측 확률
            save_path: 저장 경로
            
        Returns:
            matplotlib.Figure: 생성된 figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        
        # ROC 곡선 계산
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # ROC 곡선 그리기
        ax.plot(fpr, tpr, color=self.colors['funny'], lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color=self.colors['neutral'], 
                lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 ROC 곡선 저장: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              save_path: str, top_n: int = 20) -> plt.Figure:
        """
        피처 중요도 차트 생성
        
        Args:
            model: 학습된 모델
            feature_names: 특징명 리스트
            save_path: 저장 경로
            top_n: 표시할 상위 특징 수
            
        Returns:
            matplotlib.Figure: 생성된 figure
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        # 피처 중요도 추출 및 정렬
        importance_scores = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # 수평 막대 그래프
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                      color=self.colors['highlight'], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 특징명 설정 (간소화)
        simplified_names = []
        for name in importance_df['feature']:
            # 긴 특징명 간소화
            if len(name) > 25:
                parts = name.split('_')
                simplified = f"{parts[0]}_{parts[-2]}_{parts[-1]}"
                simplified_names.append(simplified)
            else:
                simplified_names.append(name)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(simplified_names, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (XGBoost)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(value + 0.001, i, f'{value:.3f}', 
                   va='center', ha='left', fontsize=9)
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 피처 중요도 저장: {save_path}")
        
        return fig
    
    def plot_performance_dashboard(self, metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> plt.Figure:
        """
        성능 지표 대시보드 생성 (개선된 버전)
        
        Args:
            metrics: 성능 지표 딕셔너리
            y_true: 실제 라벨
            y_pred: 예측 라벨
            save_path: 저장 경로
            
        Returns:
            matplotlib.Figure: 생성된 figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # CV vs Test 성능 비교 (Accuracy만 의미있는 비교)
        test_metrics = metrics.get('test_performance', {})
        cv_metrics = metrics.get('cross_validation', {})
        
        # 1. CV vs Test Accuracy 비교 (간단하고 명확하게)
        cv_acc = cv_metrics.get('cv_mean', 0)
        test_acc = test_metrics.get('test_accuracy', 0)
        
        comparison_data = ['CV Accuracy', 'Test Accuracy']
        comparison_scores = [cv_acc, test_acc]
        colors_comp = [self.colors['boring'], self.colors['funny']]
        
        bars1 = ax1.bar(comparison_data, comparison_scores, 
                       color=colors_comp, alpha=0.8, edgecolor='black', linewidth=1)
        
        # 값 표시
        for bar, score in zip(bars1, comparison_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax1.set_ylabel('Accuracy Score')
        ax1.set_title('CV vs Test Accuracy Comparison')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3)
        
        # 과적합 여부 표시
        if test_acc > cv_acc:
            ax1.text(0.5, 0.1, '✅ No Overfitting\n(Test > CV)', 
                    ha='center', va='center', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    fontsize=10, fontweight='bold')
        else:
            diff = cv_acc - test_acc
            if diff > 0.05:
                ax1.text(0.5, 0.1, '⚠️ Possible Overfitting', 
                        ha='center', va='center', transform=ax1.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                        fontsize=10, fontweight='bold')
        
        # 2. 클래스별 정밀한 성능 분석 (혼동행렬 기반)
        from sklearn.metrics import classification_report
        
        # 분류 보고서에서 클래스별 지표 추출
        report = classification_report(y_true, y_pred, target_names=['Boring', 'Funny'], output_dict=True)
        
        classes = ['Boring', 'Funny']
        precision_scores = [report['Boring']['precision'], report['Funny']['precision']]
        recall_scores = [report['Boring']['recall'], report['Funny']['recall']]
        f1_scores = [report['Boring']['f1-score'], report['Funny']['f1-score']]
        
        x_classes = np.arange(len(classes))
        width = 0.25
        
        bars2_1 = ax2.bar(x_classes - width, precision_scores, width, 
                         label='Precision', color=self.colors['highlight'], alpha=0.8)
        bars2_2 = ax2.bar(x_classes, recall_scores, width, 
                         label='Recall', color=self.colors['neutral'], alpha=0.8)
        bars2_3 = ax2.bar(x_classes + width, f1_scores, width, 
                         label='F1-Score', color=self.colors['funny'], alpha=0.8)
        
        # 값 표시
        for bars in [bars2_1, bars2_2, bars2_3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_title('Class-wise Performance Metrics')
        ax2.set_xticks(x_classes)
        ax2.set_xticklabels(classes)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # 3. 테스트 세트 전체 성능 지표 시각화
        all_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        all_scores = [
            test_metrics.get('test_accuracy', 0),
            test_metrics.get('test_precision', 0), 
            test_metrics.get('test_recall', 0),
            test_metrics.get('test_f1', 0),
            test_metrics.get('test_roc_auc', 0)
        ]
        
        bars3 = ax3.bar(all_metrics, all_scores, 
                       color=[self.colors['funny'], self.colors['highlight'], 
                             self.colors['neutral'], self.colors['boring'], 
                             self.colors['funny']], alpha=0.8)
        
        # 값 표시
        for bar, score in zip(bars3, all_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('Score')
        ax3.set_title('Test Set Performance Metrics')
        ax3.set_xticklabels(all_metrics, rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)
        
        # 4. 모델 정보 및 혼동행렬 요약
        ax4.axis('off')
        
        # 혼동행렬 계산
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        info_text = f"""
Model Configuration Summary

Algorithm: XGBoost
Top Features: {len(metrics.get('test_performance', {}).get('top_features', {}).get('names', []))}
Best Feature: {metrics.get('test_performance', {}).get('top_features', {}).get('names', ['N/A'])[0] if metrics.get('test_performance', {}).get('top_features', {}).get('names') else 'N/A'}

Performance Highlights:
✓ Test Accuracy: {test_metrics.get('test_accuracy', 0):.3f}
✓ F1-Score: {test_metrics.get('test_f1', 0):.3f}
✓ ROC-AUC: {test_metrics.get('test_roc_auc', 0):.3f}

Confusion Matrix:
┌─────────────┬─────────────┐
│ TN: {tn:2d}     │ FP: {fp:2d}     │
├─────────────┼─────────────┤
│ FN: {fn:2d}     │ TP: {tp:2d}     │
└─────────────┴─────────────┘

Training Info:
• CV Accuracy: {cv_metrics.get('cv_mean', 0):.3f}
• CV Folds: {metrics.get('model_config', {}).get('cv_folds', 5)}
• Total Time: {cv_metrics.get('cv_time', 0) + cv_metrics.get('final_training_time', 0):.1f}s

Interpretation:
• Precision (Funny): {report['Funny']['precision']:.3f}
• Recall (Funny): {report['Funny']['recall']:.3f}
• No False Positives: {fp == 0}
        """
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Model Performance Dashboard (Enhanced)', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"💾 성능 대시보드 저장: {save_path}")
        
        return fig
    
    def analyze_and_save_all(self, auto_close: bool = True) -> Dict[str, str]:
        """
        전체 분석 실행 및 모든 차트 저장
        
        Args:
            auto_close: 차트 자동 닫기 여부
            
        Returns:
            Dict: 저장된 파일 경로들
        """
        print("\n🎨 모델 성능 분석 및 시각화 시작")
        print("=" * 50)
        
        try:
            # 1. 결과 로드
            model, scaler, metrics, feature_names = self.load_latest_results()
            
            # 2. 저장 디렉토리 생성
            analysis_dir = self.create_results_directory()
            
            # 3. 테스트 데이터 다시 로드 (예측값 계산용)
            print("📊 테스트 데이터 로드 중...")
            dataset_path = self.config['data']['dataset_path']
            target_config = self.config['data']['target_config']
            
            X, y, _, _ = TrainingUtils.load_dataset_hdf5(dataset_path, target_config)
            
            # 데이터 분할 (학습할 때와 동일한 설정)
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X, y,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state'],
                stratify=y if self.config['data']['stratify'] else None
            )
            
            # 스케일링 (필요한 경우)
            if scaler is not None:
                X_test = scaler.transform(X_test)
            
            # 예측 수행
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]
            
            saved_files = {}
            
            # 4. 혼동행렬
            print("📈 혼동행렬 생성 중...")
            cm_path = os.path.join(analysis_dir, "confusion_matrix.png")
            fig1 = self.plot_confusion_matrix(y_test, y_pred, cm_path)
            saved_files['confusion_matrix'] = cm_path
            if auto_close: plt.close(fig1)
            
            # 5. ROC 곡선
            print("📈 ROC 곡선 생성 중...")
            roc_path = os.path.join(analysis_dir, "roc_curve.png")
            fig2 = self.plot_roc_curve(y_test, y_scores, roc_path)
            saved_files['roc_curve'] = roc_path
            if auto_close: plt.close(fig2)
            
            # 6. 피처 중요도
            print("📈 피처 중요도 차트 생성 중...")
            feature_path = os.path.join(analysis_dir, "feature_importance.png")
            top_n = self.config.get('evaluation', {}).get('feature_importance', {}).get('top_n', 20)
            fig3 = self.plot_feature_importance(model, feature_names, feature_path, top_n)
            saved_files['feature_importance'] = feature_path
            if auto_close: plt.close(fig3)
            
            # 7. 성능 대시보드 (개선된 버전 - y_true, y_pred 전달)
            print("📈 성능 대시보드 생성 중...")
            dashboard_path = os.path.join(analysis_dir, "performance_dashboard.png")
            fig4 = self.plot_performance_dashboard(metrics, y_test, y_pred, dashboard_path)
            saved_files['performance_dashboard'] = dashboard_path
            if auto_close: plt.close(fig4)
            
            # 8. 분석 요약 저장
            summary_path = os.path.join(analysis_dir, "analysis_summary.json")
            summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_type': 'XGBoost',
                'test_accuracy': float(metrics.get('test_performance', {}).get('test_accuracy', 0)),
                'test_f1': float(metrics.get('test_performance', {}).get('test_f1', 0)),
                'test_roc_auc': float(metrics.get('test_performance', {}).get('test_roc_auc', 0)),
                'top_5_features': metrics.get('test_performance', {}).get('top_features', {}).get('names', [])[:5],
                'saved_files': saved_files,
                'config_used': self.config['data']['target_config']
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            saved_files['summary'] = summary_path
            
            print("\n✅ 분석 완료!")
            print(f"📁 결과 저장 위치: {analysis_dir}")
            print(f"📊 생성된 차트: {len(saved_files)-1}개")
            print(f"🏆 최고 성능: Test Accuracy {summary['test_accuracy']:.3f}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return {}


def quick_analysis():
    """노트북용 빠른 분석 함수"""
    analyzer = ModelAnalyzer()
    return analyzer.analyze_and_save_all(auto_close=False)


def main():
    """메인 실행 함수"""
    print("🎨 모델 성능 분석기 실행")
    
    analyzer = ModelAnalyzer()
    saved_files = analyzer.analyze_and_save_all(auto_close=True)
    
    if saved_files:
        print(f"\n🎉 분석 완료! {len(saved_files)}개 파일 생성")
        for file_type, file_path in saved_files.items():
            print(f"   {file_type}: {file_path}")
    else:
        print("❌ 분석 실패")


if __name__ == "__main__":
    main()