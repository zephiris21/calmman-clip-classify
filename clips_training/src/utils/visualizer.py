import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 🎯 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ClipsVisualizer:
    """
    침착맨 클립 데이터 시각화 클래스
    - 한글 폰트 설정 자동화
    - 제목 간격 자동 조정
    - 일관된 색상 팔레트
    """
    
    def __init__(self, figsize=(12, 8), style='whitegrid'):
        """
        시각화 도구 초기화
        
        Args:
            figsize: 기본 그래프 크기
            style: seaborn 스타일
        """
        self.figsize = figsize
        self.colors = {
            'boring': '#4ECDC4',     # 청록색
            'funny': '#FF6B6B',      # 빨간색
            'highlight': '#FFD93D',   # 노란색
            'neutral': '#95A5A6'      # 회색
        }
        
        # 스타일 설정 (폰트 설정 제외)
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = 100
        
        print("✅ ClipsVisualizer 초기화 완료!")
        print(f"   기본 크기: {figsize}")
        print(f"   색상 팔레트: {len(self.colors)}개")
        print("💡 한글 폰트는 노트북에서 직접 설정하세요!")
    
        print("✅ 한글 폰트 설정 완료")
    
    def plot_class_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        클래스 분포 시각화 (파이차트 + 막대그래프)
        
        Args:
            df: 데이터프레임 (label_name 컬럼 필요)
            save_path: 저장 경로 (선택적)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        class_counts = df['label_name'].value_counts()
        colors = [self.colors['boring'], self.colors['funny']]
        
        # 파이차트
        class_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', 
                          colors=colors, startangle=90)
        axes[0].set_title('클래스 분포 (파이차트)', fontsize=14, fontweight='bold', pad=20)
        axes[0].set_ylabel('')
        
        # 막대그래프
        class_counts.plot(kind='bar', ax=axes[1], color=colors, alpha=0.8)
        axes[1].set_title('클래스 분포 (막대그래프)', fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('클래스')
        axes[1].set_ylabel('샘플 수')
        axes[1].tick_params(axis='x', rotation=0)
        
        # 막대 위에 숫자 표시
        for i, v in enumerate(class_counts.values):
            axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('침착맨 클립 재미도 분포', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_key_features_boxplot(self, df: pd.DataFrame, key_features: List[str], 
                                 save_path: Optional[str] = None) -> None:
        """
        핵심 특징별 박스플롯
        
        Args:
            df: 데이터프레임
            key_features: 핵심 특징 리스트
            save_path: 저장 경로 (선택적)
        """
        # 상위 12개만 시각화
        n_features = min(12, len(key_features))
        selected_features = key_features[:n_features]
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(selected_features):
            if feature in df.columns:
                sns.boxplot(data=df, x='label_name', y=feature, ax=axes[i], 
                           palette=[self.colors['boring'], self.colors['funny']])
                axes[i].set_title(f'{feature}', fontsize=11, pad=15)
                axes[i].set_xlabel('')
                axes[i].tick_params(axis='x', rotation=0)
            else:
                axes[i].text(0.5, 0.5, f'{feature}\n(특징 없음)', 
                            ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # 빈 서브플롯 숨기기
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('핵심 특징별 클래스 분포 (박스플롯)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, features: List[str], 
                                save_path: Optional[str] = None) -> None:
        """
        상관관계 히트맵
        
        Args:
            df: 데이터프레임
            features: 분석할 특징 리스트
            save_path: 저장 경로 (선택적)
        """
        if len(features) == 0:
            print("⚠️ 분석할 특징이 없습니다")
            return
        
        # 상관관계 계산
        corr_matrix = df[features].corr()
        
        # 히트맵 그리기
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 상삼각 마스크
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    square=True,
                    cbar_kws={'shrink': 0.8})
        
        plt.title('특징 간 상관관계 히트맵', fontsize=16, fontweight='bold', pad=30)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return corr_matrix
    
    def plot_distribution_comparison(self, df: pd.DataFrame, features: List[str], 
                                   save_path: Optional[str] = None) -> None:
        """
        클래스별 분포 히스토그램 비교
        
        Args:
            df: 데이터프레임
            features: 분석할 특징 리스트 (최대 6개)
            save_path: 저장 경로 (선택적)
        """
        n_features = min(6, len(features))
        selected_features = features[:n_features]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(selected_features):
            if feature in df.columns:
                # 클래스별 데이터 분리
                boring_data = df[df['label_name'] == 'boring'][feature].dropna()
                funny_data = df[df['label_name'] == 'funny'][feature].dropna()
                
                # 히스토그램 그리기
                axes[i].hist(boring_data, alpha=0.7, label='Boring', 
                           color=self.colors['boring'], bins=20, density=True)
                axes[i].hist(funny_data, alpha=0.7, label='Funny', 
                           color=self.colors['funny'], bins=20, density=True)
                
                axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold', pad=15)
                axes[i].set_xlabel('값')
                axes[i].set_ylabel('밀도')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 빈 서브플롯 숨기기
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('상위 유의미 특징들의 분포 비교', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def statistical_analysis(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        특징별 통계적 유의성 검증 (t-test)
        
        Args:
            df: 데이터프레임
            features: 분석할 특징 리스트
            
        Returns:
            pd.DataFrame: 통계 분석 결과
        """
        statistical_results = []
        
        for feature in features:
            if feature in df.columns:
                boring_data = df[df['label_name'] == 'boring'][feature].dropna()
                funny_data = df[df['label_name'] == 'funny'][feature].dropna()
                
                if len(boring_data) > 1 and len(funny_data) > 1:
                    # t-test 수행
                    t_stat, p_value = stats.ttest_ind(boring_data, funny_data)
                    
                    # 효과 크기 (Cohen's d) 계산
                    pooled_std = np.sqrt(((len(boring_data) - 1) * np.var(boring_data, ddof=1) + 
                                         (len(funny_data) - 1) * np.var(funny_data, ddof=1)) / 
                                        (len(boring_data) + len(funny_data) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (np.mean(funny_data) - np.mean(boring_data)) / pooled_std
                    else:
                        cohens_d = 0
                    
                    statistical_results.append({
                        'feature': feature,
                        'boring_mean': np.mean(boring_data),
                        'funny_mean': np.mean(funny_data),
                        'mean_diff': np.mean(funny_data) - np.mean(boring_data),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })
        
        # 결과를 DataFrame으로 정리
        stats_df = pd.DataFrame(statistical_results)
        if len(stats_df) > 0:
            stats_df = stats_df.sort_values('p_value')
        
        return stats_df
    
    def print_statistical_results(self, stats_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        통계 분석 결과 출력
        
        Args:
            stats_df: 통계 분석 결과 DataFrame
            top_n: 출력할 상위 특징 수
        """
        if len(stats_df) == 0:
            print("⚠️ 통계 분석 결과가 없습니다")
            return
        
        print("📊 클래스 간 차이 통계 검증 결과 (p-value 순)")
        print("=" * 80)
        
        # 컬럼 헤더
        cohens_d_header = "Cohen's d"
        print(f"{'특징명':<30} {'Boring평균':<10} {'Funny평균':<10} {'차이':<8} {'p-value':<10} {cohens_d_header:<10} {'유의'}")
        print("-" * 80)
        
        for _, row in stats_df.head(top_n).iterrows():
            significance = "✅" if row['significant'] else "❌"
            print(f"{row['feature']:<30} {row['boring_mean']:<10.3f} {row['funny_mean']:<10.3f} "
                  f"{row['mean_diff']:<8.3f} {row['p_value']:<10.3f} {row['cohens_d']:<10.3f} {significance}")
    
    def find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """
        높은 상관관계 특징 쌍 찾기
        
        Args:
            corr_matrix: 상관관계 매트릭스
            threshold: 상관관계 임계값
            
        Returns:
            List[Dict]: 높은 상관관계 특징 쌍들
        """
        high_corr_pairs = []
        features = corr_matrix.columns
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': features[i],
                        'feature2': features[j],
                        'correlation': corr_val
                    })
        
        return high_corr_pairs
    
    def generate_insights_summary(self, df: pd.DataFrame, stats_df: pd.DataFrame, 
                                 high_corr_pairs: List[Dict]) -> Dict:
        """
        EDA 인사이트 요약 생성
        
        Args:
            df: 데이터프레임
            stats_df: 통계 분석 결과
            high_corr_pairs: 높은 상관관계 쌍들
            
        Returns:
            Dict: 인사이트 요약
        """
        # 클래스 균형성
        funny_ratio = df['label_name'].value_counts().get('funny', 0) / len(df)
        is_balanced = 0.4 <= funny_ratio <= 0.6
        
        # 유의미한 특징 수
        significant_features = stats_df[stats_df['significant'] == True] if len(stats_df) > 0 else pd.DataFrame()
        
        # 데이터 품질
        nan_count = df.isnull().sum().sum()
        
        insights = {
            'data_balance': {
                'funny_ratio': funny_ratio,
                'is_balanced': is_balanced,
                'status': "균형잡힌" if is_balanced else "불균형"
            },
            'feature_significance': {
                'total_features': len(stats_df),
                'significant_count': len(significant_features),
                'significance_ratio': len(significant_features) / len(stats_df) if len(stats_df) > 0 else 0
            },
            'multicollinearity': {
                'high_corr_count': len(high_corr_pairs),
                'risk_level': "낮음" if len(high_corr_pairs) == 0 else f"주의 ({len(high_corr_pairs)}개 쌍)"
            },
            'data_quality': {
                'nan_count': nan_count,
                'quality_score': 100 - (nan_count / df.size * 100) if df.size > 0 else 100,
                'status': "좋음" if nan_count == 0 else f"결측값 {nan_count}개"
            }
        }
        
        return insights
    
    def print_insights_summary(self, insights: Dict, stats_df: pd.DataFrame) -> None:
        """
        인사이트 요약 출력
        
        Args:
            insights: 인사이트 딕셔너리
            stats_df: 통계 분석 결과
        """
        print("🎯 EDA 핵심 인사이트 요약")
        print("=" * 60)
        
        # 1. 데이터 균형성
        balance = insights['data_balance']
        print(f"1️⃣ 데이터 균형성: {balance['status']} (Funny: {balance['funny_ratio']:.1%})")
        
        # 2. 유의미한 특징 수
        significance = insights['feature_significance']
        if significance['total_features'] > 0:
            print(f"2️⃣ 통계적 유의미한 특징: {significance['significant_count']}/{significance['total_features']} "
                  f"({significance['significance_ratio']*100:.1f}%)")
            
            # 3. 가장 차별적인 특징 top 3
            significant_features = stats_df[stats_df['significant'] == True]
            if len(significant_features) > 0:
                print(f"3️⃣ 가장 차별적인 특징 Top 3:")
                for i, (_, row) in enumerate(significant_features.head(3).iterrows()):
                    direction = "Funny > Boring" if row['mean_diff'] > 0 else "Boring > Funny"
                    print(f"   {i+1}. {row['feature']}: p={row['p_value']:.3f}, d={row['cohens_d']:.2f} ({direction})")
        
        # 4. 다중공선성 위험
        multicollinearity = insights['multicollinearity']
        print(f"4️⃣ 다중공선성 위험: {multicollinearity['risk_level']}")
        
        # 5. 데이터 품질
        quality = insights['data_quality']
        print(f"5️⃣ 데이터 품질: {quality['status']}")
        
        # 6. 권장사항
        print(f"\n💡 학습 모델 권장사항:")
        print(f"   - XGBoost/RandomForest 등 트리 기반 모델 적합")
        
        if significance['total_features'] > 0:
            feature_selection_needed = significance['significance_ratio'] < 0.7
            print(f"   - 특징 선택 필요성: {'높음' if feature_selection_needed else '낮음'}")
        
        class_weight_needed = not balance['is_balanced']
        print(f"   - 클래스 가중치 조정: {'필요' if class_weight_needed else '불필요'}")


def main():
    """테스트 실행"""
    print("🧪 ClipsVisualizer 테스트 시작")
    
    # 가상 데이터로 테스트
    np.random.seed(42)
    n_samples = 100
    
    test_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(1, 1.5, n_samples),
        'label': np.random.choice([0, 1], n_samples),
    })
    test_df['label_name'] = test_df['label'].map({0: 'boring', 1: 'funny'})
    
    # 시각화 도구 초기화
    visualizer = ClipsVisualizer()
    
    # 클래스 분포 테스트
    visualizer.plot_class_distribution(test_df)
    
    print("✅ ClipsVisualizer 테스트 성공!")


if __name__ == "__main__":
    main()