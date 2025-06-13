#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from sklearn.cluster import DBSCAN
import logging

# 프로젝트 루트 찾기
current_file = Path(__file__)
project_root = current_file.parent.parent.parent  # pipeline/modules/highlight_clusterer.py
sys.path.insert(0, str(project_root))

# 파이프라인 유틸리티 import
sys.path.append('pipeline')
from utils.pipeline_utils import PipelineUtils


class HighlightClusterer:
    """
    텐션 하이라이트 클러스터링 시스템
    - DBSCAN으로 밀집 구간 탐지
    - 단일 포인트 클러스터 확장 (±3초)
    - 윈도우 생성을 위한 클러스터 범위 계산
    """
    
    def __init__(self, config_path: str = None):
        """
        하이라이트 클러스터링 초기화
        
        Args:
            config_path (str): config 파일 경로 (기본: funclip_extraction_config.yaml)
        """
        # 프로젝트 루트로 작업 디렉토리 변경
        os.chdir(project_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Config 로드
        if config_path is None:
            config_path = "pipeline/configs/funclip_extraction_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 클러스터링 파라미터 설정
        clustering_config = config['clustering']
        self.eps = clustering_config['eps']  # 9초
        self.min_samples = clustering_config['min_samples']  # 1
        self.expansion_buffer = clustering_config['expansion_buffer']  # 3초
        
        self.logger.info(f"✅ 하이라이트 클러스터링 초기화 완료")
        self.logger.info(f"   DBSCAN 파라미터: eps={self.eps}초, min_samples={self.min_samples}")
        self.logger.info(f"   단일 클러스터 확장: ±{self.expansion_buffer}초")
    
    def load_tension_highlights(self, tension_json_path: str) -> List[Dict]:
        """
        텐션 분석 JSON에서 하이라이트 로드
        
        Args:
            tension_json_path (str): 텐션 JSON 파일 경로
            
        Returns:
            List[Dict]: 하이라이트 리스트
        """
        try:
            with open(tension_json_path, 'r', encoding='utf-8') as f:
                tension_data = json.load(f)
            
            highlights = tension_data['edit_suggestions']['highlights']
            
            self.logger.info(f"📊 하이라이트 로드 완료: {len(highlights)}개")
            self.logger.info(f"   소스: {os.path.basename(tension_json_path)}")
            
            # 시간 범위 정보
            if highlights:
                timestamps = [h['timestamp'] for h in highlights]
                self.logger.info(f"   시간 범위: {min(timestamps):.1f}초 ~ {max(timestamps):.1f}초")
            
            return highlights
            
        except Exception as e:
            self.logger.error(f"❌ 하이라이트 로드 실패: {e}")
            raise
    
    def cluster_with_dbscan(self, highlights: List[Dict]) -> List[Dict]:
        """
        DBSCAN으로 하이라이트 클러스터링 (원본 포인트만 사용)
        
        Args:
            highlights (List[Dict]): 하이라이트 리스트
            
        Returns:
            List[Dict]: 클러스터 리스트
        """
        if not highlights:
            self.logger.warning("⚠️ 하이라이트가 없어 빈 클러스터 반환")
            return []
        
        self.logger.info("🔍 DBSCAN 클러스터링 시작...")
        
        # 타임스탬프 추출 (1차원 배열로 변환)
        timestamps = np.array([[h['timestamp']] for h in highlights])
        
        # DBSCAN 클러스터링
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(timestamps)
        
        # 클러스터별 그룹화
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': i,
                'highlight': highlights[i]
            })
        
        # 클러스터 리스트 생성
        cluster_list = []
        cluster_id = 0
        
        for label, points in clusters.items():
            # 노이즈 포인트(-1)도 개별 클러스터로 처리
            cluster_points = [p['highlight'] for p in points]
            
            cluster_list.append({
                'cluster_id': cluster_id,
                'original_label': int(label),
                'points': cluster_points,
                'is_expanded': False  # 아직 확장 안됨
            })
            cluster_id += 1
        
        self.logger.info(f"✅ DBSCAN 완료: {len(highlights)}개 → {len(cluster_list)}개 클러스터")
        
        # 클러스터 정보 출력
        for cluster in cluster_list:
            point_count = len(cluster['points'])
            timestamps = [p['timestamp'] for p in cluster['points']]
            time_range = f"{min(timestamps):.1f}~{max(timestamps):.1f}초" if point_count > 1 else f"{timestamps[0]:.1f}초"
            self.logger.info(f"   클러스터 {cluster['cluster_id']}: {point_count}개 포인트 ({time_range})")
        
        return cluster_list
    
    def expand_single_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """
        단일 포인트 클러스터를 ±buffer초로 확장
        
        Args:
            clusters (List[Dict]): 원본 클러스터 리스트
            
        Returns:
            List[Dict]: 확장된 클러스터 리스트
        """
        self.logger.info("🔧 단일 포인트 클러스터 확장 중...")
        
        expanded_clusters = []
        single_count = 0
        
        for cluster in clusters:
            if len(cluster['points']) == 1:
                # 단일 포인트 클러스터 → 확장
                expanded_cluster = self._expand_single_cluster(cluster)
                expanded_clusters.append(expanded_cluster)
                single_count += 1
                
                original_time = cluster['points'][0]['timestamp']
                self.logger.info(f"   클러스터 {cluster['cluster_id']}: {original_time:.1f}초 → "
                               f"{original_time - self.expansion_buffer:.1f}~{original_time + self.expansion_buffer:.1f}초")
            else:
                # 멀티 포인트 클러스터 → 그대로 유지
                expanded_clusters.append(cluster)
        
        self.logger.info(f"✅ 단일 클러스터 확장 완료: {single_count}개 확장됨")
        return expanded_clusters
    
    def _expand_single_cluster(self, cluster: Dict) -> Dict:
        """
        단일 포인트 클러스터를 ±buffer초로 확장
        
        Args:
            cluster (Dict): 단일 포인트 클러스터
            
        Returns:
            Dict: 확장된 클러스터
        """
        original_point = cluster['points'][0]
        original_time = original_point['timestamp']
        
        # 가상 포인트 생성
        virtual_start = {
            'timestamp': original_time - self.expansion_buffer,
            'tension': 0.0,  # 가상 포인트는 텐션 0
            'type': 'virtual_start'
        }
        
        virtual_end = {
            'timestamp': original_time + self.expansion_buffer,
            'tension': 0.0,
            'type': 'virtual_end'
        }
        
        # 원본 포인트에 타입 추가
        enhanced_original = original_point.copy()
        enhanced_original['type'] = 'original'
        
        # 확장된 클러스터 생성
        expanded_cluster = cluster.copy()
        expanded_cluster['points'] = [virtual_start, enhanced_original, virtual_end]
        expanded_cluster['is_expanded'] = True
        
        return expanded_cluster
    
    def get_cluster_spans(self, clusters: List[Dict]) -> List[Dict]:
        """
        각 클러스터의 시간 범위 계산
        
        Args:
            clusters (List[Dict]): 클러스터 리스트
            
        Returns:
            List[Dict]: 시간 범위가 추가된 클러스터 리스트
        """
        self.logger.info("⏱️ 클러스터 시간 범위 계산 중...")
        
        clusters_with_spans = []
        
        for cluster in clusters:
            timestamps = [p['timestamp'] for p in cluster['points']]
            
            span = {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration': max(timestamps) - min(timestamps)
            }
            
            # 클러스터에 span 정보 추가
            cluster_with_span = cluster.copy()
            cluster_with_span['span'] = span
            
            clusters_with_spans.append(cluster_with_span)
            
            self.logger.info(f"   클러스터 {cluster['cluster_id']}: "
                           f"{span['start']:.1f}~{span['end']:.1f}초 (지속시간: {span['duration']:.1f}초)")
        
        self.logger.info("✅ 클러스터 시간 범위 계산 완료")
        return clusters_with_spans
    
    def process_highlights(self, tension_json_path: str) -> Dict:
        """
        전체 하이라이트 클러스터링 프로세스
        
        Args:
            tension_json_path (str): 텐션 JSON 파일 경로
            
        Returns:
            Dict: 클러스터링 결과 (메타데이터 포함)
        """
        self.logger.info("🎪 하이라이트 클러스터링 프로세스 시작")
        
        # 1. 하이라이트 로드
        highlights = self.load_tension_highlights(tension_json_path)
        
        # 2. DBSCAN 클러스터링
        clusters = self.cluster_with_dbscan(highlights)
        
        # 3. 단일 클러스터 확장
        expanded_clusters = self.expand_single_clusters(clusters)
        
        # 4. 시간 범위 계산
        final_clusters = self.get_cluster_spans(expanded_clusters)
        
        # 5. 결과 구성
        result = {
            'metadata': {
                'video_name': self._extract_video_name(tension_json_path),
                'source_file': os.path.basename(tension_json_path),
                'total_highlights': len(highlights),
                'total_clusters': len(final_clusters),
                'single_expanded_count': sum(1 for c in final_clusters if c.get('is_expanded', False)),
                'clustered_at': datetime.now().isoformat(),
                'config': {
                    'eps': self.eps,
                    'min_samples': self.min_samples,
                    'expansion_buffer': self.expansion_buffer
                }
            },
            'clusters': final_clusters
        }
        
        self.logger.info("🎪 하이라이트 클러스터링 완료!")
        self.logger.info(f"   최종 결과: {len(highlights)}개 하이라이트 → {len(final_clusters)}개 클러스터")
        
        return result
    
    def _extract_video_name(self, tension_json_path: str) -> str:
        """텐션 JSON 파일명에서 비디오 이름 추출"""
        filename = os.path.basename(tension_json_path)
        # tension_f_001_박정민_유튜브_살리기_11.0_24.0_20250612_135829.json
        # → f_001_박정민_유튜브_살리기_11.0_24.0
        if filename.startswith('tension_'):
            name_part = filename[8:]  # 'tension_' 제거
            # 마지막 타임스탬프 부분 제거 (_20250612_135829.json)
            parts = name_part.split('_')
            if len(parts) >= 3:
                # 마지막 2개 부분이 날짜+시간 형식이면 제거
                if parts[-1].endswith('.json'):
                    parts[-1] = parts[-1].replace('.json', '')
                if len(parts) > 2 and len(parts[-1]) == 6 and parts[-1].isdigit():  # 시간 부분
                    parts = parts[:-1]
                if len(parts) > 2 and len(parts[-1]) == 8 and parts[-1].isdigit():  # 날짜 부분
                    parts = parts[:-1]
                return '_'.join(parts)
        
        # 추출 실패 시 확장자만 제거
        return os.path.splitext(filename)[0]
    
    def save_clusters(self, clusters_result: Dict, output_path: str) -> None:
        """
        클러스터링 결과를 JSON 파일로 저장
        
        Args:
            clusters_result (Dict): 클러스터링 결과
            output_path (str): 저장할 파일 경로 (프로젝트 루트 기준)
        """
        try:
            # 프로젝트 루트 기준 절대 경로 생성
            if not os.path.isabs(output_path):
                output_path = os.path.join(project_root, output_path)
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # JSON 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(clusters_result, f, indent=2, ensure_ascii=False)
            
            # 상대 경로로 로그 출력
            relative_path = os.path.relpath(output_path, project_root)
            self.logger.info(f"💾 클러스터링 결과 저장 완료: {relative_path}")
            
            # 요약 정보 출력
            metadata = clusters_result['metadata']
            self.logger.info(f"   총 클러스터: {metadata['total_clusters']}개")
            self.logger.info(f"   확장된 클러스터: {metadata['single_expanded_count']}개")
            
        except Exception as e:
            self.logger.error(f"❌ 클러스터링 결과 저장 실패: {e}")
            raise


def main():
    """테스트 실행"""
    import argparse
    
    # 프로젝트 루트로 작업 디렉토리 변경
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description='하이라이트 클러스터링')
    parser.add_argument('tension_json', help='텐션 JSON 파일 경로 (프로젝트 루트 기준)')
    parser.add_argument('--output', help='출력 JSON 경로 (기본: outputs/clip_analysis/clusters.json)')
    parser.add_argument('--config', help='Config 파일 경로')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 클러스터링 실행
        clusterer = HighlightClusterer(config_path=args.config)
        result = clusterer.process_highlights(args.tension_json)
        
        # 결과 저장 (프로젝트 루트 기준 경로)
        if args.output:
            output_path = args.output
        else:
            # 기본 출력 경로 생성
            video_name = result['metadata']['video_name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"outputs/clip_analysis/{video_name}/clusters_{video_name}_{timestamp}.json"
        
        clusterer.save_clusters(result, output_path)
        
        print(f"\n✅ 하이라이트 클러스터링 완료!")
        print(f"📊 {result['metadata']['total_highlights']}개 하이라이트 → {result['metadata']['total_clusters']}개 클러스터")
        print(f"💾 결과 저장: {os.path.relpath(output_path if os.path.isabs(output_path) else os.path.join(project_root, output_path), project_root)}")
        
    except Exception as e:
        print(f"❌ 클러스터링 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()