#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class ClipSelector:
    """
    점수 기반 클립 자동 선별기
    - scored_windows.json에서 윈도우 불러오기
    - config 기준 필터링 및 NMS 적용
    - 최종 클립 selected_clips.json 저장
    """
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        # Config 로드
        if config_path is None:
            config_path = "pipeline/configs/funclip_extraction_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # selection 섹션에서 파라미터 추출
        selection_config = config.get('selection', {})
        self.iou_threshold = selection_config.get('iou_threshold', 0.5)
        self.target_clips = selection_config.get('target_clips', 5)
        self.min_score = selection_config.get('min_score', 0.0)
        self.min_duration = selection_config.get('min_duration', 15)
        self.max_duration = selection_config.get('max_duration', 40)

    def load_scored_windows(self, scored_windows_path: str) -> Dict:
        """scored_windows.json에서 윈도우 및 메타데이터 로드"""
        with open(scored_windows_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 전체 데이터(메타데이터 포함) 반환
        self.logger.info(f"📊 윈도우 로드 완료: {len(data['windows'])}개")
        if 'metadata' in data and 'video_name' in data['metadata']:
            self.logger.info(f"   비디오: {data['metadata']['video_name']}")
        return data

    def filter_windows(self, data: Dict) -> Tuple[Dict, List[Dict]]:
        """윈도우 필터링 (메타데이터 유지)"""
        windows = data['windows']
        filtered = [
            w for w in windows
            if w['fun_score'] >= self.min_score and self.min_duration <= w['duration'] <= self.max_duration
        ]
        self.logger.info(f"1차 필터링: {len(filtered)}개 (점수≥{self.min_score}, 길이 {self.min_duration}~{self.max_duration})")
        return data, filtered

    def calculate_iou(self, win1: Dict, win2: Dict) -> float:
        # 시간 구간 IoU 계산
        s1, e1 = win1['start_time'], win1['end_time']
        s2, e2 = win2['start_time'], win2['end_time']
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    def non_max_suppression(self, windows: List[Dict]) -> List[Dict]:
        # 점수 내림차순 정렬
        windows = sorted(windows, key=lambda w: w['fun_score'], reverse=True)
        selected = []
        for win in windows:
            if all(self.calculate_iou(win, sel) < self.iou_threshold for sel in selected):
                selected.append(win)
            if len(selected) >= self.target_clips:
                break
        self.logger.info(f"NMS 적용 후: {len(selected)}개 (IoU<{self.iou_threshold})")
        return selected

    def select_clips(self, scored_windows_path: str) -> Tuple[Dict, List[Dict]]:
        """점수 기반 클립 선별 (메타데이터 유지)"""
        data = self.load_scored_windows(scored_windows_path)
        data, filtered = self.filter_windows(data)
        selected = self.non_max_suppression(filtered)
        return data, selected

    def save_selected_clips(self, original_data: Dict, clips: List[Dict], output_path: str, config_used: Dict):
        """선별된 클립 저장 (메타데이터 포함)"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 원본 메타데이터 가져오기
        metadata = original_data.get('metadata', {})
        video_name = metadata.get('video_name', 'unknown')
        
        result = {
            'metadata': {
                'video_name': video_name,
                'source_scored_windows': os.path.basename(original_data.get('source_path', 'unknown')),
                'total_clips': len(clips),
                'total_original_windows': len(original_data.get('windows', [])),
                'original_score_statistics': metadata.get('score_statistics', {})
            },
            'selection_info': {
                'selected_at': __import__('datetime').datetime.now().isoformat(),
                'config_used': config_used
            },
            'clips': [
                {
                    'rank': i+1,
                    'start_time': c['start_time'],
                    'end_time': c['end_time'],
                    'duration': c['duration'],
                    'fun_score': c['fun_score'],
                    'cluster_id': c.get('cluster_id', -1)
                } for i, c in enumerate(clips)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 클립 선별 결과 저장 완료: {output_path}")
        self.logger.info(f"   비디오: {video_name}")
        self.logger.info(f"   최종 클립: {len(clips)}개")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='점수 기반 클립 자동 선별')
    parser.add_argument('scored_windows', help='scored_windows.json 경로')
    parser.add_argument('--config', help='Config 파일 경로')
    parser.add_argument('--output', help='출력 JSON 경로')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        selector = ClipSelector(config_path=args.config)
        
        # 원본 파일 경로 저장
        original_data, selected = selector.select_clips(args.scored_windows)
        original_data['source_path'] = args.scored_windows
        
        # 출력 경로 결정
        if args.output:
            output_path = args.output
        else:
            # 비디오 이름과 타임스탬프를 포함한 파일명 생성
            base_dir = os.path.dirname(args.scored_windows)
            video_name = original_data.get('metadata', {}).get('video_name', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(base_dir, f'selected_clips_{video_name}_{timestamp}.json')
        
        # config 정보 저장
        config_used = {
            'iou_threshold': selector.iou_threshold,
            'target_clips': selector.target_clips,
            'min_score': selector.min_score,
            'min_duration': selector.min_duration,
            'max_duration': selector.max_duration
        }
        selector.save_selected_clips(original_data, selected, output_path, config_used)
        print(f"\n✅ 클립 선별 완료!")
        print(f"📊 {len(selected)}개 클립 저장: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"❌ 클립 선별 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 