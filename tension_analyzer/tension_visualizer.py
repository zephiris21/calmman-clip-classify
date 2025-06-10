#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import glob
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# 한글 폰트 설정 (선택사항)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

class TensionVisualizer:
    """
    텐션 분석 결과 시각화 클래스
    주피터랩에서 셀 단위로 편리하게 사용할 수 있도록 설계
    """
    
    def __init__(self, debug_faces_dir: str = "video_analyzer/preprocessed_data/debug_faces/chimchakman"):
        """
        시각화 도구 초기화
        
        Args:
            debug_faces_dir (str): 얼굴 이미지가 저장된 디렉토리
        """
        self.debug_faces_dir = debug_faces_dir
        self.tension_data = None
        self.video_name = None
        
        # 감정 레이블 및 색상
        self.emotion_labels = [
            'Anger', 'Contempt', 'Disgust', 'Fear', 
            'Happiness', 'Neutral', 'Sadness', 'Surprise'
        ]
        
        # 감정별 색상 (matplotlib 기본 팔레트)
        self.emotion_colors = [
            '#ff4444',  # Anger - 빨강
            '#8B4513',  # Contempt - 갈색  
            '#32CD32',  # Disgust - 녹색
            '#9370DB',  # Fear - 보라
            '#FFD700',  # Happiness - 노랑
            '#808080',  # Neutral - 회색
            '#4169E1',  # Sadness - 파랑
            '#FF8C00'   # Surprise - 주황
        ]
        
        print(f"✅ TensionVisualizer 초기화 완료")
        print(f"   얼굴 이미지 디렉토리: {debug_faces_dir}")
    
    def load_tension_data(self, json_file_path: str) -> bool:
        """
        텐션 분석 JSON 결과 로드
        
        Args:
            json_file_path (str): JSON 파일 경로 또는 패턴
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            # 패턴으로 파일 찾기
            if not os.path.exists(json_file_path):
                # tension_analyzer/outputs/tension_data/ 에서 패턴 검색
                search_dir = "tension_analyzer/outputs/tension_data"
                if os.path.exists(search_dir):
                    pattern = os.path.join(search_dir, f"*{json_file_path}*.json")
                    files = glob.glob(pattern)
                    if files:
                        json_file_path = files[0]  # 첫 번째 매칭 파일 사용
                        print(f"📂 패턴 매칭 파일 발견: {os.path.basename(json_file_path)}")
                    else:
                        print(f"❌ 패턴에 맞는 파일을 찾을 수 없음: {json_file_path}")
                        return False
                else:
                    print(f"❌ 텐션 데이터 디렉토리가 없음: {search_dir}")
                    return False
            
            # JSON 로드
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.tension_data = json.load(f)
            
            self.video_name = self.tension_data['metadata']['video_name']
            
            print(f"✅ 텐션 데이터 로드 완료")
            print(f"   영상명: {self.video_name}")
            print(f"   지속시간: {self.tension_data['metadata']['duration']:.1f}초")
            print(f"   텐션 포인트: {len(self.tension_data['tension_timeline']['timestamps'])}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 텐션 데이터 로드 실패: {e}")
            return False
    
    def plot_tension_curves(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        3가지 텐션 곡선 시각화
        
        Args:
            figsize: 그래프 크기
        """
        if self.tension_data is None:
            print("❌ 텐션 데이터가 로드되지 않았습니다. load_tension_data()를 먼저 호출하세요.")
            return
        
        timeline = self.tension_data['tension_timeline']
        timestamps = np.array(timeline['timestamps'])
        emotion_tension = np.array(timeline['emotion_tension'])
        audio_tension = np.array(timeline['audio_tension'])
        combined_tension = np.array(timeline['combined_tension'])
        
        # 시간을 분:초 형태로 변환
        time_labels = [str(timedelta(seconds=int(t))) for t in timestamps[::len(timestamps)//10]]
        time_indices = np.linspace(0, len(timestamps)-1, len(time_labels), dtype=int)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle(f'텐션 분석 결과: {self.video_name}', fontsize=16, fontweight='bold')
        
        # 1. 감정 텐션
        axes[0].plot(timestamps, emotion_tension, color='#ff6b6b', linewidth=2, label='Emotion Tension')
        axes[0].fill_between(timestamps, emotion_tension, alpha=0.3, color='#ff6b6b')
        axes[0].set_ylabel('감정 텐션', fontsize=12)
        axes[0].set_title('감정 기반 텐션 (중립 제외 7감정 + Arousal×10)', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. 오디오 텐션
        axes[1].plot(timestamps, audio_tension, color='#4ecdc4', linewidth=2, label='Audio Tension')
        axes[1].fill_between(timestamps, audio_tension, alpha=0.3, color='#4ecdc4')
        axes[1].set_ylabel('오디오 텐션', fontsize=12)
        axes[1].set_title('음성 기반 텐션 (VAD 필터링된 Voice RMS)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. 결합 텐션 + 하이라이트 표시
        axes[2].plot(timestamps, combined_tension, color='#45b7d1', linewidth=2, label='Combined Tension')
        axes[2].fill_between(timestamps, combined_tension, alpha=0.3, color='#45b7d1')
        
        # 하이라이트 포인트 표시
        highlights = self.tension_data['edit_suggestions']['highlights']
        if highlights:
            highlight_times = [h['timestamp'] for h in highlights]
            highlight_tensions = [h['tension'] for h in highlights]
            axes[2].scatter(highlight_times, highlight_tensions, 
                          color='red', s=100, zorder=5, label=f'Highlights ({len(highlights)}개)')
        
        axes[2].set_ylabel('결합 텐션', fontsize=12)
        axes[2].set_xlabel('시간', fontsize=12)
        axes[2].set_title('결합 텐션 (감정 70% + 오디오 30%) + 하이라이트', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # X축 시간 라벨 설정
        axes[2].set_xticks(timestamps[time_indices])
        axes[2].set_xticklabels(time_labels, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 정보 출력
        stats = self.tension_data['statistics']
        print(f"\n📊 텐션 통계:")
        print(f"   평균: {stats['avg_tension']:.2f}")
        print(f"   최대: {stats['max_tension']:.2f}")
        print(f"   최소: {stats['min_tension']:.2f}")
        print(f"   표준편차: {stats['std_tension']:.2f}")
        print(f"   음성 활동 비율: {stats['voice_activity_ratio']:.1%}")
    
    def plot_emotion_pie_chart(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        감정 분포 파이차트 시각화
        HDF5에서 감정 데이터를 로드해서 통계 계산
        """
        if self.tension_data is None:
            print("❌ 텐션 데이터가 로드되지 않았습니다.")
            return
        
        # HDF5에서 감정 데이터 로드 시도
        try:
            video_h5_path = self._find_video_h5_file()
            if video_h5_path is None:
                print("❌ 비디오 HDF5 파일을 찾을 수 없습니다.")
                return
            
            import h5py
            with h5py.File(video_h5_path, 'r') as f:
                emotions = f['sequences/emotions'][:]  # [N, 10]
                face_detected = f['sequences/face_detected'][:]
            
            print(f"✅ 감정 데이터 로드: {emotions.shape}")
            
        except Exception as e:
            print(f"❌ 감정 데이터 로드 실패: {e}")
            return
        
        # 얼굴이 탐지된 프레임만 사용
        valid_emotions = emotions[face_detected == True]
        if len(valid_emotions) == 0:
            print("❌ 유효한 감정 데이터가 없습니다.")
            return
        
        # 8개 감정별로 양수값만 필터링해서 평균 계산
        emotion_means = []
        for i in range(8):
            emotion_values = valid_emotions[:, i]
            positive_values = emotion_values[emotion_values > 0]  # 양수만 필터링
            
            if len(positive_values) > 0:
                emotion_mean = np.mean(positive_values)
            else:
                emotion_mean = 0.0  # 양수값이 없으면 0
            
            emotion_means.append(emotion_mean)
            
            # 각 감정별 통계 출력 (디버깅)
            print(f"   {self.emotion_labels[i]}: 전체 {len(emotion_values)}개, 양수 {len(positive_values)}개, 평균 {emotion_mean:.4f}")
        
        emotion_means = np.array(emotion_means)
        
        # 원시 값 확인을 위한 로깅
        print(f"📊 양수값만 필터링한 감정 평균값: {emotion_means}")
        print(f"   데이터 범위: 최소 {np.min(emotion_means):.3f}, 최대 {np.max(emotion_means):.3f}")
        
        # 총합이 0이면 균등 분포로 처리
        if np.sum(emotion_means) == 0:
            emotion_means = np.ones(8) / 8
            print("⚠️ 모든 감정의 양수값이 없어 균등 분포로 설정")
        
        # 총합이 0이면 균등 분포로 처리
        if np.sum(emotion_means) == 0:
            emotion_means = np.ones(8) / 8
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'감정 분석: {self.video_name}', fontsize=16, fontweight='bold')
        
        # 1. 파이차트
        wedges, texts, autotexts = ax1.pie(emotion_means, labels=self.emotion_labels, 
                                          colors=self.emotion_colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
        ax1.set_title('감정 분포 (얼굴 탐지된 프레임 기준)', fontsize=12)
        
        # 2. 막대 그래프
        bars = ax2.bar(self.emotion_labels, emotion_means, color=self.emotion_colors, alpha=0.7)
        ax2.set_title('감정별 평균 강도', fontsize=12)
        ax2.set_ylabel('평균 감정 값', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, emotion_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # 상위 3개 감정 출력
        top_emotions = np.argsort(emotion_means)[-3:][::-1]
        print(f"\n🎭 주요 감정:")
        for i, idx in enumerate(top_emotions, 1):
            print(f"   {i}. {self.emotion_labels[idx]}: {emotion_means[idx]:.3f}")
    
    def show_emotion_peak_faces(self, top_n: int = 1, figsize: Tuple[int, int] = (16, 10)) -> None:
        """
        감정별 피크 시점의 실제 얼굴 이미지 표시 (깔끔한 subplot 버전)
        
        Args:
            top_n: 각 감정별로 표시할 상위 N개 (기본값 1)
            figsize: 전체 그래프 크기
        """
        if self.tension_data is None:
            print("❌ 텐션 데이터가 로드되지 않았습니다.")
            return
        
        try:
            # HDF5에서 감정 데이터 로드
            video_h5_path = self._find_video_h5_file()
            if video_h5_path is None:
                return
            
            import h5py
            with h5py.File(video_h5_path, 'r') as f:
                emotions = f['sequences/emotions'][:]  # [N, 10]
                face_detected = f['sequences/face_detected'][:]
                timestamps = f['sequences/timestamps'][:]
            
            print(f"✅ 감정 데이터 로드: {emotions.shape}")
            
        except Exception as e:
            print(f"❌ 감정 데이터 로드 실패: {e}")
            return
        
        # 유효한 프레임만 필터링
        valid_mask = face_detected == True
        valid_emotions = emotions[valid_mask]
        valid_timestamps = timestamps[valid_mask]
        
        if len(valid_emotions) == 0:
            print("❌ 유효한 감정 데이터가 없습니다.")
            return
        
        # 피크 데이터 수집 (7개 감정 + Valence + Arousal)
        peak_data = []
        
        # 7개 감정 (Neutral 제외)
        for i, emotion_name in enumerate(self.emotion_labels):
            if emotion_name == 'Neutral':
                continue
                
            emotion_values = valid_emotions[:, i]
            emotion_values = np.maximum(emotion_values, 0)  # 양수만
            
            if np.max(emotion_values) > 0:
                peak_idx = np.argmax(emotion_values)
                peak_data.append({
                    'name': emotion_name,
                    'timestamp': valid_timestamps[peak_idx],
                    'value': emotion_values[peak_idx],
                    'type': 'emotion'
                })
        
        # Valence (8번 인덱스)
        valence_values = valid_emotions[:, 8]
        if len(valence_values) > 0:
            # Valence는 절댓값이 큰 것 (극값)
            abs_valence = np.abs(valence_values)
            peak_idx = np.argmax(abs_valence)
            peak_data.append({
                'name': 'Valence',
                'timestamp': valid_timestamps[peak_idx],
                'value': valence_values[peak_idx],
                'type': 'va'
            })
        
        # Arousal (9번 인덱스)
        arousal_values = valid_emotions[:, 9]
        if len(arousal_values) > 0:
            peak_idx = np.argmax(arousal_values)
            peak_data.append({
                'name': 'Arousal',
                'timestamp': valid_timestamps[peak_idx],
                'value': arousal_values[peak_idx],
                'type': 'va'
            })
        
        if not peak_data:
            print("❌ 표시할 감정 피크가 없습니다.")
            return
        
        # 3x3 그리드로 배치 (최대 9개)
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'감정별 피크 순간의 실제 얼굴: {self.video_name}', fontsize=16, fontweight='bold', y=0.99)
        
        # axes를 1차원으로 변환
        axes_flat = axes.flatten()
        
        for i, peak in enumerate(peak_data[:9]):  # 최대 9개만 표시
            ax = axes_flat[i]
            
            # 해당 시점의 얼굴 이미지 찾기
            face_image = self._find_face_image_at_time(peak['timestamp'])
            
            if face_image is not None:
                ax.imshow(face_image)
                
                # 제목 설정 (감정별 색상)
                color = self.emotion_colors[self.emotion_labels.index(peak['name'])] if peak['name'] in self.emotion_labels else '#333333'
                
                time_str = str(timedelta(seconds=int(peak['timestamp'])))
                if peak['type'] == 'va':
                    title = f"{peak['name']}\n{time_str}\n값: {peak['value']:.3f}"
                else:
                    title = f"{peak['name']}\n{time_str}\n값: {peak['value']:.3f}"
                    
                ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=10)
            else:
                # 이미지를 찾을 수 없으면 빈 박스
                ax.set_facecolor('#f5f5f5')
                time_str = str(timedelta(seconds=int(peak['timestamp'])))
                ax.text(0.5, 0.5, f"{peak['name']}\n{time_str}\n이미지 없음", 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=10, color='#666666')
            
            ax.axis('off')
        
        # 사용하지 않는 subplot 숨기기
        for i in range(len(peak_data), 9):
            axes_flat[i].axis('off')
        
        # 여백 조정 (겹침 방지)
        plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.2)
        plt.show()
        
        print(f"🎭 감정 피크 표시 완료 ({len(peak_data)}개: 7감정 + VA)")
    
    def _find_video_h5_file(self) -> Optional[str]:
        """비디오 HDF5 파일 찾기"""
        if self.video_name is None:
            return None
        
        video_sequences_dir = "video_analyzer/preprocessed_data/video_sequences"
        if not os.path.exists(video_sequences_dir):
            return None
        
        # 비디오명으로 파일 찾기
        for file in os.listdir(video_sequences_dir):
            if self.video_name.replace('.mp4', '') in file and file.endswith('.h5'):
                return os.path.join(video_sequences_dir, file)
        
        return None
    
    def _auto_detect_face_dir(self) -> bool:
        """HDF5에서 얼굴 폴더 경로 자동 감지"""
        try:
            video_h5_path = self._find_video_h5_file()
            if video_h5_path is None:
                return False
            
            import h5py
            with h5py.File(video_h5_path, 'r') as f:
                # HDF5에 저장된 얼굴 폴더 경로 읽기
                if 'chimchakman_faces_dir' in f.attrs:
                    auto_detected_dir = f.attrs['chimchakman_faces_dir']
                    if isinstance(auto_detected_dir, bytes):
                        auto_detected_dir = auto_detected_dir.decode('utf-8')
                    
                    if os.path.exists(auto_detected_dir):
                        self.debug_faces_dir = auto_detected_dir
                        print(f"✅ HDF5에서 얼굴 폴더 자동 감지: {auto_detected_dir}")
                        return True
                    else:
                        print(f"⚠️ HDF5 경로가 존재하지 않음: {auto_detected_dir}")
                
                # 대안: face_images_dir에서 chimchakman 하위 폴더 찾기
                elif 'face_images_dir' in f.attrs:
                    base_face_dir = f.attrs['face_images_dir']
                    if isinstance(base_face_dir, bytes):
                        base_face_dir = base_face_dir.decode('utf-8')
                    
                    chimchakman_dir = os.path.join(base_face_dir, "chimchakman")
                    if os.path.exists(chimchakman_dir):
                        self.debug_faces_dir = chimchakman_dir
                        print(f"✅ HDF5에서 얼굴 폴더 자동 구성: {chimchakman_dir}")
                        return True
                
                print("⚠️ HDF5에 얼굴 폴더 정보가 없습니다")
                return False
                
        except Exception as e:
            print(f"⚠️ 얼굴 폴더 자동 감지 실패: {e}")
            return False
    
    def _find_face_image_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """특정 시간의 얼굴 이미지 찾기 (타임스탬프 기반)"""
        # 먼저 HDF5에서 얼굴 폴더 자동 감지 시도
        if not os.path.exists(self.debug_faces_dir):
            if not self._auto_detect_face_dir():
                return None
        
        if not os.path.exists(self.debug_faces_dir):
            return None
        
        # 타임스탬프를 밀리초 단위 정수로 변환 (15.750초 → 015750)
        timestamp_ms = int(timestamp * 1000)
        
        # 타임스탬프 기반 파일명 패턴
        timestamp_pattern = f"timestamp_{timestamp_ms:06d}_face0_chimchakman_*.jpg"
        
        # 정확한 매칭 시도
        search_pattern = os.path.join(self.debug_faces_dir, timestamp_pattern)
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            try:
                image = Image.open(matching_files[0])
                return np.array(image)
            except Exception as e:
                print(f"⚠️ 이미지 로드 실패: {matching_files[0]} - {e}")
        
        # 정확한 매칭이 안되면 주변 시간 검색 (±0.25초 = ±250ms)
        for offset_ms in [-250, -125, 125, 250]:
            alt_timestamp_ms = timestamp_ms + offset_ms
            if alt_timestamp_ms >= 0:
                alt_pattern = f"timestamp_{alt_timestamp_ms:06d}_face0_chimchakman_*.jpg"
                alt_search = os.path.join(self.debug_faces_dir, alt_pattern)
                alt_files = glob.glob(alt_search)
                
                if alt_files:
                    try:
                        image = Image.open(alt_files[0])
                        return np.array(image)
                    except Exception:
                        continue
        
        return None
    
    def auto_find_faces_dir(self, video_name: str = None) -> bool:
        """
        영상명 기반 얼굴 폴더 자동 탐지 (공개 메서드)
        
        Args:
            video_name (str): 영상명 (옵션, 없으면 현재 로드된 영상명 사용)
            
        Returns:
            bool: 탐지 성공 여부
        """
        if video_name:
            self.video_name = video_name
        
        if self.video_name is None:
            print("❌ 영상명이 설정되지 않았습니다")
            return False
        
        return self._auto_detect_face_dir()
    
    def show_summary(self) -> None:
        """전체 분석 요약 정보 출력"""
        if self.tension_data is None:
            print("❌ 텐션 데이터가 로드되지 않았습니다.")
            return
        
        print("="*60)
        print(f"📊 텐션 분석 요약: {self.video_name}")
        print("="*60)
        
        stats = self.tension_data['statistics']
        print(f"🎬 기본 정보:")
        print(f"   영상 길이: {self.tension_data['metadata']['duration']:.1f}초")
        print(f"   처리 시간: {self.tension_data['metadata']['processed_at']}")
        
        print(f"\n⚡ 텐션 통계:")
        print(f"   평균 텐션: {stats['avg_tension']:.2f}")
        print(f"   최대 텐션: {stats['max_tension']:.2f}")
        print(f"   최소 텐션: {stats['min_tension']:.2f}")
        print(f"   표준편차: {stats['std_tension']:.2f}")
        
        print(f"\n✂️ 편집 제안:")
        print(f"   하이라이트: {stats['highlight_count']}개")
        print(f"   컷 포인트: {stats['cut_point_count']}개")
        print(f"   저에너지 구간: {stats['low_energy_count']}개")
        
        print(f"\n🎵 음성 정보:")
        print(f"   음성 활동 비율: {stats['voice_activity_ratio']:.1%}")
        
        # 상위 3개 하이라이트
        highlights = self.tension_data['edit_suggestions']['highlights']
        if highlights:
            top_highlights = sorted(highlights, key=lambda x: x['tension'], reverse=True)[:3]
            print(f"\n🎯 주요 하이라이트:")
            for i, hl in enumerate(top_highlights, 1):
                time_str = str(timedelta(seconds=int(hl['timestamp'])))
                print(f"   {i}. {time_str} (텐션: {hl['tension']:.2f})")
        
        print("="*60)


# 주피터랩에서 사용할 수 있는 편의 함수들
def quick_load_and_visualize(filename_pattern: str, auto_detect_faces: bool = True) -> TensionVisualizer:
    """
    빠른 로드 및 시각화 (주피터랩 용)
    
    Args:
        filename_pattern: 파일명 패턴
        auto_detect_faces: 얼굴 폴더 자동 감지 여부
        
    Returns:
        TensionVisualizer: 초기화된 시각화 객체
    """
    viz = TensionVisualizer()
    if viz.load_tension_data(filename_pattern):
        # 얼굴 폴더 자동 감지
        if auto_detect_faces:
            viz.auto_find_faces_dir()
        viz.show_summary()
        return viz
    else:
        return None

def show_all_plots(filename_pattern: str, auto_detect_faces: bool = True) -> None:
    """
    모든 플롯을 한번에 표시 (주피터랩 용)
    
    Args:
        filename_pattern: 파일명 패턴
    """
    viz = TensionVisualizer()
    if viz.load_tension_data(filename_pattern):
        viz.plot_tension_curves()
        viz.plot_emotion_pie_chart()
        viz.show_emotion_peak_faces()
    else:
        print("❌ 데이터 로드 실패")


if __name__ == "__main__":
    # 테스트 코드
    print("🎨 TensionVisualizer 테스트")
    
    # 사용 예시 출력
    print("\n📋 주피터랩 사용 예시:")
    print("```python")
    print("from tension_visualizer import TensionVisualizer, quick_load_and_visualize")
    print("")
    print("# 방법 1: 단계별 시각화 (자동 얼굴 폴더 감지)")
    print("viz = TensionVisualizer()")
    print("viz.load_tension_data('your_filename_pattern')")
    print("viz.auto_find_faces_dir()         # HDF5에서 자동 감지")
    print("viz.plot_tension_curves()        # 셀 1")
    print("viz.plot_emotion_pie_chart()     # 셀 2")
    print("viz.show_emotion_peak_faces()    # 셀 3")
    print("")
    print("# 방법 2: 빠른 로드 (자동 감지 포함)")
    print("viz = quick_load_and_visualize('your_filename_pattern')")
    print("viz.plot_tension_curves()")
    print("")
    print("# 방법 3: 모든 플롯 한번에 (자동 감지 포함)")
    print("show_all_plots('your_filename_pattern')")
    print("")
    print("# 방법 4: 수동 폴더 지정")
    print("viz = TensionVisualizer(debug_faces_dir='custom_path/chimchakman')")
    print("viz.load_tension_data('your_filename_pattern')")
    print("")
    print("# 📁 타임스탬프 파일명 형식: timestamp_015750_face0_chimchakman_sim0.748.jpg")