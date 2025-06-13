#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import json
import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import numpy as np

# 페이지 설정
st.set_page_config(
   page_title="재미 클립 분석 대시보드",
   page_icon="🎬",
   layout="wide",
   initial_sidebar_state="expanded"
)

def load_latest_files(base_dir="outputs"):
   """최신 분석 결과 파일들 찾기"""
   files = {}
   
   # clip_analysis 디렉토리에서 최신 파일들 찾기
   clip_analysis_dir = os.path.join(base_dir, "clip_analysis")
   if os.path.exists(clip_analysis_dir):
       for video_dir in os.listdir(clip_analysis_dir):
           video_path = os.path.join(clip_analysis_dir, video_dir)
           if os.path.isdir(video_path):
               video_files = {}
               
               # 각 타입별 최신 파일 찾기
               file_patterns = {
                   'clusters': 'clusters_*.json',
                   'scored_windows': 'scored_windows_*.json',
                   'selected_clips': 'selected_clips_*.json',
                   'refined_clips': 'refined_clips_*.json'
               }
               
               for file_type, pattern in file_patterns.items():
                   matches = glob.glob(os.path.join(video_path, pattern))
                   if matches:
                       video_files[file_type] = max(matches, key=os.path.getmtime)
               
               if video_files:
                   files[video_dir] = video_files
   
   # 기존 시각화 PNG 파일들 찾기
   viz_dir = os.path.join(base_dir, "visualization")
   if os.path.exists(viz_dir):
       for video_dir in os.listdir(viz_dir):
           video_path = os.path.join(viz_dir, video_dir)
           if os.path.isdir(video_path):
               if video_dir not in files:
                   files[video_dir] = {}
               
               # PNG 파일들 찾기
               png_files = glob.glob(os.path.join(video_path, "*.png"))
               if png_files:
                   files[video_dir]['visualizations'] = png_files
   
   return files

def load_json_file(file_path):
   """JSON 파일 로드"""
   try:
       with open(file_path, 'r', encoding='utf-8') as f:
           return json.load(f)
   except Exception as e:
       st.error(f"파일 로드 실패: {e}")
       return None

def create_score_distribution_chart(scored_windows_data):
   """재미도 점수 분포 차트"""
   if not scored_windows_data or 'windows' not in scored_windows_data:
       return None
   
   scores = [w['fun_score'] for w in scored_windows_data['windows']]
   
   # 히스토그램
   fig = px.histogram(
       x=scores,
       nbins=30,
       title="재미도 점수 분포",
       labels={'x': '재미도 점수', 'y': '윈도우 개수'},
       color_discrete_sequence=['#FF6B6B']
   )
   
   # 평균선 추가
   mean_score = np.mean(scores)
   fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                 annotation_text=f"평균: {mean_score:.3f}")
   
   return fig

def create_clips_timeline_chart(selected_clips_data):
   """선택된 클립들 타임라인"""
   if not selected_clips_data or 'clips' not in selected_clips_data:
       return None
   
   clips = selected_clips_data['clips']
   
   # 타임라인 차트 데이터 준비
   timeline_data = []
   for clip in clips:
       timeline_data.append({
           'Clip': f"#{clip['rank']}",
           'Start': clip['start_time'],
           'End': clip['end_time'],
           'Score': clip['fun_score'],
           'Duration': clip['duration']
       })
   
   df = pd.DataFrame(timeline_data)
   
   # Gantt 차트 스타일
   fig = px.timeline(
       df,
       x_start="Start",
       x_end="End", 
       y="Clip",
       color="Score",
       title="Selected Clips Timeline",  # 영문으로 변경
       labels={'Start': 'Start Time (sec)', 'End': 'End Time (sec)'},  # 영문으로 변경
       color_continuous_scale='Viridis'
   )
   
   # X축을 초 단위로 명시적 포맷
   fig.update_xaxes(
       title="Time (seconds)",  # 영문으로 변경
       tickformat=".0f"
   )
   
   return fig

def create_cluster_analysis_chart(clusters_data):
   """클러스터 분석 차트"""
   if not clusters_data or 'clusters' not in clusters_data:
       return None
   
   clusters = clusters_data['clusters']
   
   # 클러스터별 정보 수집
   cluster_info = []
   for cluster in clusters:
       cluster_info.append({
           'Cluster ID': cluster['cluster_id'],
           'Points': len(cluster['points']),
           'Duration': cluster['span']['duration'] if 'span' in cluster else 0,
           'Start Time': cluster['span']['start'] if 'span' in cluster else 0,
           'Expanded': cluster.get('is_expanded', False)
       })
   
   df = pd.DataFrame(cluster_info)
   
   # 산점도 차트
   fig = px.scatter(
       df,
       x='Start Time',
       y='Duration',
       size='Points',
       color='Expanded',
       title="클러스터 분포 (시작 시간 vs 지속 시간)",
       labels={'Start Time': '시작 시간(초)', 'Duration': '지속 시간(초)'},
       hover_data=['Cluster ID', 'Points']
   )
   
   return fig

def display_statistics_summary(files_data):
   """통계 요약 표시"""
   st.subheader("📊 처리 통계 요약")
   
   cols = st.columns(4)
   
   if 'scored_windows' in files_data:
       scored_data = load_json_file(files_data['scored_windows'])
       if scored_data:
           with cols[0]:
               st.metric("전체 윈도우", f"{scored_data['metadata']['total_windows']:,}개")
           
           with cols[1]:
               avg_score = scored_data['metadata']['score_statistics']['mean']
               st.metric("평균 재미도", f"{avg_score:.3f}")
   
   if 'selected_clips' in files_data:
       clips_data = load_json_file(files_data['selected_clips'])
       if clips_data:
           with cols[2]:
               st.metric("선별된 클립", f"{len(clips_data['clips'])}개")
           
           with cols[3]:
               avg_clip_score = np.mean([c['fun_score'] for c in clips_data['clips']])
               st.metric("클립 평균 점수", f"{avg_clip_score:.3f}")

def display_clips_table(selected_clips_data):
   """클립 정보 테이블"""
   if not selected_clips_data or 'clips' not in selected_clips_data:
       return
   
   clips = selected_clips_data['clips']
   
   # 테이블 데이터 준비
   table_data = []
   for clip in clips:
       table_data.append({
           '순위': clip['rank'],
           '시작 시간': f"{clip['start_time']:.1f}초",
           '종료 시간': f"{clip['end_time']:.1f}초", 
           '길이': f"{clip['duration']:.1f}초",
           '재미도 점수': f"{clip['fun_score']:.3f}",
           '클러스터 ID': clip.get('cluster_id', 'N/A')
       })
   
   df = pd.DataFrame(table_data)
   
   # 컬러맵 적용
   st.dataframe(
       df,
       use_container_width=True,
       column_config={
           '재미도 점수': st.column_config.ProgressColumn(
               '재미도 점수',
               min_value=0,
               max_value=1,
               format="%.3f"
           )
       }
   )

def main():
   st.title("🎬 재미 클립 분석 대시보드")
   st.markdown("---")
   
   # 사이드바 - 비디오 선택
   st.sidebar.title("📁 비디오 선택")
   
   # 최신 파일들 로드
   all_files = load_latest_files()
   
   if not all_files:
       st.warning("분석 결과 파일을 찾을 수 없습니다.")
       st.info("먼저 clip_extract_pipeline.py를 실행하여 클립을 추출해주세요.")
       return
   
   # 비디오 선택
   video_names = list(all_files.keys())
   selected_video = st.sidebar.selectbox("비디오 선택", video_names)
   
   if selected_video:
       files_data = all_files[selected_video]
       
       st.header(f"📊 {selected_video} 분석 결과")
       
       # 파일 상태 확인
       st.sidebar.subheader("📋 파일 상태")
       file_status = {
           'clusters': '🎪 클러스터링',
           'scored_windows': '📊 점수 계산',
           'selected_clips': '🎯 클립 선별',
           'refined_clips': '🔧 경계 조정'
       }
       
       for file_type, label in file_status.items():
           if file_type in files_data:
               st.sidebar.success(f"{label} ✅")
           else:
               st.sidebar.error(f"{label} ❌")
       
       # 통계 요약
       display_statistics_summary(files_data)
       
       # 텐션 점수와 베스트 썸네일 (상단으로 이동)
       st.subheader("⚡ 영상 분석 결과")
       
       # 배경색 적용을 위한 CSS - 상단에 정의하여 모든 컨테이너에 적용
       st.markdown("""
       <style>
       .metric-container {
           background-color: #1E1E1E;
           border-radius: 10px;
           padding: 10px 15px;
           margin-bottom: 10px;
           height: 100%;
       }
       </style>
       """, unsafe_allow_html=True)
       
       col1, col2 = st.columns([3, 2])  # 점수:썸네일 = 3:2 비율
       
       with col1:
           # 텐션 점수들 계산
           avg_tension = None
           max_tension = None
           
           # 텐션 JSON 직접 찾아서 로드
           tension_pattern = os.path.join("outputs", "tension_data", f"tension_{selected_video}_*.json") 
           tension_files = glob.glob(tension_pattern)
           
           if tension_files:
               latest_tension = max(tension_files, key=os.path.getmtime)
               tension_data = load_json_file(latest_tension)
               if tension_data and 'statistics' in tension_data:
                   avg_tension = tension_data['statistics'].get('avg_tension', 0.0)
                   # 최대 텐션을 다른 방법으로 시도
                   if 'max_tension' in tension_data['statistics']:
                       max_tension = tension_data['statistics']['max_tension']
                   elif 'tension_analysis' in tension_data and 'tension_timeline' in tension_data['tension_analysis']:
                       # tension_timeline에서 최대값 직접 계산
                       timeline = tension_data['tension_analysis']['tension_timeline']
                       if timeline and len(timeline) > 0:
                           max_tension = max(timeline)
           
           # 평균 텐션 점수
           st.markdown('<div class="metric-container">', unsafe_allow_html=True)
           st.markdown("#### :red[평균 텐션 점수]")
           if avg_tension is not None:
               st.markdown(f"<h1 style='color:#4B9FE1; margin:0;'>{avg_tension:.1f}</h1>", unsafe_allow_html=True)
           else:
               st.markdown("데이터 없음")
           st.markdown('</div>', unsafe_allow_html=True)
           
           # 최대 텐션 점수
           if max_tension is not None:
               st.markdown('<div class="metric-container">', unsafe_allow_html=True)
               st.markdown("#### :red[최대 텐션 점수]")
               st.markdown(f"<h2 style='color:#4B9FE1; margin:0;'>{max_tension:.1f}</h2>", unsafe_allow_html=True)
               st.markdown('</div>', unsafe_allow_html=True)
       
       with col2:
           # 최고 confidence 썸네일 표시
           best_thumbnail = None
           best_conf = 0.0
           
           # 썸네일 직접 찾기
           classification_dir = os.path.join("outputs", "classification")
           if os.path.exists(classification_dir):
               for subdir in os.listdir(classification_dir):
                   if selected_video in subdir:
                       thumbnail_dir = os.path.join(classification_dir, subdir)
                       thumbnail_files = glob.glob(os.path.join(thumbnail_dir, "thumbnail_*.jpg"))
                       
                       for thumb_file in thumbnail_files:
                           try:
                               filename = os.path.basename(thumb_file)
                               conf_part = filename.split('conf')[1].split('.jpg')[0]
                               conf_score = float(conf_part)
                               
                               if conf_score > best_conf:
                                   best_conf = conf_score
                                   best_thumbnail = thumb_file
                           except:
                               continue
                       break
           
           # 썸네일 표시 - 높이를 더 크게 조정하고 세로 가운데 정렬
           if best_thumbnail:
               st.markdown('<div class="metric-container">', unsafe_allow_html=True)
               st.markdown("#### :blue[베스트 썸네일]")
               st.caption(f"Confidence: {best_conf:.3f}")
               st.image(best_thumbnail, width=300)  # 썸네일 크기 증가
               st.markdown('</div>', unsafe_allow_html=True)
           else:
               st.info("썸네일을 찾을 수 없습니다.")
       
       st.markdown("---")
       
       # 클립 정보 테이블 (영상 분석 결과 이후로 이동)
       if 'selected_clips' in files_data:
           clips_data = load_json_file(files_data['selected_clips'])
           if clips_data:
               st.subheader("📋 선별된 클립 상세 정보")
               display_clips_table(clips_data)
       
       st.markdown("---")
       
       # 메인 차트들
       col1, col2 = st.columns(2)
       
       with col1:
           st.subheader("📈 재미도 점수 분포")
           if 'scored_windows' in files_data:
               scored_data = load_json_file(files_data['scored_windows'])
               if scored_data:
                   score_chart = create_score_distribution_chart(scored_data)
                   if score_chart:
                       st.plotly_chart(score_chart, use_container_width=True)
               else:
                   st.error("점수 데이터 로드 실패")
           else:
               st.info("점수 데이터가 없습니다.")
       
       with col2:
           st.subheader("🎯 클러스터 분포")
           if 'clusters' in files_data:
               clusters_data = load_json_file(files_data['clusters'])
               if clusters_data:
                   cluster_chart = create_cluster_analysis_chart(clusters_data)
                   if cluster_chart:
                       st.plotly_chart(cluster_chart, use_container_width=True)
               else:
                   st.error("클러스터 데이터 로드 실패")
           else:
               st.info("클러스터 데이터가 없습니다.")
       
       # 기존 시각화 PNG 파일들 표시
       if 'visualizations' in files_data:
           st.subheader("🎨 기존 시각화 결과")
           
           viz_files = files_data['visualizations']
           
           for viz_file in viz_files:
               file_name = os.path.basename(viz_file)
               
               if 'tension_curves' in file_name:
                   st.markdown("### 📈 텐션 곡선")
                   st.image(viz_file, use_column_width=True)
               
               elif 'emotion_distribution' in file_name:
                   st.markdown("### 🎭 감정 분포")
                   st.image(viz_file, use_column_width=True)
               
               elif 'emotion_peak_faces' in file_name:
                   st.markdown("### 😊 감정 피크 얼굴")
                   st.image(viz_file, use_column_width=True)
       
       # 생성된 클립 파일들 확인
       clips_dir = os.path.join("outputs", "funclips", selected_video)
       if os.path.exists(clips_dir):
           clip_files = glob.glob(os.path.join(clips_dir, "*.mp4"))
           if clip_files:
               st.subheader("🎬 생성된 클립 파일")
               
               clip_info = []
               for clip_file in clip_files:
                   file_size = os.path.getsize(clip_file) / (1024 * 1024)  # MB
                   clip_info.append({
                       '파일명': os.path.basename(clip_file),
                       '크기 (MB)': f"{file_size:.1f}",
                       '경로': clip_file
                   })
               
               clip_df = pd.DataFrame(clip_info)
               st.dataframe(clip_df[['파일명', '크기 (MB)']], use_container_width=True)
       
       # JSON 파일 내용 확인 (고급 사용자용)
       with st.expander("🔍 원본 JSON 데이터 확인"):
           json_type = st.selectbox("JSON 타입 선택", 
                                  [key for key in files_data.keys() if key != 'visualizations'])
           
           if json_type and json_type in files_data:
               json_data = load_json_file(files_data[json_type])
               if json_data:
                   st.json(json_data)

if __name__ == "__main__":
   main()