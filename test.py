#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys

def test_opencv_video_reading(video_path):
    """OpenCV 비디오 읽기 테스트"""
    
    print("=" * 50)
    print("🔍 OpenCV 비디오 읽기 디버깅 테스트")
    print("=" * 50)
    
    # 1. 환경 정보
    print(f"📍 현재 작업 디렉토리: {os.getcwd()}")
    print(f"🐍 Python 실행 경로: {sys.executable}")
    print(f"📦 OpenCV 버전: {cv2.__version__}")
    print(f"📁 비디오 파일: {video_path}")
    print(f"📏 파일 크기: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    print()
    
    # 2. 비디오 파일 존재 확인
    if not os.path.exists(video_path):
        print("❌ 비디오 파일이 존재하지 않습니다!")
        return False
    
    # 3. 기본 백엔드 테스트 (torch_video_processor 방식)
    print("🧪 테스트 1: 기본 백엔드 (torch_video_processor 방식)")
    try:
        cap = cv2.VideoCapture(video_path)
        
        # 메타데이터 확인
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"   📊 메타데이터: {fps:.1f} FPS, {frame_count} 프레임, {duration:.1f}초")
        print(f"   🔓 isOpened(): {cap.isOpened()}")
        
        # 첫 10프레임 읽기 테스트
        successful_reads = 0
        failed_reads = 0
        
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                successful_reads += 1
                if i == 0:
                    print(f"   ✅ 첫 번째 프레임 읽기 성공: {frame.shape}")
            else:
                failed_reads += 1
                if i == 0:
                    print(f"   ❌ 첫 번째 프레임 읽기 실패")
                break
        
        print(f"   📈 10프레임 테스트: 성공 {successful_reads}개, 실패 {failed_reads}개")
        cap.release()
        
        if successful_reads > 0:
            print("   🎉 기본 백엔드 성공!")
            return True
        else:
            print("   💥 기본 백엔드 실패!")
    
    except Exception as e:
        print(f"   ⚠️ 기본 백엔드 예외: {e}")
    
    print()
    
    # 4. 명시적 백엔드들 테스트
    print("🧪 테스트 2: 명시적 백엔드들")
    
    backends = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_MSMF, "MSMF"), 
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_GSTREAMER, "GSTREAMER")
    ]
    
    for backend_id, backend_name in backends:
        try:
            print(f"   🔄 {backend_name} ({backend_id}) 테스트...")
            cap = cv2.VideoCapture(video_path, backend_id)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"      ✅ {backend_name} 성공! 프레임: {frame.shape}")
                    cap.release()
                    return True
                else:
                    print(f"      ❌ {backend_name} 열기 성공, 읽기 실패")
            else:
                print(f"      ❌ {backend_name} 열기 실패")
                
            cap.release()
            
        except Exception as e:
            print(f"      ⚠️ {backend_name} 예외: {e}")
    
    print()
    
    # 5. OpenCV 빌드 정보 (간략)
    print("🔧 OpenCV 빌드 정보 (비디오 관련):")
    build_info = cv2.getBuildInformation()
    
    # 비디오 관련 정보만 추출
    lines = build_info.split('\n')
    video_related = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ['ffmpeg', 'gstreamer', 'video', 'media']):
            video_related.append(line.strip())
    
    for info in video_related[:10]:  # 최대 10줄만
        if info:
            print(f"   {info}")
    
    print()
    print("❌ 모든 백엔드로 비디오 읽기 실패!")
    return False


def compare_environments():
    """두 프로젝트 환경 비교"""
    print("=" * 50)
    print("🔄 환경 비교")
    print("=" * 50)
    
    print("현재 환경:")
    print(f"  작업 디렉토리: {os.getcwd()}")
    print(f"  Python 경로: {sys.executable}")
    print(f"  OpenCV 버전: {cv2.__version__}")
    
    # 환경변수 확인
    relevant_env_vars = ['PATH', 'PYTHONPATH', 'OPENCV_LOG_LEVEL', 'OPENCV_FFMPEG_CAPTURE_OPTIONS']
    print("\n관련 환경변수:")
    for var in relevant_env_vars:
        value = os.environ.get(var, "설정되지 않음")
        if var == 'PATH':
            print(f"  {var}: {len(value.split(';'))} 개 경로")
        else:
            print(f"  {var}: {value}")


def main():
    """메인 테스트"""
    # 비디오 파일 경로
    video_path = "D:/my_projects/funny_clip_classify/data/clips/funny/video/video.mp4"
    
    # 환경 비교
    compare_environments()
    print()
    
    # OpenCV 테스트
    success = test_opencv_video_reading(video_path)
    
    print()
    print("=" * 50)
    if success:
        print("✅ 결론: OpenCV로 비디오 읽기 가능!")
        print("   torch_video_processor와 동일하게 동작해야 함")
    else:
        print("❌ 결론: OpenCV로 비디오 읽기 불가능!")
        print("   ffmpeg 전처리가 필요함")
    print("=" * 50)


if __name__ == "__main__":
    main()