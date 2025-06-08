#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import List, Tuple
import argparse

def sanitize_filename(filename: str) -> str:
    """
    파일명에서 한글 및 특수문자 제거
    
    Args:
        filename (str): 원본 파일명
        
    Returns:
        str: 정리된 파일명
    """
    # 확장자 분리
    name, ext = os.path.splitext(filename)
    
    # 한글, 특수문자 제거하고 영문, 숫자, 언더스코어, 점, 하이픈만 유지
    sanitized = re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)
    
    # 연속된 언더스코어 정리
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # 시작과 끝의 언더스코어 제거
    sanitized = sanitized.strip('_')
    
    # 확장자 다시 붙이기
    return sanitized + ext

def has_korean_or_special_chars(text: str) -> bool:
    """
    텍스트에 한글이나 특수문자가 포함되어 있는지 확인
    
    Args:
        text (str): 확인할 텍스트
        
    Returns:
        bool: 한글이나 특수문자 포함 여부
    """
    # 한글 유니코드 범위: 가-힣, ㄱ-ㅎ, ㅏ-ㅣ
    korean_pattern = r'[가-힣ㄱ-ㅎㅏ-ㅣ]'
    
    # 특수문자 (영문, 숫자, 기본 기호 제외)
    special_pattern = r'[^\w\-_.\s]'
    
    return bool(re.search(korean_pattern, text) or re.search(special_pattern, text))

def rename_files_in_directory(directory: str, dry_run: bool = False, verbose: bool = True) -> Tuple[int, int, List[str]]:
    """
    디렉토리 내 파일들의 한글 파일명 변경
    
    Args:
        directory (str): 처리할 디렉토리 경로
        dry_run (bool): 실제 변경 없이 시뮬레이션만 수행
        verbose (bool): 상세 로그 출력
        
    Returns:
        Tuple[int, int, List[str]]: (성공 수, 실패 수, 에러 메시지들)
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        return 0, 1, [f"디렉토리가 존재하지 않습니다: {directory}"]
    
    success_count = 0
    failed_count = 0
    error_messages = []
    
    # 모든 파일 찾기 (재귀적)
    all_files = list(directory_path.rglob('*'))
    video_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    
    if verbose:
        print(f"📁 처리 대상 디렉토리: {directory}")
        print(f"🎬 발견된 비디오 파일: {len(video_files)}개")
        if dry_run:
            print("🧪 DRY RUN 모드 - 실제 변경하지 않음")
        print("-" * 60)
    
    for file_path in video_files:
        try:
            original_name = file_path.name
            
            # 한글이나 특수문자가 없으면 스킵
            if not has_korean_or_special_chars(original_name):
                if verbose:
                    print(f"⏭️  스킵: {original_name} (변경 불필요)")
                continue
            
            # 새 파일명 생성
            new_name = sanitize_filename(original_name)
            new_path = file_path.parent / new_name
            
            # 중복 파일명 처리
            counter = 1
            original_new_name = new_name
            while new_path.exists() and new_path != file_path:
                name, ext = os.path.splitext(original_new_name)
                new_name = f"{name}_{counter:03d}{ext}"
                new_path = file_path.parent / new_name
                counter += 1
            
            if verbose:
                print(f"📝 변경 예정:")
                print(f"   이전: {original_name}")
                print(f"   이후: {new_name}")
                print(f"   경로: {file_path.parent}")
            
            # 실제 변경 수행 (dry_run이 아닌 경우)
            if not dry_run:
                file_path.rename(new_path)
                success_count += 1
                if verbose:
                    print(f"✅ 변경 완료")
            else:
                success_count += 1
                if verbose:
                    print(f"🧪 시뮬레이션 완료")
            
            if verbose:
                print("-" * 40)
                
        except Exception as e:
            failed_count += 1
            error_msg = f"❌ 실패: {file_path.name} - {str(e)}"
            error_messages.append(error_msg)
            if verbose:
                print(error_msg)
                print("-" * 40)
    
    return success_count, failed_count, error_messages

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='데이터셋 파일명에서 한글 및 특수문자 제거',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python rename_korean_files.py                           # data/clips 전체 처리
  python rename_korean_files.py --directory data/clips/funny  # 특정 폴더만 처리
  python rename_korean_files.py --dry-run                 # 시뮬레이션만 수행
  python rename_korean_files.py --quiet                   # 간단한 로그만 출력
        """
    )
    
    parser.add_argument(
        '--directory', '-d',
        default='data/clips',
        help='처리할 디렉토리 경로 (기본: data/clips)'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='실제 변경 없이 시뮬레이션만 수행'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='간단한 결과만 출력 (상세 로그 비활성화)'
    )
    
    args = parser.parse_args()
    
    print("🎬 한글 파일명 정리 스크립트")
    print("=" * 50)
    
    # 파일명 변경 실행
    success_count, failed_count, error_messages = rename_files_in_directory(
        directory=args.directory,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )
    
    # 결과 요약
    print("\n📊 처리 결과 요약:")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {failed_count}개")
    
    if error_messages:
        print(f"\n🚨 오류 목록:")
        for error in error_messages:
            print(f"  {error}")
    
    if args.dry_run:
        print(f"\n🧪 시뮬레이션 완료! 실제 적용하려면 --dry-run 옵션을 제거하세요.")
    else:
        print(f"\n🎯 파일명 정리 완료!")
    
    # 예시 보여주기
    if success_count > 0 and not args.quiet:
        print(f"\n💡 변경 예시:")
        print(f"  이전: f_022_시청자_취미_구경하기_2523.0_2558.0.mp4")
        print(f"  이후: f_022______2523.0_2558.0.mp4")

if __name__ == "__main__":
    main()