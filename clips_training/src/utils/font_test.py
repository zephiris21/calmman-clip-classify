#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

def test_korean_fonts():
    """한글 폰트 테스트 및 설정"""
    print("🔧 한글 폰트 진단 시작")
    print("=" * 50)
    
    # 1. 현재 matplotlib 설정 확인
    print("1️⃣ 현재 matplotlib 설정:")
    print(f"   font.family: {plt.rcParams['font.family']}")
    print(f"   font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"   axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")
    
    # 2. 시스템에서 사용 가능한 한글 폰트 찾기
    print("\n2️⃣ 시스템 한글 폰트 검색:")
    korean_fonts = []
    korean_keywords = ['malgun', '맑은', 'nanum', '나눔', 'gothic', 'batang', '바탕']
    
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if any(keyword in font_name for keyword in korean_keywords):
            korean_fonts.append({
                'name': font.name,
                'path': font.fname,
                'style': font.style
            })
    
    # 중복 제거 및 정렬
    unique_fonts = {}
    for font in korean_fonts:
        if font['name'] not in unique_fonts:
            unique_fonts[font['name']] = font
    
    korean_fonts = list(unique_fonts.values())
    korean_fonts.sort(key=lambda x: x['name'])
    
    if korean_fonts:
        print(f"   발견된 한글 폰트: {len(korean_fonts)}개")
        for i, font in enumerate(korean_fonts[:5]):  # 상위 5개만 출력
            print(f"   {i+1}. {font['name']} ({font['style']})")
            print(f"      경로: {font['path']}")
        if len(korean_fonts) > 5:
            print(f"   ... 외 {len(korean_fonts)-5}개")
    else:
        print("   ❌ 한글 폰트를 찾을 수 없습니다!")
    
    # 3. Windows 기본 폰트 경로 확인
    print("\n3️⃣ Windows 기본 폰트 경로 확인:")
    windows_fonts = [
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/malgunbd.ttf',
        'C:/Windows/Fonts/gulim.ttc',
        'C:/Windows/Fonts/batang.ttc'
    ]
    
    available_windows_fonts = []
    for font_path in windows_fonts:
        if os.path.exists(font_path):
            available_windows_fonts.append(font_path)
            print(f"   ✅ {os.path.basename(font_path)}: {font_path}")
        else:
            print(f"   ❌ {os.path.basename(font_path)}: 없음")
    
    # 4. 최적 폰트 선택 및 설정
    print("\n4️⃣ 최적 폰트 설정:")
    
    best_font = None
    if korean_fonts:
        # 맑은고딕 우선 선택
        for font in korean_fonts:
            if 'malgun' in font['name'].lower():
                best_font = font['name']
                break
        
        if not best_font:
            best_font = korean_fonts[0]['name']
    
    if best_font:
        print(f"   선택된 폰트: {best_font}")
        
        # 폰트 설정 적용
        plt.rcParams['font.family'] = [best_font, 'DejaVu Sans']
        plt.rcParams['font.sans-serif'] = [best_font, 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("   ✅ 폰트 설정 적용 완료")
        
    elif available_windows_fonts:
        # 직접 경로로 폰트 설정
        font_path = available_windows_fonts[0]
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        
        print(f"   직접 경로 설정: {font_name}")
        print(f"   폰트 파일: {font_path}")
        print("   ✅ 직접 폰트 설정 완료")
        
        best_font = font_name
    else:
        print("   ❌ 사용 가능한 한글 폰트가 없습니다!")
        return None
    
    return best_font

def test_korean_text_rendering(font_name=None):
    """한글 텍스트 렌더링 테스트"""
    print("\n5️⃣ 한글 텍스트 렌더링 테스트:")
    
    test_texts = [
        "안녕하세요",
        "클래스 분포",
        "특징명",
        "재미도 분석",
        "Boring vs Funny"
    ]
    
    # 간단한 테스트 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 테스트 데이터
    categories = ['Boring', 'Funny']
    values = [48, 52]
    colors = ['#4ECDC4', '#FF6B6B']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # 한글 텍스트 적용
    ax.set_title('🎭 침착맨 클립 재미도 분포', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('클래스', fontsize=12)
    ax.set_ylabel('샘플 수', fontsize=12)
    
    # 막대 위에 값 표시
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}개', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 파일로 저장 (한글 깨짐 확인용)
    try:
        save_path = 'font_test_result.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"   ✅ 테스트 그래프 저장: {save_path}")
        
        # 화면에 표시
        plt.show()
        
        print("   📊 그래프에서 한글이 제대로 표시되는지 확인하세요!")
        
    except Exception as e:
        print(f"   ❌ 그래프 저장 실패: {e}")
        plt.show()
    
    # 개별 텍스트 렌더링 테스트
    print("\n   개별 텍스트 렌더링 테스트:")
    for text in test_texts:
        try:
            # 텍스트 렌더링 시도
            fig_test, ax_test = plt.subplots(figsize=(6, 2))
            ax_test.text(0.5, 0.5, text, ha='center', va='center', 
                        fontsize=14, transform=ax_test.transAxes)
            ax_test.set_xlim(0, 1)
            ax_test.set_ylim(0, 1)
            ax_test.axis('off')
            plt.close(fig_test)  # 화면에 표시하지 않고 테스트만
            print(f"      ✅ '{text}' - 렌더링 성공")
        except Exception as e:
            print(f"      ❌ '{text}' - 렌더링 실패: {e}")

def get_font_recommendations():
    """폰트 문제 해결 권장사항"""
    print("\n6️⃣ 폰트 문제 해결 권장사항:")
    print("=" * 50)
    
    print("🔧 방법 1: matplotlib 캐시 초기화")
    print("   import matplotlib")
    print("   matplotlib.font_manager._rebuild()")
    print("   # 그 후 Python 재시작")
    
    print("\n🔧 방법 2: 직접 폰트 파일 지정")
    print("   import matplotlib.font_manager as fm")
    print("   font_path = 'C:/Windows/Fonts/malgun.ttf'")
    print("   font_prop = fm.FontProperties(fname=font_path)")
    print("   plt.rcParams['font.family'] = font_prop.get_name()")
    
    print("\n🔧 방법 3: 나눔고딕 설치")
    print("   1. https://hangeul.naver.com/font 에서 나눔고딕 다운로드")
    print("   2. 설치 후 Python 재시작")
    print("   3. plt.rcParams['font.family'] = 'NanumGothic'")
    
    print("\n🔧 방법 4: 영어 라벨 사용 (임시)")
    print("   한글 대신 영어로 라벨 작성")
    print("   예: '클래스' → 'Class', '재미도' → 'Fun Score'")

def main():
    """메인 실행"""
    print("🚀 한글 폰트 진단 도구")
    print("이 도구는 matplotlib에서 한글 폰트 문제를 진단합니다.")
    print()
    
    try:
        # 1. 한글 폰트 테스트
        best_font = test_korean_fonts()
        
        if best_font:
            # 2. 텍스트 렌더링 테스트
            test_korean_text_rendering(best_font)
        
        # 3. 권장사항 출력
        get_font_recommendations()
        
        print("\n✅ 진단 완료!")
        if best_font:
            print(f"💡 권장 설정:")
            print(f"   plt.rcParams['font.family'] = '{best_font}'")
            print(f"   plt.rcParams['axes.unicode_minus'] = False")
        else:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 권장사항을 참고하세요.")
            
    except Exception as e:
        print(f"❌ 진단 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()