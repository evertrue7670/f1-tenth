#!/usr/bin/env python3
"""
트랙 폭 확인 스크립트
각 맵의 트랙 폭 정보를 확인하고 표로 출력합니다.

중요 설명:
1. Centerline 폭 (w_tr_left + w_tr_right):
   - 모든 트랙에서 2.20m로 고정되어 있음 (README: "fixed track width of 2,20m")
   - 스무딩 과정에서 모든 트랙의 폭을 2.20m로 통일함

2. YAML width:
   - 원본 트랙 데이터에서 추출한 실제 트랙 폭 정보
   - 트랙마다 다름 (예: Montreal 1.41m, Austin 2.03m, BrandsHatch 2.56m)
   - Centerline 폭(2.20m)과 차이가 나는 이유:
     * Centerline은 스무딩 과정에서 2.20m로 고정됨
     * YAML width는 원본 트랙의 실제 폭 정보를 보존
     * 따라서 YAML width가 centerline 폭보다 작거나 클 수 있음
"""

import pathlib
import yaml
import numpy as np
from typing import Dict, Optional, Tuple


def get_track_widths(map_dir: pathlib.Path) -> Dict[str, Dict]:
    """
    모든 트랙의 폭 정보를 수집합니다.
    
    Parameters
    ----------
    map_dir : pathlib.Path
        맵 디렉토리 경로
        
    Returns
    -------
    Dict[str, Dict]
        트랙 이름을 키로 하는 폭 정보 딕셔너리
    """
    results = {}
    
    # 맵 디렉토리 내의 모든 하위 디렉토리 확인
    for track_dir in sorted(map_dir.iterdir()):
        if not track_dir.is_dir():
            continue
            
        track_name = track_dir.name
        
        # 파일이나 특수 디렉토리 제외
        if track_name in ['__pycache__', '.git']:
            continue
            
        track_info = {
            'yaml_width': None,
            'centerline_avg_width': None,
            'centerline_min_width': None,
            'centerline_max_width': None,
            'centerline_std_width': None,
            'has_centerline': False,
        }
        
        # YAML 파일에서 width 읽기
        yaml_file = track_dir / f"{track_name}_map.yaml"
        if yaml_file.exists():
            try:
                with open(yaml_file, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'width' in yaml_data:
                        track_info['yaml_width'] = float(yaml_data['width'])
            except Exception as e:
                print(f"Warning: Could not read {yaml_file}: {e}")
        
        # Centerline 파일에서 폭 정보 읽기
        centerline_file = track_dir / f"{track_name}_centerline.csv"
        if centerline_file.exists():
            try:
                # CSV 파일 읽기 (주석 라인 제외)
                data = np.loadtxt(centerline_file, delimiter=',', comments='#')
                
                if data.size > 0 and data.shape[1] >= 4:
                    # w_tr_right_m, w_tr_left_m 컬럼 (인덱스 2, 3)
                    w_right = data[:, 2]
                    w_left = data[:, 3]
                    
                    # 전체 폭 = 왼쪽 폭 + 오른쪽 폭
                    total_widths = w_left + w_right
                    
                    track_info['centerline_avg_width'] = float(np.mean(total_widths))
                    track_info['centerline_min_width'] = float(np.min(total_widths))
                    track_info['centerline_max_width'] = float(np.max(total_widths))
                    track_info['centerline_std_width'] = float(np.std(total_widths))
                    track_info['has_centerline'] = True
            except Exception as e:
                print(f"Warning: Could not read {centerline_file}: {e}")
        
        results[track_name] = track_info
    
    return results


def print_results(results: Dict[str, Dict]):
    """
    결과를 표 형식으로 출력합니다.
    
    Parameters
    ----------
    results : Dict[str, Dict]
        트랙 폭 정보 딕셔너리
    """
    print("\n" + "="*100)
    print("트랙 폭 정보 요약")
    print("="*100)
    print("설명:")
    print("  - YAML Width: 원본 트랙 데이터에서 추출한 실제 트랙 폭 (미터)")
    print("    → 트랙마다 다르며, centerline 스무딩 전 원본 정보")
    print("  - Centerline 폭: w_tr_left + w_tr_right (모든 트랙에서 2.20m로 고정)")
    print("    → 스무딩 과정에서 모든 트랙의 폭을 2.20m로 통일")
    print("  ⚠️  주의: YAML width와 Centerline 폭이 다를 수 있음 (원본 vs 고정값)")
    print("="*100)
    print(f"{'트랙 이름':<20} {'YAML Width':<12} {'Centerline 평균':<15} {'Centerline 최소':<15} {'Centerline 최대':<15} {'표준편차':<10}")
    print("-"*100)
    
    for track_name, info in sorted(results.items()):
        yaml_w = f"{info['yaml_width']:.2f}" if info['yaml_width'] is not None else "N/A"
        avg_w = f"{info['centerline_avg_width']:.2f}" if info['centerline_avg_width'] is not None else "N/A"
        min_w = f"{info['centerline_min_width']:.2f}" if info['centerline_min_width'] is not None else "N/A"
        max_w = f"{info['centerline_max_width']:.2f}" if info['centerline_max_width'] is not None else "N/A"
        std_w = f"{info['centerline_std_width']:.2f}" if info['centerline_std_width'] is not None else "N/A"
        
        print(f"{track_name:<20} {yaml_w:<12} {avg_w:<15} {min_w:<15} {max_w:<15} {std_w:<10}")
    
    print("="*100)
    
    # 통계 요약
    print("\n통계 요약:")
    yaml_widths = [info['yaml_width'] for info in results.values() if info['yaml_width'] is not None]
    centerline_widths = [info['centerline_avg_width'] for info in results.values() 
                        if info['centerline_avg_width'] is not None]
    
    if yaml_widths:
        print(f"YAML Width - 평균: {np.mean(yaml_widths):.2f}m, 최소: {np.min(yaml_widths):.2f}m, 최대: {np.max(yaml_widths):.2f}m")
    
    if centerline_widths:
        print(f"Centerline 평균 폭 - 평균: {np.mean(centerline_widths):.2f}m, 최소: {np.min(centerline_widths):.2f}m, 최대: {np.max(centerline_widths):.2f}m")
    
    print(f"\n총 {len(results)}개 트랙 확인 완료")
    print(f"  - Centerline 파일 있는 트랙: {sum(1 for info in results.values() if info['has_centerline'])}개")


def main():
    """메인 함수"""
    # 맵 디렉토리 경로 설정
    script_dir = pathlib.Path(__file__).parent
    map_dir = script_dir / "installation" / "maps"
    
    if not map_dir.exists():
        print(f"Error: 맵 디렉토리를 찾을 수 없습니다: {map_dir}")
        return
    
    print(f"맵 디렉토리: {map_dir}")
    print("트랙 폭 정보 수집 중...")
    
    # 트랙 폭 정보 수집
    results = get_track_widths(map_dir)
    
    # 결과 출력
    print_results(results)


if __name__ == "__main__":
    main()

