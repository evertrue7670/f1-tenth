#!/usr/bin/env python3
"""
트랙 폭 조정 스크립트
실제 차량 크기 대비 트랙 폭 비율을 계산하고, 시뮬레이션에서 동일한 비율을 유지하도록 
변환된 새로운 map yaml, centerline.csv, raceline.csv 파일을 생성합니다.

사용법:
    python adjust_track_width.py --real_car_width 2.0 --real_track_width 12.0 --sim_car_width 0.29 --track Austin --output_dir converted_maps
"""

import argparse
import pathlib
import yaml
import shutil
import numpy as np
from typing import Optional


def calculate_track_width_ratio(real_car_width: float, real_track_width: float) -> float:
    """
    실제 차량/트랙 비율 계산
    
    Parameters
    ----------
    real_car_width : float
        실제 차량 폭 (미터)
    real_track_width : float
        실제 트랙 폭 (미터)
        
    Returns
    -------
    float
        트랙 폭 / 차량 폭 비율
    """
    return real_track_width / real_car_width


def calculate_sim_track_width(sim_car_width: float, ratio: float) -> float:
    """
    시뮬레이션 트랙 폭 계산
    
    Parameters
    ----------
    sim_car_width : float
        시뮬레이션 차량 폭 (미터)
    ratio : float
        실제 트랙/차량 비율
        
    Returns
    -------
    float
        시뮬레이션 트랙 폭 (미터)
    """
    return sim_car_width * ratio


def create_converted_yaml(track_dir: pathlib.Path, output_dir: pathlib.Path, new_width: float) -> bool:
    """
    변환된 YAML 파일 생성
    
    Parameters
    ----------
    track_dir : pathlib.Path
        원본 트랙 디렉토리
    output_dir : pathlib.Path
        출력 디렉토리
    new_width : float
        새로운 트랙 폭 (미터)
        
    Returns
    -------
    bool
        성공 여부
    """
    yaml_file = track_dir / f"{track_dir.name}_map.yaml"
    output_yaml = output_dir / f"{track_dir.name}_map.yaml"
    
    if not yaml_file.exists():
        print(f"  Warning: YAML 파일을 찾을 수 없습니다: {yaml_file}")
        return False
    
    try:
        # 원본 YAML 읽기
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        old_width = data.get('width')
        data['width'] = new_width
        
        # 새 YAML 파일 저장
        with open(output_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ YAML 파일 생성: {output_yaml}")
        print(f"    기존 width: {old_width}m → 새로운 width: {new_width:.3f}m")
        return True
    except Exception as e:
        print(f"  ✗ YAML 파일 생성 실패: {e}")
        return False


def create_converted_centerline(track_dir: pathlib.Path, output_dir: pathlib.Path, new_total_width: float) -> bool:
    """
    변환된 Centerline CSV 파일 생성
    
    Parameters
    ----------
    track_dir : pathlib.Path
        원본 트랙 디렉토리
    output_dir : pathlib.Path
        출력 디렉토리
    new_total_width : float
        새로운 전체 트랙 폭 (w_left + w_right, 미터)
        
    Returns
    -------
    bool
        성공 여부
    """
    centerline_file = track_dir / f"{track_dir.name}_centerline.csv"
    output_centerline = output_dir / f"{track_dir.name}_centerline.csv"
    
    if not centerline_file.exists():
        print(f"  Warning: Centerline 파일을 찾을 수 없습니다: {centerline_file}")
        return False
    
    try:
        # 원본 centerline 읽기 (헤더 포함)
        with open(centerline_file, 'r') as f:
            lines = f.readlines()
        
        # 헤더 찾기
        header_lines = []
        data_lines = []
        for line in lines:
            if line.strip().startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        # 데이터 파싱
        if not data_lines:
            print(f"  ✗ Centerline 파일에 데이터가 없습니다.")
            return False
        
        data = np.loadtxt(data_lines, delimiter=',')
        
        if data.size == 0 or data.shape[1] < 4:
            print(f"  ✗ Centerline 파일 형식이 올바르지 않습니다.")
            return False
        
        # 기존 폭 정보
        old_w_left = data[:, 3]
        old_w_right = data[:, 2]
        old_total = old_w_left + old_w_right
        
        # 새로운 폭 계산 (비율 유지)
        if np.allclose(old_total, old_total[0]):
            # 모든 waypoint가 같은 폭이면 균등 분배
            new_w_left = new_total_width / 2.0
            new_w_right = new_total_width / 2.0
        else:
            # 비율 유지하면서 조정
            ratio = new_total_width / np.mean(old_total)
            new_w_left = old_w_left * ratio
            new_w_right = old_w_right * ratio
        
        # 데이터 업데이트
        data[:, 2] = new_w_right
        data[:, 3] = new_w_left
        
        # 새 파일 저장 (헤더 포함)
        with open(output_centerline, 'w') as f:
            # 헤더 쓰기
            for header_line in header_lines:
                f.write(header_line)
            # 데이터 쓰기
            np.savetxt(f, data, delimiter=',', fmt='%.10f')
        
        print(f"  ✓ Centerline 파일 생성: {output_centerline}")
        print(f"    기존 전체 폭: {np.mean(old_total):.3f}m → 새로운 전체 폭: {new_total_width:.3f}m")
        return True
    except Exception as e:
        print(f"  ✗ Centerline 파일 생성 실패: {e}")
        return False


def create_converted_raceline(track_dir: pathlib.Path, output_dir: pathlib.Path, width_ratio: float) -> bool:
    """
    변환된 Raceline CSV 파일 생성
    
    Parameters
    ----------
    track_dir : pathlib.Path
        원본 트랙 디렉토리
    output_dir : pathlib.Path
        출력 디렉토리
    width_ratio : float
        폭 비율 (new_width / old_width)
        
    Returns
    -------
    bool
        성공 여부
    """
    raceline_file = track_dir / f"{track_dir.name}_raceline.csv"
    output_raceline = output_dir / f"{track_dir.name}_raceline.csv"
    
    if not raceline_file.exists():
        print(f"  Warning: Raceline 파일을 찾을 수 없습니다: {raceline_file}")
        return False
    
    try:
        # 원본 raceline 읽기 (헤더 포함)
        with open(raceline_file, 'r') as f:
            lines = f.readlines()
        
        # 헤더 찾기
        header_lines = []
        data_lines = []
        for line in lines:
            if line.strip().startswith('#'):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        # 데이터 파싱 (delimiter는 세미콜론)
        if not data_lines:
            print(f"  ✗ Raceline 파일에 데이터가 없습니다.")
            return False
        
        data = np.loadtxt(data_lines, delimiter=';')
        
        if data.size == 0 or data.shape[1] < 7:
            print(f"  ✗ Raceline 파일 형식이 올바르지 않습니다. (예상: 7개 컬럼)")
            return False
        
        # Raceline은 좌표(x, y)는 그대로 두고, 속도나 가속도는 필요시 조정 가능
        # 하지만 일반적으로 raceline은 트랙 폭과 직접적인 관계가 없으므로 그대로 복사
        # 필요시 나중에 최적화를 다시 수행해야 할 수 있음
        
        # 새 파일 저장 (헤더 포함)
        with open(output_raceline, 'w') as f:
            # 헤더 쓰기
            for header_line in header_lines:
                f.write(header_line)
            # 데이터 쓰기
            np.savetxt(f, data, delimiter=';', fmt='%.7f')
        
        print(f"  ✓ Raceline 파일 생성: {output_raceline}")
        print(f"    참고: Raceline은 트랙 폭과 직접적인 관계가 없어 원본 그대로 복사되었습니다.")
        print(f"          필요시 트랙 폭에 맞게 최적화를 다시 수행해야 할 수 있습니다.")
        return True
    except Exception as e:
        print(f"  ✗ Raceline 파일 생성 실패: {e}")
        return False


def copy_map_image(track_dir: pathlib.Path, output_dir: pathlib.Path) -> bool:
    """
    맵 이미지 파일 복사
    
    Parameters
    ----------
    track_dir : pathlib.Path
        원본 트랙 디렉토리
    output_dir : pathlib.Path
        출력 디렉토리
        
    Returns
    -------
    bool
        성공 여부
    """
    # 가능한 이미지 파일 확장자
    image_extensions = ['.png', '.pgm', '.jpg', '.jpeg']
    
    for ext in image_extensions:
        image_file = track_dir / f"{track_dir.name}_map{ext}"
        if image_file.exists():
            output_image = output_dir / f"{track_dir.name}_map{ext}"
            shutil.copy2(image_file, output_image)
            print(f"  ✓ 맵 이미지 복사: {output_image}")
            return True
    
    print(f"  Warning: 맵 이미지 파일을 찾을 수 없습니다.")
    return False


def convert_track(track_dir: pathlib.Path, output_dir: pathlib.Path, sim_track_width: float) -> dict:
    """
    트랙 파일 변환 (YAML, Centerline, Raceline)
    
    Parameters
    ----------
    track_dir : pathlib.Path
        원본 트랙 디렉토리
    output_dir : pathlib.Path
        출력 디렉토리
    sim_track_width : float
        시뮬레이션 트랙 폭 (미터)
        
    Returns
    -------
    dict
        변환 결과
    """
    results = {
        'yaml': False,
        'centerline': False,
        'raceline': False,
        'image': False
    }
    
    print(f"\n트랙 '{track_dir.name}' 변환 중...")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # YAML 파일 생성
    results['yaml'] = create_converted_yaml(track_dir, output_dir, sim_track_width)
    
    # Centerline 파일 생성
    results['centerline'] = create_converted_centerline(track_dir, output_dir, sim_track_width)
    
    # Raceline 파일 생성
    results['raceline'] = create_converted_raceline(track_dir, output_dir, 1.0)  # raceline은 그대로 복사
    
    # 맵 이미지 복사
    results['image'] = copy_map_image(track_dir, output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='트랙 폭을 실제 차량/트랙 비율에 맞게 변환하여 새로운 파일 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 실제 F1 차량 폭 2.0m, 트랙 폭 12.0m, 시뮬레이션 차량 폭 0.29m
  python adjust_track_width.py --real_car_width 2.0 --real_track_width 12.0 --sim_car_width 0.29 --track Austin --output_dir converted_maps
  
  # 모든 트랙 변환
  python adjust_track_width.py --real_car_width 2.0 --real_track_width 12.0 --sim_car_width 0.29 --output_dir converted_maps
  
  # 계산만 확인 (dry-run)
  python adjust_track_width.py --real_car_width 2.0 --real_track_width 12.0 --sim_car_width 0.29 --dry-run
        """
    )
    
    parser.add_argument(
        '--real_car_width',
        type=float,
        required=True,
        help='실제 차량 폭 (미터), 예: F1 차량 2.0m'
    )
    
    parser.add_argument(
        '--real_track_width',
        type=float,
        required=True,
        help='실제 트랙 평균 폭 (미터), 예: F1 트랙 12.0m'
    )
    
    parser.add_argument(
        '--sim_car_width',
        type=float,
        default=0.29,
        help='시뮬레이션 차량 폭 (미터), 기본값: 0.29m'
    )
    
    parser.add_argument(
        '--track',
        type=str,
        default=None,
        help='특정 트랙만 변환 (지정하지 않으면 모든 트랙 변환)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='converted_maps',
        help='변환된 파일을 저장할 디렉토리 (기본값: converted_maps)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제로 변환하지 않고 계산 결과만 출력'
    )
    
    args = parser.parse_args()
    
    # 비율 계산
    ratio = calculate_track_width_ratio(args.real_car_width, args.real_track_width)
    sim_track_width = calculate_sim_track_width(args.sim_car_width, ratio)
    
    print("="*80)
    print("트랙 폭 변환 계산")
    print("="*80)
    print(f"실제 차량 폭: {args.real_car_width}m")
    print(f"실제 트랙 평균 폭: {args.real_track_width}m")
    print(f"비율 (트랙/차량): {ratio:.2f}")
    print()
    print(f"시뮬레이션 차량 폭: {args.sim_car_width}m")
    print(f"시뮬레이션 트랙 폭: {sim_track_width:.3f}m")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN] 실제로 변환하지 않습니다.")
        return
    
    # 맵 디렉토리 경로
    script_dir = pathlib.Path(__file__).parent
    map_dir = script_dir / "installation" / "maps"
    
    if not map_dir.exists():
        print(f"Error: 맵 디렉토리를 찾을 수 없습니다: {map_dir}")
        return
    
    # 출력 디렉토리
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n출력 디렉토리: {output_dir.absolute()}")
    
    # 트랙 변환
    if args.track:
        # 특정 트랙만 변환
        track_dir = map_dir / args.track
        if not track_dir.exists():
            print(f"Error: 트랙을 찾을 수 없습니다: {track_dir}")
            return
        
        results = convert_track(track_dir, output_dir, sim_track_width)
        
        print("\n" + "="*80)
        print("변환 완료")
        print("="*80)
        print(f"YAML: {'✓' if results['yaml'] else '✗'}")
        print(f"Centerline: {'✓' if results['centerline'] else '✗'}")
        print(f"Raceline: {'✓' if results['raceline'] else '✗'}")
        print(f"맵 이미지: {'✓' if results['image'] else '✗'}")
    else:
        # 모든 트랙 변환
        print(f"\n모든 트랙 변환 중...")
        
        all_results = {
            'success': 0,
            'failed': 0,
            'tracks': []
        }
        
        for track_dir in sorted(map_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            
            track_name = track_dir.name
            if track_name in ['__pycache__', '.git']:
                continue
            
            yaml_file = track_dir / f"{track_name}_map.yaml"
            if not yaml_file.exists():
                continue
            
            results = convert_track(track_dir, output_dir, sim_track_width)
            
            if results['yaml'] or results['centerline']:
                all_results['success'] += 1
                all_results['tracks'].append(track_name)
            else:
                all_results['failed'] += 1
        
        print("\n" + "="*80)
        print("변환 완료")
        print("="*80)
        print(f"성공: {all_results['success']}개")
        print(f"실패: {all_results['failed']}개")
        if all_results['tracks']:
            print(f"\n변환된 트랙: {', '.join(all_results['tracks'])}")
    
    print("\n참고:")
    print("  - 변환된 파일들이 출력 디렉토리에 생성되었습니다.")
    print("  - 원본 파일은 변경되지 않았습니다.")
    print("  - Raceline은 트랙 폭과 직접적인 관계가 없어 원본 그대로 복사되었습니다.")
    print("    필요시 트랙 폭에 맞게 최적화를 다시 수행해야 할 수 있습니다.")


if __name__ == "__main__":
    main()
