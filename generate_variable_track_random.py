#!/usr/bin/env python3
"""
F1TENTH 가변 폭 트랙 생성기

기능:
1. 기존 트랙의 Centerline을 읽어옵니다.
2. 사용자가 지정한 평균/최소/최대 폭 범위 내에서 부드럽게 변하는 트랙 폭을 생성합니다.
3. 새로운 Centerline, Map Image(.png), Map Config(.yaml)를 생성합니다.
   (Raceline은 원본을 그대로 복사합니다)

사용법:
    python generate_variable_track.py --track Catalunya --avg_width 1.5 --min_width 1.0 --max_width 2.0 --output_dir my_maps
"""

import argparse
import pathlib
import yaml
import numpy as np
import cv2
import math
from typing import Tuple, List, Optional

def generate_smooth_width_profile(n_points: int, avg_w: float, min_w: float, max_w: float, noise_scale: float = 1.0) -> np.ndarray:
    """
    부드럽게 변하는 트랙 폭 프로파일 생성 (여러 주파수의 사인파 합성)
    """
    # 0 ~ 2pi 사이의 위상
    t = np.linspace(0, 4 * np.pi, n_points)
    
    # 여러 주파수의 사인파 합성으로 랜덤성 부여
    # 저주파 성분이 주를 이뤄야 트랙 폭이 급격하게 변하지 않음
    noise = np.zeros(n_points)
    frequencies = [1, 2, 3, 5]
    amplitudes = [0.5, 0.3, 0.2, 0.1]
    phases = np.random.rand(len(frequencies)) * 2 * np.pi
    
    for f, a, p in zip(frequencies, amplitudes, phases):
        noise += a * np.sin(f * t + p)
    
    # 정규화 (-1 ~ 1)
    noise = noise / np.max(np.abs(noise))
    
    # 목표 범위로 스케일링
    # variation: 평균을 중심으로 (max-min) 범위 내에서 변동
    variation_range = (max_w - min_w) / 2.0
    width_profile = avg_w + noise * variation_range * noise_scale
    
    # Hard clamp
    width_profile = np.clip(width_profile, min_w, max_w)
    
    return width_profile

def world_to_pixel(x: float, y: float, origin: List[float], resolution: float, img_height: int) -> Tuple[int, int]:
    """
    월드 좌표(m)를 픽셀 좌표로 변환
    origin: [x, y, theta]
    """
    ox, oy = origin[0], origin[1]
    
    # 맵 원점 기준으로 변환
    px = (x - ox) / resolution
    py = (y - oy) / resolution
    
    # Y축 뒤집기 (이미지 좌표계는 위에서 아래로 증가, 일반 좌표계는 아래에서 위로 증가)
    # 하지만 F1TENTH 맵의 경우 origin이 보통 왼쪽 아래이므로, 단순히 스케일링만 하면 됨
    # 단, 이미지 라이브러리에 따라 y축 방향이 다를 수 있음. 
    # 여기서는 map.yaml의 origin이 bottom-left라고 가정.
    
    return int(px), int(py)

def calculate_track_boundaries(cx, cy, yaws, w_left, w_right):
    """
    Centerline과 폭 정보를 이용해 좌우 경계선 좌표 계산
    """
    left_bound = []
    right_bound = []
    
    for i in range(len(cx)):
        yaw = yaws[i]
        # 법선 벡터 방향 (yaw + 90도)
        norm_x = -np.sin(yaw)
        norm_y = np.cos(yaw)
        
        # 왼쪽 경계
        lx = cx[i] + norm_x * w_left[i]
        ly = cy[i] + norm_y * w_left[i]
        left_bound.append([lx, ly])
        
        # 오른쪽 경계 (반대 방향)
        rx = cx[i] - norm_x * w_right[i]
        ry = cy[i] - norm_y * w_right[i]
        right_bound.append([rx, ry])
        
    return np.array(left_bound), np.array(right_bound)

def calculate_yaw(x, y):
    """
    좌표열에서 yaw(헤딩) 계산
    """
    yaws = []
    n = len(x)
    for i in range(n):
        # 전후 포인트를 이용해 접선 방향 계산
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        dx = x[next_idx] - x[prev_idx]
        dy = y[next_idx] - y[prev_idx]
        yaws.append(math.atan2(dy, dx))
    return np.array(yaws)

def calculate_s(x, y):
    """
    누적 거리(s) 계산
    """
    s = [0.0]
    for i in range(1, len(x)):
        dist = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        s.append(s[-1] + dist)
    return np.array(s)

def process_track(track_name, map_dir, output_dir, avg_w, min_w, max_w):
    track_dir = map_dir / track_name
    if not track_dir.exists():
        print(f"Error: 트랙을 찾을 수 없습니다: {track_dir}")
        return

    print(f"Processing track: {track_name}")
    output_subdir = output_dir / track_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    # 1. Load Original YAML
    yaml_file = track_dir / f"{track_name}_map.yaml"
    with open(yaml_file, 'r') as f:
        map_config = yaml.safe_load(f)
    
    resolution = map_config['resolution']
    origin = map_config['origin']
    
    # 2. Load Centerline
    centerline_file = track_dir / f"{track_name}_centerline.csv"
    if not centerline_file.exists():
        print("Error: Centerline file not found.")
        return
    
    # Load centerline data (skip header if exists)
    try:
        cl_data = np.loadtxt(centerline_file, delimiter=',', comments='#')
    except:
        # 헤더가 있는 경우 처리
        with open(centerline_file, 'r') as f:
            lines = f.readlines()
            data_lines = [l for l in lines if not l.strip().startswith('#')]
            cl_data = np.loadtxt(data_lines, delimiter=',')

    xs = cl_data[:, 0]
    ys = cl_data[:, 1]
    n_points = len(xs)
    
    # 3. Generate Variable Width
    # 전체 폭 생성
    width_profile = generate_smooth_width_profile(n_points, avg_w, min_w, max_w)
    
    # 좌우로 배분 (중앙 주행 가정)
    w_left = width_profile / 2.0
    w_right = width_profile / 2.0
    
    # 4. Calculate Properties (Yaw, s, Boundaries)
    yaws = calculate_yaw(xs, ys)
    ss = calculate_s(xs, ys)
    left_bound, right_bound = calculate_track_boundaries(xs, ys, yaws, w_left, w_right)
    
    # 5. Draw New Map Image
    # 기존 맵 이미지 로드하여 크기 참조
    orig_img_path = track_dir / map_config['image']
    if orig_img_path.exists():
        orig_img = cv2.imread(str(orig_img_path), cv2.IMREAD_GRAYSCALE)
        h, w = orig_img.shape
    else:
        # 이미지가 없으면 좌표 범위를 보고 추정 (여유 있게)
        min_x, max_x = np.min(xs) - 10, np.max(xs) + 10
        min_y, max_y = np.min(ys) - 10, np.max(ys) + 10
        w = int((max_x - origin[0]) / resolution) + 100
        h = int((max_y - origin[1]) / resolution) + 100
        print(f"Warning: Original map image not found. Estimated size: {w}x{h}")

    # 새 캔버스 생성 (검은색 배경 = Occupied)
    # F1TENTH Gym: 0(Black) = Occupied, 255(White) = Free
    # 주의: Occupancy Grid에서는 보통 0이 Free, 100이 Occupied인 경우도 있지만
    # F1TENTH Gym map loader는: image pixel 255 -> free, 0 -> occupied 로 처리함 (track.py 참조)
    new_map_img = np.zeros((h, w), dtype=np.uint8)
    
    # 픽셀 좌표로 변환
    left_pixels = []
    right_pixels = []
    
    for i in range(n_points):
        lx, ly = world_to_pixel(left_bound[i][0], left_bound[i][1], origin, resolution, h)
        rx, ry = world_to_pixel(right_bound[i][0], right_bound[i][1], origin, resolution, h)
        
        # Y축 좌표계 확인: 이미지는 (0,0)이 Top-Left.
        # map.yaml의 origin이 Bottom-Left 기준이라면 y좌표 뒤집어야 함.
        # F1TENTH Gym의 일반적인 map.pgm은 (0,0)이 Bottom-Left 기준의 좌표계와 매핑될 때
        # 이미지를 로드할 때 FLIP_TOP_BOTTOM을 수행함 (track.py Line 124).
        # 따라서 여기서는 이미지를 그냥 '바르게' 그리면 됨 (좌표 변환시 y축 반전 없이).
        # 단, cv2는 (0,0)이 Top-Left이므로, world y가 증가할수록 pixel y는 감소해야 하는지 확인 필요.
        # 보통 map_server 형식은 origin이 bottom-left이고, 이미지는 그대로 저장됨.
        # Gym에서 Load할 때 Flip 하므로, 우리는 Flip 되기 전 상태인 "원본 이미지 좌표계"에 맞춰야 함.
        # 원본 이미지는 (0,0) Top-Left 기준으로 그려져 있음.
        # World (x,y) -> Image (u,v):
        # u = (x - origin_x) / resolution
        # v = height - (y - origin_y) / resolution  <-- Y축 뒤집기 필요
        
        lx_img = int((left_bound[i][0] - origin[0]) / resolution)
        ly_img = int(h - (left_bound[i][1] - origin[1]) / resolution)
        
        rx_img = int((right_bound[i][0] - origin[0]) / resolution)
        ry_img = int(h - (right_bound[i][1] - origin[1]) / resolution)
        
        left_pixels.append([lx_img, ly_img])
        right_pixels.append([rx_img, ry_img])
        
    left_pixels = np.array(left_pixels, dtype=np.int32)
    right_pixels = np.array(right_pixels, dtype=np.int32)
    
    # Polygon 생성 (Left 점들 -> Right 점들(역순) -> 닫힘)
    track_poly = np.concatenate([left_pixels, right_pixels[::-1]])
    
    # 트랙 내부를 흰색(255, Free)으로 채움
    cv2.fillPoly(new_map_img, [track_poly], 255)
    
    # 6. Save Files
    
    # 6-1. Map Image
    new_img_name = f"{track_name}_map.png"
    cv2.imwrite(str(output_subdir / new_img_name), new_map_img)
    
    # 6-2. Centerline CSV
    # Format: x, y, w_tr_right, w_tr_left
    new_cl_data = np.column_stack((xs, ys, w_right, w_left))
    with open(output_subdir / f"{track_name}_centerline.csv", 'w') as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        np.savetxt(f, new_cl_data, delimiter=',', fmt='%.10f')
        
    # 6-3. Raceline CSV
    # 사용자의 요청에 따라 Raceline은 새로 생성하지 않고 원본을 그대로 복사합니다.
    # 단, 폭 변화로 인해 원본 Raceline이 벽과 충돌할 가능성이 있으니 주의가 필요합니다.
    orig_raceline = track_dir / f"{track_name}_raceline.csv"
    if orig_raceline.exists():
        import shutil
        shutil.copy2(orig_raceline, output_subdir / f"{track_name}_raceline.csv")
        print(f"  Copied original raceline to {output_subdir}")
    else:
        print("  Warning: Original raceline not found, skipping.")

    # 6-4. Map YAML
    map_config['width'] = float(avg_w) # 대표 폭은 평균값으로
    map_config['image'] = new_img_name
    with open(output_subdir / f"{track_name}_map.yaml", 'w') as f:
        yaml.dump(map_config, f, sort_keys=False)

    print(f"  Saved to {output_subdir}")
    print(f"  - Width stats: Avg {np.mean(width_profile):.2f}m, Min {np.min(width_profile):.2f}m, Max {np.max(width_profile):.2f}m")

def main():
    parser = argparse.ArgumentParser(description="Generate variable width tracks for F1TENTH")
    parser.add_argument("--track", type=str, default=None, help="Target track name (e.g., Catalunya)")
    parser.add_argument("--avg_width", type=float, default=1.5, help="Target average width (m)")
    parser.add_argument("--min_width", type=float, default=1.0, help="Minimum width (m)")
    parser.add_argument("--max_width", type=float, default=2.0, help="Maximum width (m)")
    parser.add_argument("--output_dir", type=str, default="variable_width_maps", help="Output directory")
    
    args = parser.parse_args()
    
    script_dir = pathlib.Path(__file__).parent
    map_dir = script_dir / "installation" / "maps"
    output_dir = pathlib.Path(args.output_dir)
    
    if args.track:
        process_track(args.track, map_dir, output_dir, args.avg_width, args.min_width, args.max_width)
    else:
        # Process all tracks
        for track_dir in map_dir.iterdir():
            if track_dir.is_dir() and not track_dir.name.startswith('.'):
                try:
                    process_track(track_dir.name, map_dir, output_dir, args.avg_width, args.min_width, args.max_width)
                except Exception as e:
                    print(f"Skipping {track_dir.name}: {e}")

if __name__ == "__main__":
    main()

