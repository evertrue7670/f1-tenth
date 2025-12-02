#!/usr/bin/env python3
"""
트랙 폭 스케일링 스크립트

기능:
1. variable_width_maps_1.0 폴더의 모든 트랙을 읽어옵니다.
2. Centerline의 폭(w_left, w_right)을 지정된 배율로 스케일링합니다.
3. 새로운 Map Image(.png)를 생성합니다.
4. Map Config(.yaml)의 width 값을 업데이트합니다.
5. Raceline은 원본을 그대로 복사합니다.

사용법:
    python scale_track_width.py --scale 1.5 --input_dir variable_width_maps_1.0 --output_dir variable_width_maps_1.5
"""

import argparse
import pathlib
import yaml
import numpy as np
import cv2
import math
import shutil
from typing import Tuple, List

def calculate_yaw(x, y):
    """좌표열에서 yaw(헤딩) 계산"""
    yaws = []
    n = len(x)
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        dx = x[next_idx] - x[prev_idx]
        dy = y[next_idx] - y[prev_idx]
        yaws.append(math.atan2(dy, dx))
    return np.array(yaws)

def calculate_track_boundaries(cx, cy, yaws, w_left, w_right):
    """Centerline과 폭 정보를 이용해 좌우 경계선 좌표 계산"""
    left_bound = []
    right_bound = []
    
    for i in range(len(cx)):
        yaw = yaws[i]
        norm_x = -np.sin(yaw)
        norm_y = np.cos(yaw)
        
        lx = cx[i] + norm_x * w_left[i]
        ly = cy[i] + norm_y * w_left[i]
        left_bound.append([lx, ly])
        
        rx = cx[i] - norm_x * w_right[i]
        ry = cy[i] - norm_y * w_right[i]
        right_bound.append([rx, ry])
        
    return np.array(left_bound), np.array(right_bound)

def process_track(track_name, input_dir, output_dir, scale_factor):
    """트랙 폭을 스케일링하여 새 파일 생성"""
    track_input_dir = input_dir / track_name
    track_output_dir = output_dir / track_name
    
    if not track_input_dir.exists():
        print(f"Error: 트랙을 찾을 수 없습니다: {track_input_dir}")
        return False
    
    print(f"\nProcessing track: {track_name}")
    track_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Map YAML
    yaml_file = track_input_dir / f"{track_name}_map.yaml"
    if not yaml_file.exists():
        print(f"  Error: YAML 파일을 찾을 수 없습니다: {yaml_file}")
        return False
    
    with open(yaml_file, 'r') as f:
        map_config = yaml.safe_load(f)
    
    resolution = map_config['resolution']
    origin = map_config['origin']
    
    # 2. Load Centerline
    centerline_file = track_input_dir / f"{track_name}_centerline.csv"
    if not centerline_file.exists():
        print(f"  Error: Centerline 파일을 찾을 수 없습니다: {centerline_file}")
        return False
    
    # Load centerline data (skip header if exists)
    try:
        cl_data = np.loadtxt(centerline_file, delimiter=',', comments='#')
    except:
        with open(centerline_file, 'r') as f:
            lines = f.readlines()
            data_lines = [l for l in lines if not l.strip().startswith('#')]
            cl_data = np.loadtxt(data_lines, delimiter=',')
    
    if cl_data.shape[1] < 4:
        print(f"  Error: Centerline 파일 형식이 올바르지 않습니다.")
        return False
    
    xs = cl_data[:, 0]
    ys = cl_data[:, 1]
    w_right_orig = cl_data[:, 2]
    w_left_orig = cl_data[:, 3]
    
    # 3. Scale Width
    w_right_scaled = w_right_orig * scale_factor
    w_left_scaled = w_left_orig * scale_factor
    
    # 4. Calculate Properties
    yaws = calculate_yaw(xs, ys)
    left_bound, right_bound = calculate_track_boundaries(xs, ys, yaws, w_left_scaled, w_right_scaled)
    
    # 5. Load Original Map Image for Size Reference
    orig_img_path = track_input_dir / map_config['image']
    if orig_img_path.exists():
        orig_img = cv2.imread(str(orig_img_path), cv2.IMREAD_GRAYSCALE)
        h, w = orig_img.shape
    else:
        # Estimate size from coordinates
        min_x, max_x = np.min(xs) - 10, np.max(xs) + 10
        min_y, max_y = np.min(ys) - 10, np.max(ys) + 10
        w = int((max_x - origin[0]) / resolution) + 100
        h = int((max_y - origin[1]) / resolution) + 100
        print(f"  Warning: Original map image not found. Estimated size: {w}x{h}")
    
    # 6. Draw New Map Image
    new_map_img = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to pixel coordinates
    left_pixels = []
    right_pixels = []
    
    for i in range(len(xs)):
        # World to pixel conversion (Y-axis flip for image coordinates)
        lx_img = int((left_bound[i][0] - origin[0]) / resolution)
        ly_img = int(h - (left_bound[i][1] - origin[1]) / resolution)
        
        rx_img = int((right_bound[i][0] - origin[0]) / resolution)
        ry_img = int(h - (right_bound[i][1] - origin[1]) / resolution)
        
        left_pixels.append([lx_img, ly_img])
        right_pixels.append([rx_img, ry_img])
    
    left_pixels = np.array(left_pixels, dtype=np.int32)
    right_pixels = np.array(right_pixels, dtype=np.int32)
    
    # Create polygon (Left points -> Right points (reversed) -> closed)
    track_poly = np.concatenate([left_pixels, right_pixels[::-1]])
    
    # Fill track interior with white (255 = Free space)
    cv2.fillPoly(new_map_img, [track_poly], 255)
    
    # 7. Save Files
    
    # 7-1. Map Image
    new_img_name = f"{track_name}_map.png"
    cv2.imwrite(str(track_output_dir / new_img_name), new_map_img)
    print(f"  ✓ Map image saved: {new_img_name}")
    
    # 7-2. Centerline CSV
    new_cl_data = np.column_stack((xs, ys, w_right_scaled, w_left_scaled))
    with open(track_output_dir / f"{track_name}_centerline.csv", 'w') as f:
        f.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        np.savetxt(f, new_cl_data, delimiter=',', fmt='%.10f')
    print(f"  ✓ Centerline saved (scaled by {scale_factor}x)")
    
    # 7-3. Map YAML
    old_width = map_config.get('width', 0.0)
    map_config['width'] = float(old_width * scale_factor)
    map_config['image'] = new_img_name
    with open(track_output_dir / f"{track_name}_map.yaml", 'w') as f:
        yaml.dump(map_config, f, sort_keys=False, default_flow_style=False)
    print(f"  ✓ Map YAML saved (width: {old_width:.3f}m -> {map_config['width']:.3f}m)")
    
    # 7-4. Raceline (원본 복사)
    orig_raceline = track_input_dir / f"{track_name}_raceline.csv"
    if orig_raceline.exists():
        shutil.copy2(orig_raceline, track_output_dir / f"{track_name}_raceline.csv")
        print(f"  ✓ Raceline copied from original")
    else:
        print(f"  ⚠ Warning: Original raceline not found, skipping.")
    
    # Print statistics
    avg_width_orig = np.mean(w_left_orig + w_right_orig)
    avg_width_scaled = np.mean(w_left_scaled + w_right_scaled)
    print(f"  - Average width: {avg_width_orig:.3f}m -> {avg_width_scaled:.3f}m")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Scale track width for all tracks")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale factor for track width (default: 1.5)")
    parser.add_argument("--input_dir", type=str, default="variable_width_maps_1.0", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="variable_width_maps_1.5", help="Output directory")
    
    args = parser.parse_args()
    
    script_dir = pathlib.Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    print("="*80)
    print("트랙 폭 스케일링")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor: {args.scale}x")
    print("="*80)
    
    # Find all track directories
    tracks = []
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and not d.name.startswith('.'):
            tracks.append(d.name)
    
    print(f"\nFound {len(tracks)} tracks: {', '.join(tracks)}")
    
    # Process each track
    success_count = 0
    fail_count = 0
    
    for track_name in tracks:
        try:
            if process_track(track_name, input_dir, output_dir, args.scale):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {track_name}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print("\n" + "="*80)
    print("스케일링 완료")
    print("="*80)
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"출력 디렉토리: {output_dir}")

if __name__ == "__main__":
    main()

