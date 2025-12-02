#!/usr/bin/env python3
"""
F1TENTH Raceline Optimizer

기능:
1. F1TENTH dynamic.yaml 설정을 로드하여 racecar.ini에 반영합니다.
2. 지정된 트랙의 Centerline을 가져와 최적화 알고리즘(Minimum Time)을 실행합니다.
3. 최적화된 Raceline을 F1TENTH 호환 포맷(raceline.csv)으로 저장합니다.

사용법:
    python optimize_raceline.py --track Catalunya
"""

import os
import sys
import shutil
import yaml
import configparser
import json
import pandas as pd
import numpy as np
import argparse
import importlib.util

# 경로 설정
F1TENTH_ROOT = os.path.dirname(os.path.abspath(__file__))
OPT_ROOT = os.path.join(F1TENTH_ROOT, "global_racetrajectory_optimization-master")
INPUT_TRACKS_DIR = os.path.join(OPT_ROOT, "inputs", "tracks")
PARAMS_DIR = os.path.join(OPT_ROOT, "params")
OUTPUT_DIR = os.path.join(OPT_ROOT, "outputs")

def load_dynamic_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def update_racecar_ini(ini_path, dyn_config):
    """dynamic.yaml 값을 바탕으로 racecar.ini 업데이트"""
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    
    # veh_params 업데이트
    veh_params = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
    veh_params['mass'] = dyn_config['m']
    veh_params['length'] = dyn_config['length']
    veh_params['width'] = dyn_config['width']
    # v_max 등 다른 파라미터도 필요시 매핑 (여기서는 기본값 유지 또는 dynamic.yaml에 있다면 사용)
    # veh_params['v_max'] = dyn_config.get('v_max', 20.0) # 예시
    
    parser.set('GENERAL_OPTIONS', 'veh_params', json.dumps(veh_params))
    
    # optimization options 업데이트 (mintime)
    optim_opts = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
    optim_opts['width_opt'] = dyn_config['width'] + 0.1 # 안전 여유 10cm
    optim_opts['mue'] = dyn_config['mu']
    
    parser.set('OPTIMIZATION_OPTIONS', 'optim_opts_mintime', json.dumps(optim_opts))
    
    # vehicle_params_mintime 업데이트
    veh_mintime = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
    veh_mintime['wheelbase_front'] = dyn_config['length_f']
    veh_mintime['wheelbase_rear'] = dyn_config['length_r']
    # track_width는 dynamic.yaml에 없으면 차량 폭으로 근사
    veh_mintime['track_width_front'] = dyn_config['width'] * 0.8 
    veh_mintime['track_width_rear'] = dyn_config['width'] * 0.8
    veh_mintime['cog_z'] = dyn_config['h']
    veh_mintime['I_z'] = dyn_config['I']
    
    parser.set('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime', json.dumps(veh_mintime))
    
    # tire_params_mintime 업데이트
    tire_mintime = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'tire_params_mintime'))
    # B, C, E 파라미터는 Pacejka 매직 포뮬러 계수인데, dynamic.yaml의 C_Sf, C_Sr과는 다름.
    # 단순 변환 불가하므로 여기서는 기본값 유지하거나, 필요시 별도 튜닝 필요.
    # dynamic.yaml의 C_Sf는 cornering stiffness.
    
    with open(ini_path, 'w') as f:
        parser.write(f)
    print(f"Updated {ini_path} with parameters from dynamic.yaml")

def prepare_track_file(track_name, variable_maps_dir):
    """variable_width_maps의 centerline을 optimization용 input으로 변환"""
    src_path = os.path.join(variable_maps_dir, track_name, f"{track_name}_centerline.csv")
    dst_name = f"{track_name.lower()}_var"
    dst_path = os.path.join(INPUT_TRACKS_DIR, f"{dst_name}.csv")
    
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source centerline not found: {src_path}")
        
    # Load and remove header
    try:
        df = pd.read_csv(src_path, comment='#', header=None)
        # format: x, y, w_right, w_left (F1TENTH gym)
        # optimization input: x_m, y_m, w_tr_right_m, w_tr_left_m (동일 순서)
        
        # 데이터가 4열인지 확인
        if df.shape[1] < 4:
             # 헤더가 있을 수 있으므로 다시 시도
             df = pd.read_csv(src_path, comment='#')
             if df.shape[1] < 4:
                 raise ValueError(f"Invalid centerline format: {src_path}")
    except Exception as e:
        print(f"Error reading csv: {e}")
        # 직접 텍스트 파싱 시도 (헤더 건너뛰기)
        data = np.loadtxt(src_path, delimiter=',', comments='#')
        df = pd.DataFrame(data)

    # Save without header
    df.iloc[:, :4].to_csv(dst_path, header=False, index=False)
    print(f"Created track input file: {dst_path}")
    return dst_name

def run_optimization(track_name_input):
    """main_globaltraj.py 실행"""
    sys.path.append(OPT_ROOT)
    
    # main_globaltraj.py를 직접 import하는 대신, 내용을 수정해서 실행하거나 
    # 필요한 함수를 호출해야 하는데, main_globaltraj.py는 스크립트 형태임.
    # 따라서 subprocess로 실행하거나, 모듈로 로드하여 변수 수정 후 실행.
    # 여기서는 모듈 로드 방식 사용.
    
    main_script_path = os.path.join(OPT_ROOT, "main_globaltraj.py")
    
    # 모듈 스펙 생성
    spec = importlib.util.spec_from_file_location("main_globaltraj", main_script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_globaltraj"] = module
    
    # 소스 코드 읽기
    with open(main_script_path, 'r') as f:
        source = f.read()
    
    # 트랙 이름 강제 주입 (소스 코드 수정 없이 실행 컨텍스트 조작은 어려우므로 exec 사용)
    # track_name 변수를 덮어씌우기 위해 source의 해당 라인을 찾거나, 
    # 단순히 전역 변수로 설정하고 실행.
    
    # 더 안전한 방법: main_globaltraj.py를 수정하여 인자를 받게 하거나,
    # file_paths["track_name"] = "..." 부분을 우리가 원하는 값으로 replace하고 exec
    
    new_source = source.replace(
        'file_paths["track_name"] = "berlin_2018"', 
        f'file_paths["track_name"] = "{track_name_input}"'
    )
    
    # 최적화 타입 설정 (mintime)
    # 기본값이 mintime이므로 그대로 둠.
    
    # 실행 디렉토리 변경 (relative import 문제 해결)
    cwd = os.getcwd()
    os.chdir(OPT_ROOT)
    
    try:
        exec(new_source, module.__dict__)
        print("Optimization finished successfully.")
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(cwd)

def convert_output_to_raceline(track_name, variable_maps_dir):
    """결과 파일을 raceline.csv 형식으로 변환"""
    src_path = os.path.join(OUTPUT_DIR, "traj_race_cl.csv")
    dst_path = os.path.join(variable_maps_dir, track_name, f"{track_name}_raceline_opti.csv")
    
    if not os.path.exists(src_path):
        print(f"Output file not found: {src_path}")
        return
    
    # Load result
    # traj_race_cl.csv format (from main_globaltraj.py):
    # s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2
    df = pd.read_csv(src_path, comment='#', header=None)
    
    # F1TENTH raceline format:
    # # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    # Data separated by semicolon
    
    with open(dst_path, 'w') as f:
        f.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        # pandas to csv with semicolon
        df.to_csv(f, sep=';', header=False, index=False, float_format='%.7f')
        
    print(f"Saved optimized raceline to: {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="Optimize Raceline for Variable Width Track")
    parser.add_argument("--track", type=str, required=True, help="Track name (e.g. Catalunya)")
    args = parser.parse_args()
    
    track_name = args.track
    variable_maps_dir = os.path.join(F1TENTH_ROOT, "variable_width_maps")
    
    # 1. Dynamic Config Load
    dyn_yaml_path = os.path.join(F1TENTH_ROOT, "configs", "task", "dynamic.yaml")
    dyn_config = load_dynamic_yaml(dyn_yaml_path)
    
    # 2. Update racecar.ini (한 번만 업데이트)
    ini_path = os.path.join(PARAMS_DIR, "racecar.ini")
    update_racecar_ini(ini_path, dyn_config)
    
    # 트랙 목록 결정
    if args.track:
        tracks = [args.track]
    else:
        # 모든 하위 폴더 탐색
        tracks = []
        if os.path.exists(variable_maps_dir):
            for d in sorted(os.listdir(variable_maps_dir)):
                if os.path.isdir(os.path.join(variable_maps_dir, d)) and not d.startswith('.'):
                    tracks.append(d)
        else:
            print(f"Error: Directory not found: {variable_maps_dir}")
            return

    print(f"Target tracks: {tracks}")

    # 각 트랙에 대해 최적화 실행
    for track_name in tracks:
        print(f"\n{'='*50}")
        print(f"Processing track: {track_name}")
        print(f"{'='*50}")
        
        try:
            # 3. Prepare Track Input
            track_input_name = prepare_track_file(track_name, variable_maps_dir)
            
            # 4. Run Optimization
            run_optimization(track_input_name)
            
            # 5. Save Result
            convert_output_to_raceline(track_name, variable_maps_dir)
            
        except Exception as e:
            print(f"Failed to process {track_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

