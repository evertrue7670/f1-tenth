from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-8

def unnormalize_speed(value, minimum, maximum):
    """
    Unnormalize speed from [0, 1] (또는 [-1, 1]에서 잘린 값) range to [minimum, maximum] range.
    Speed는 항상 >= 0 으로 보장.
    """
    value = np.asarray(value)

    # 음수는 브레이크로 간주
    value = np.maximum(value, 0.0)

    # [-1,1] 이었더라도 step에서 이미 [0,1]로 클리핑한 상태가 들어옴.
    # 혹시 모를 overflow 방지.
    value = np.clip(value, 0.0, 1.0)

    # [0,1] -> [min, max]
    temp_a = (maximum - minimum) / 2.0
    temp_b = (maximum + minimum) / 2.0

    temp_a = np.ones_like(value) * temp_a
    temp_b = np.ones_like(value) * temp_b
    result = temp_a * value * 2.0 + temp_b

    # 최종 안전장치
    min_speed = max(0.0, minimum)
    result = np.clip(result, min_speed, maximum)
    return result


class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None) -> None:

        self._env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            args=args,
            maps=maps,
            render_mode=render_mode
        )
        super().__init__(self._env)
        self.show_centerline = args.show_centerline

        # for control
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(
            -np.inf * np.ones(self.obs_dim),
            np.inf * np.ones(self.obs_dim),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(self.action_dim),
            np.ones(self.action_dim),
            dtype=np.float32
        )

        # pose 관련 상태
        self.position_frenet = np.zeros(2)
        self.yaw_frenet = 0.0
        self.delta_s = 0.0
        self.collision = False

        # steering smoothness 용
        self.prev_steer = 0.0

    # ------------------------ Pose 업데이트 ------------------------ #
    def _reset_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]

        # cartesian pose
        poses_x = np.nan_to_num(obs_dict['poses_x'][0], nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_y = np.nan_to_num(obs_dict['poses_y'][0], nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_theta = np.nan_to_num(obs_dict['poses_theta'][0], nan=0.0, posinf=np.pi, neginf=-np.pi)
        poses_theta = np.clip(poses_theta, -2*np.pi, 2*np.pi)

        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = poses_theta

        # frenet pose
        try:
            s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)
            if np.isnan(s) or np.isnan(ey) or np.isnan(phi) or \
               np.isinf(s) or np.isinf(ey) or np.isinf(phi):
                s, ey, phi = 0.0, 0.0, 0.0
            else:
                s = float(np.clip(s, -1000.0, 10000.0))
                ey = float(np.clip(ey, -50.0, 50.0))
                phi = float(np.clip(phi, -2*np.pi, 2*np.pi))
        except Exception:
            s, ey, phi = 0.0, 0.0, 0.0

        self.position_frenet = np.array([s, ey], dtype=np.float32)
        self.yaw_frenet = float(phi)
        self.delta_s = 0.0

    def _step_pose(self, obs_dict):
        self.collision = obs_dict['collisions'][0]

        poses_x = np.nan_to_num(obs_dict['poses_x'][0], nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_y = np.nan_to_num(obs_dict['poses_y'][0], nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_theta = np.nan_to_num(obs_dict['poses_theta'][0], nan=0.0, posinf=np.pi, neginf=-np.pi)
        poses_theta = np.clip(poses_theta, -2*np.pi, 2*np.pi)

        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = poses_theta

        prev_s = float(self.position_frenet[0]) if hasattr(self, 'position_frenet') else 0.0
        prev_ey = float(self.position_frenet[1]) if hasattr(self, 'position_frenet') else 0.0
        prev_phi = float(self.yaw_frenet) if hasattr(self, 'yaw_frenet') else 0.0

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s, ey, phi = self._env.track.cartesian_to_frenet2(poses_x, poses_y, poses_theta)

            if np.isnan(s) or np.isnan(ey) or np.isnan(phi) or \
               np.isinf(s) or np.isinf(ey) or np.isinf(phi):
                raise ValueError("Invalid Frenet Coordinates")

            s = float(np.clip(s, -1000.0, 10000.0))
            ey = float(np.clip(ey, -50.0, 50.0))
            phi = float(np.clip(phi, -2*np.pi, 2*np.pi))

        except Exception:
            # 실패하면 살짝 앞으로 간다고 가정 (stuck 방지용)
            s, ey, phi = prev_s + 0.1, prev_ey, prev_phi

        # delta s 계산
        self.delta_s = s - prev_s

        # 랩 크로싱 보정
        try:
            total_track_s = self._env.track.centerline.spline.s[-1]
            if abs(self.delta_s) > total_track_s / 2.0:
                if self.delta_s < 0:
                    self.delta_s += total_track_s
                else:
                    self.delta_s -= total_track_s
        except:
            pass

        self.delta_s = float(np.nan_to_num(self.delta_s, nan=0.0))
        self.delta_s = float(np.clip(self.delta_s, -100.0, 100.0))

        self.position_frenet = np.array([s, ey], dtype=np.float32)
        self.yaw_frenet = float(phi)

    # ------------------------ Curvature 계산 ------------------------ #
    def _get_curvature(self, s):
        """
        주어진 Frenet s 에서 중심선 곡률 κ(s)를 추정.
        트랙 구현에 따라 centerline에 get_curvature 등이 없을 수도 있으므로
        최대한 안전하게 처리 후 실패하면 0 반환.
        """
        try:
            cl = self._env.track.centerline

            # 1) get_curvature(s) 메서드가 있는 경우
            if hasattr(cl, "get_curvature"):
                kappa = cl.get_curvature(s)

            # 2) curvature array + s array 형태인 경우
            elif hasattr(cl, "curvatures") and hasattr(cl, "ss"):
                ss = np.asarray(cl.ss)
                curvs = np.asarray(cl.curvatures)
                idx = np.argmin(np.abs(ss - s))
                kappa = curvs[idx]
            else:
                kappa = 0.0

            kappa = float(np.clip(np.nan_to_num(kappa), -5.0, 5.0))
        except Exception:
            kappa = 0.0

        return kappa

    # ------------------------ reset ------------------------ #
    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        self.prev_steer = 0.0

        try:
            obs_dict, info = self._env.reset(**kwargs)
        except ValueError:
            print("Warning: Env reset failed with ValueError. Retrying...")
            return self.observation_space.sample(), {}

        if self.show_centerline and self._env.unwrapped.renderer is not None:
            try:
                self._env.unwrapped.add_render_callback(
                    self._env.track.centerline.render_waypoints
                )
            except:
                pass

        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        return obs, info

    # ------------------------ Reward ------------------------ #
    def calc_reward(self, action):
        """
        연속형 reward:
        1) progress (delta_s)  : 앞으로 많이 갈수록 좋음  [메인]
        2) centerline tracking : ey가 0에 가까울수록 좋음
        3) steering smoothness : |steer - prev_steer|가 작을수록 좋음
        4) curvature-based speed envelope : 코너에서 과속하면 penalty
        5) collision penalty
        """

        try:
            delta_s = float(self.delta_s)
            ey = float(self.position_frenet[1])
            heading_error = float(self.yaw_frenet)  # 현재는 reward에는 직접 사용 X
            velocity = float(self._env.sim.agents[0].state[3])  # vx
            collision = bool(self.collision)
        except:
            delta_s, ey, heading_error, velocity, collision = 0.0, 0.0, 0.0, 0.0, False

        # 0) delta_s: 너무 큰 음수는 잘라 줌 (후진 방지용)
        delta_s_clipped = max(delta_s, -0.5)

        # 1) Progress Reward (가중치 크게)
        w_progress = 20.0          # ★ 중요: 메인 드라이버
        R_progress = w_progress * delta_s_clipped

        # 2) Centerline (Gaussian on ey)
        ey = np.clip(ey, -5.0, 5.0)
        ey_scale = 0.7
        w_center = 0.3             # 예전 0.5 → 살짝 줄임
        R_center = w_center * np.exp(-0.5 * (ey / ey_scale) ** 2)

        # 3) Steering smoothness
        steer = float(action[0])
        steer = np.clip(steer, -1.0, 1.0)
        w_smooth = 0.05            # 예전 0.1 → 절반
        R_smooth = -w_smooth * abs(steer - self.prev_steer)
        self.prev_steer = steer

        # 4) Curvature-based speed envelope
        curvature = abs(self._get_curvature(self.position_frenet[0]))
        v_max_straight = self.max_speed
        curvature_gain = 8.0
        v_ref = v_max_straight / (1.0 + curvature_gain * curvature)
        v_ref = float(np.clip(v_ref, self.min_speed, self.max_speed))

        velocity = float(np.clip(np.nan_to_num(velocity), -5.0, 20.0))
        speed_excess = max(0.0, velocity - v_ref)
        w_speed_env = 0.1          # 예전 0.2 → 절반
        R_speed_env = -w_speed_env * (speed_excess ** 2)

        # 5) Collision penalty
        w_collision = 500.0
        R_collision = -w_collision if collision else 0.0

        # 작은 시간 패널티 (optional)
        time_penalty = 0.01

        reward = (
            R_progress +
            R_center +
            R_smooth +
            R_speed_env +
            R_collision -
            time_penalty
        )

        if np.isnan(reward) or np.isinf(reward):
            reward = R_collision  # 최소한 충돌 패널티만 남기기

        reward_dict = {
            "R_progress": float(R_progress),
            "R_center": float(R_center),
            "R_smooth": float(R_smooth),
            "R_speed_env": float(R_speed_env),
            "R_collision": float(R_collision),
            "delta_s": float(delta_s),
            "velocity": float(velocity),
            "v_ref": float(v_ref),
            "curvature": float(curvature),
        }
        return float(reward), reward_dict

    # ------------------------ step ------------------------ #
    def step(self, action: np.array):
        # 1) Action sanitization
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros_like(action)

        action = np.clip(action, -1.0, 1.0)
        _action = action.copy()

        # Steer: [-1,1] -> [-max_steer, max_steer]
        _action[0] = np.clip(_action[0] * self.max_steer,
                             -self.max_steer, self.max_steer)

        # Speed: [-1,1] -> [min_speed, max_speed] (음수는 브레이크)
        normalized_speed = _action[1]
        if normalized_speed < 0:
            normalized_speed = 0.0

        _action[1] = unnormalize_speed(
            normalized_speed,
            self.min_speed,
            self.max_speed
        )
        _action[1] = max(0.0, _action[1])

        # 2) Env step with safety
        try:
            obs_dict, _, terminate, truncate, info = self._env.step(_action)

            self._step_pose(obs_dict)
            obs = self.getObs(obs_dict)

            reward, reward_dict = self.calc_reward(_action)
            info['obs_dict'] = obs_dict
            info.update(reward_dict)
            return obs, reward, terminate, truncate, info

        except ValueError as e:
            error_msg = str(e)
            if "math domain error" in error_msg:
                pass
            else:
                print(f"Warning: Caught ValueError in env.step(): {error_msg}. Action: {action}")

            dummy_obs = np.zeros(self.obs_dim, dtype=np.float32)
            penalty_reward = -100.0
            return dummy_obs, penalty_reward, True, True, {"error": error_msg, "action": action.tolist()}

        except Exception as e:
            print(f"Warning: Critical error in env.step(): {e}")
            dummy_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return dummy_obs, -100.0, True, True, {"error": str(e)}

    # ------------------------ Observation ------------------------ #
    def getObs(self, obs_dict, reset=False):
        scans = np.array(obs_dict['scans'])
        if scans.ndim == 2:
            scan = scans[0]
        else:
            scan = scans.flatten()

        scan = np.nan_to_num(scan, nan=10.0, posinf=10.0, neginf=0.0)
        scan = np.clip(scan, 0.0, 10.0)
        scan = scan.reshape(-1, 4).mean(axis=1)

        linear_vels_x = np.array(obs_dict['linear_vels_x'])
        linear_vels_y = np.array(obs_dict['linear_vels_y'])
        ang_vels_z = np.array(obs_dict['ang_vels_z'])

        linear_vel_x = float(linear_vels_x[0]) if linear_vels_x.ndim > 0 else float(linear_vels_x)
        linear_vel_y = float(linear_vels_y[0]) if linear_vels_y.ndim > 0 else float(linear_vels_y)
        ang_vel_z = float(ang_vels_z[0]) if ang_vels_z.ndim > 0 else float(ang_vels_z)

        linear_vel_x = np.clip(np.nan_to_num(linear_vel_x), -20.0, 20.0)
        linear_vel_y = np.clip(np.nan_to_num(linear_vel_y), -20.0, 20.0)
        ang_vel_z = np.clip(np.nan_to_num(ang_vel_z), -10.0, 10.0)

        lat_dev = float(self.position_frenet[1])
        head_err = float(self.yaw_frenet)

        lat_dev = np.clip(np.nan_to_num(lat_dev), -10.0, 10.0)
        head_err = np.clip(np.nan_to_num(head_err), -2*np.pi, 2*np.pi)

        observation = np.concatenate([
            scan,
            [linear_vel_x],
            [linear_vel_y],
            [ang_vel_z],
            [lat_dev],
            [head_err]
        ]).astype(np.float32)

        return observation

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
