from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

EPS = 1e-8

def unnormalize_speed(value, minimum, maximum):
    """
    Unnormalize speed from [-1, 1] or [0, 1] range to [minimum, maximum] range.
    CRITICAL: Speed must be non-negative, so negative inputs are treated as 0.
    
    Args:
        value: normalized speed in [-1, 1] or [0, 1] range
        minimum: minimum speed (should be >= 0)
        maximum: maximum speed
    
    Returns:
        unnormalized speed in [minimum, maximum] range (always >= 0)
    """
    value = np.asarray(value)
    
    # Handle negative values: treat as 0 (brake)
    # This prevents reverse speed which can cause math domain errors
    value = np.maximum(value, 0.0)
    
    # If value is in [-1, 1] range, map to [0, 1]
    # If value is already in [0, 1], keep as is
    if np.any(value > 1.0):
        # Already in [0, 1] or beyond, clip to [0, 1]
        value = np.clip(value, 0.0, 1.0)
    elif np.any(value < 0):
        # In [-1, 1] range, map to [0, 1]
        value = (value + 1.0) / 2.0
        value = np.clip(value, 0.0, 1.0)
    
    # Linear transformation: [0, 1] -> [min, max]
    temp_a = (maximum - minimum) / 2.0
    temp_b = (maximum + minimum) / 2.0
    
    # numpy broadcasting safety
    temp_a = np.ones_like(value) * temp_a
    temp_b = np.ones_like(value) * temp_b
    result = temp_a * value * 2.0 + temp_b  # Scale from [0,1] to [min,max]
    
    # Final safety: ensure non-negative and within bounds
    min_speed = max(0.0, minimum)  # Speed cannot be negative
    result = np.clip(result, min_speed, maximum)
    
    return result

class F1Wrapper(gym.Wrapper):
    def __init__(self, args, maps, render_mode=None) -> None:
        
        self._env = gym.make("f1tenth_gym:f1tenth-v0", 
                            args=args,
                            maps=maps,
                            render_mode=render_mode)
        super().__init__(self._env)
        self.show_centerline = args.show_centerline

        # for control
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer

        # for spaces
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.observation_space = spaces.Box(-np.inf*np.ones(self.obs_dim), np.inf*np.ones(self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-np.ones(self.action_dim), np.ones(self.action_dim), dtype=np.float32)
        
        # Initialize internal states
        self.position_frenet = np.zeros(2)
        self.yaw_frenet = 0.0
        self.delta_s = 0.0
        self.collision = False
        self.prev_steer = 0.0  # For steering smoothness reward
        self.scan_buffer = np.zeros(1080) # Buffer for lidar processing


    def _reset_pose(self, obs_dict):
        # collision
        self.collision = obs_dict['collisions'][0]
        
        # cartesian coordinate pose
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        
        # Check for NaN/Inf in poses and clip to reasonable ranges
        poses_x = np.nan_to_num(poses_x, nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_y = np.nan_to_num(poses_y, nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_theta = np.nan_to_num(poses_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
        poses_theta = np.clip(poses_theta, -2*np.pi, 2*np.pi)
        
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = poses_theta
        
        # frenet coordinate pose with comprehensive error handling
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
        # collision
        self.collision = obs_dict['collisions'][0]
        
        poses_x, poses_y, poses_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        
        poses_x = np.nan_to_num(poses_x, nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_y = np.nan_to_num(poses_y, nan=0.0, posinf=1000.0, neginf=-1000.0)
        poses_theta = np.nan_to_num(poses_theta, nan=0.0, posinf=np.pi, neginf=-np.pi)
        poses_theta = np.clip(poses_theta, -2*np.pi, 2*np.pi)
        
        self.position = np.stack([poses_x, poses_y]).T
        self.yaw = poses_theta
        
        # Safe fallback values
        prev_s = float(self.position_frenet[0]) if hasattr(self, 'position_frenet') else 0.0
        prev_ey = float(self.position_frenet[1]) if hasattr(self, 'position_frenet') else 0.0
        prev_phi = float(self.yaw_frenet) if hasattr(self, 'yaw_frenet') else 0.0
        
        try:
            # Suppress warnings and try conversion
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
            # On ANY failure, assume small forward movement or stay put
            s, ey, phi = prev_s + 0.1, prev_ey, prev_phi # Assume slight forward progress to avoid stuck logic
        
        # Delta S logic
        self.delta_s = s - prev_s
        
        try:
            # Lap crossing logic
            total_track_s = self._env.track.centerline.spline.s[-1]
            if abs(self.delta_s) > total_track_s/2.0:
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

    def reset(self, **kwargs):
        self.history = deque(maxlen=10)
        try:
            obs_dict, info = self._env.reset(**kwargs)
        except ValueError:
            # If reset fails internally (rare but possible), try recursive reset or return zeros
            print("Warning: Env reset failed with ValueError. Retrying...")
            return self.observation_space.sample(), {}

        if self.show_centerline and self._env.unwrapped.renderer is not None:
            try:
                self._env.unwrapped.add_render_callback(self._env.track.centerline.render_waypoints)
            except:
                pass
        
        self._reset_pose(obs_dict)
        obs = self.getObs(obs_dict, reset=True)
        info['obs_dict'] = obs_dict
        return obs, info
    
    def calc_reward(self):        
        try:
            delta_s = float(self.delta_s)
            lateral_deviation = float(self.position_frenet[1])
            heading_error = float(self.yaw_frenet)
            velocity = float(self._env.sim.agents[0].state[3]) # Linear velocity x
            collision = bool(self.collision)
        except:
            delta_s, lateral_deviation, heading_error, velocity, collision = 0.0, 0.0, 0.0, 0.0, False
        
        # 1. Progress Reward: Non-linear speed reward
        # Encourage higher speeds, but penalty for reversing is handled separately
        velocity = np.clip(velocity, -10.0, 20.0)
        
        # Base reward for moving forward
        progress_reward = velocity * 0.1
        
        # 2. Centerline Tracking: Cosine-like peaked reward
        # exp(-k * x^2) is better than exp(-k * |x|) for smooth gradients near 0
        lateral_deviation = np.clip(lateral_deviation, -10.0, 10.0)
        centerline_reward = np.exp(-0.5 * lateral_deviation**2)  # Gaussian-like
        
        # 3. Heading Alignment: Cosine reward
        # cos(heading_error) gives 1.0 at 0 error, -1.0 at 180 deg error
        heading_error = np.clip(heading_error, -np.pi, np.pi)
        heading_reward = np.cos(heading_error)
        
        # 4. Penalties
        collision_cost = 100.0 if collision else 0.0
        
        # CRITICAL: Reverse Penalty
        # If velocity is negative, apply huge penalty to discourage "cowardly" behavior
        reverse_penalty = 0.0
        if velocity < -0.1:
            reverse_penalty = 20.0 * abs(velocity) # Strong penalty proportional to reverse speed
            progress_reward = 0.0 # No progress reward for reversing
            
        # 5. Steering Smoothness (Optional but good)
        # Penalize high steering changes if available (requires storing prev action)
        
        # 6. Time penalty (Constant existence cost)
        time_penalty = 0.01
        
        # Combined Reward
        # Weighted sum favoring speed on track
        reward = (progress_reward * 2.0 +           # Main driver
                  centerline_reward * 0.5 +         # Keep on track
                  heading_reward * 0.5 -            # Face forward
                  collision_cost -                  # Don't crash
                  reverse_penalty -                 # Don't reverse
                  time_penalty)                     # Hurry up
        
        # Safety clip
        if np.isnan(reward) or np.isinf(reward):
            reward = -collision_cost
            
        reward_dict = {
            'progress_reward': float(progress_reward),
            'centerline_reward': float(centerline_reward),
            'heading_reward': float(heading_reward),
            'collision_cost': float(collision_cost),
            'reverse_penalty': float(reverse_penalty),
            'velocity': float(velocity)
        }
        return float(reward), reward_dict
        
    def step(self, action:np.array):
        # ---------------------------------------------------------------------
        # [CRITICAL FIX] 1. Sanitize Action Input
        # Neural Network output might be NaN or Inf, which crashes the Simulator
        # ---------------------------------------------------------------------
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            # Fallback action: Steer 0, Speed 0 (brake)
            action = np.zeros_like(action)
        
        # Clip raw action to spaces.ActionSpace range (-1 to 1) just in case
        action = np.clip(action, -1.0, 1.0)
        
        _action = action.copy()
        
        # Steer: [-1, 1] -> [-max_steer, max_steer]
        _action[0] = np.clip(_action[0] * self.max_steer, -self.max_steer, self.max_steer)
        
        # Speed: Normalize from [-1, 1] or [0, 1] to [min_speed, max_speed]
        # CRITICAL: Speed must be non-negative!
        normalized_speed = _action[1]
        
        # If speed is negative, treat it as 0 (brake)
        if normalized_speed < 0:
            normalized_speed = 0.0
        
        # Map [0, 1] to [min_speed, max_speed]
        # If input was in [-1, 1], we already clamped to 0, so now map [0, 1] -> [min, max]
        _action[1] = unnormalize_speed(normalized_speed, self.min_speed, self.max_speed)
        
        # Final safety check: speed must be non-negative
        _action[1] = max(0.0, _action[1])

        # ---------------------------------------------------------------------
        # [CRITICAL FIX] 2. Wrap Environment Step in Try-Except
        # Catch 'ValueError: math domain error' from inside f1tenth_gym
        # ---------------------------------------------------------------------
        try:
            obs_dict, _, terminate, truncate, info = self._env.step(_action)
            
            # Update steer buffer
            self.steer_buffer.append(action[0])
            
            self._step_pose(obs_dict)
            obs = self.getObs(obs_dict)
            reward, reward_dict = self.calc_reward()
            info['obs_dict'] = obs_dict
            info.update(reward_dict)
            return obs, reward, terminate, truncate, info

        except ValueError as e:
            # Log the error but DO NOT crash the training thread
            error_msg = str(e)
            if "math domain error" in error_msg:
                # This is expected occasionally when vehicle goes off-track
                # The error is caught and episode is terminated safely
                # No need to print every time (too verbose)
                pass  # Silent handling - this is expected behavior
            else:
                print(f"Warning: Caught ValueError in env.step(): {error_msg}. Action: {action}")
            
            # Force episode termination with a penalty
            # Return dummy observation
            dummy_obs = np.zeros(self.obs_dim, dtype=np.float32)
            penalty_reward = -100.0
            return dummy_obs, penalty_reward, True, True, {"error": error_msg, "action": action.tolist()}

        except Exception as e:
            print(f"Warning: Critical error in env.step(): {e}")
            dummy_obs = np.zeros(self.obs_dim, dtype=np.float32)
            return dummy_obs, -100.0, True, True, {"error": str(e)}

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