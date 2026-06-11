import os
from datetime import datetime
from typing import Callable
from contextlib import redirect_stdout
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from elastica import *
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

from encoder import StateEncoder

class PneumaticSimulator(BaseSystemCollection, Constraints, Damping):
    pass

class PneumaticArmEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(PneumaticArmEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # 12D Observation: Tip, Error Vector, Velocity, Pressure
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.dt = 1e-4  
        self.rl_step_time = 0.1  
        self.steps_per_action = int(self.rl_step_time / self.dt)
        self.max_episode_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.simulator = PneumaticSimulator()
        
        # DOMAIN RANDOMIZATION: For Sim2Real Transfer
        # Randomize Silicone Stiffness (Young's Modulus)
        base_E = 1e5
        E_randomized = base_E * np.random.uniform(0.8, 1.2) 
        
        # Randomize Internal Friction (Damping)
        base_damping = 0.8
        damping_randomized = base_damping * np.random.uniform(0.5, 1.5)
        
        # Randomize Pneumatic Latency (Airflow speed)
        # 1.0 for instant application, 0.4 for delayed response
        self.alpha_randomized = np.random.uniform(0.4, 1.0)
        
        # Geometrically feasible targets
        target_kappa = np.random.uniform(0.2, 2.5) 
        target_yaw = np.random.uniform(0, 2 * np.pi)
        
        R = 1.0 / target_kappa
        theta = 0.6 * target_kappa 
        
        horizontal_reach = R * (1.0 - np.cos(theta))
        z_height = R * np.sin(theta)
        
        self.target_position = np.array([
            horizontal_reach * np.cos(target_yaw),
            horizontal_reach * np.sin(target_yaw),
            z_height
        ])

        # Physical arm setup
        n_elements = 20
        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([1.0, 0.0, 0.0])
        start_pos = np.array([0.0, 0.0, 0.0])
        
        self.arm = CosseratRod.straight_rod(
            n_elements, start_pos, direction, normal, 
            base_length=0.6, base_radius=0.02, density=1000,
            youngs_modulus=E_randomized, shear_modulus=E_randomized / 3.0
        )
        
        self.simulator.append(self.arm)
        self.simulator.constrain(self.arm).using(OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,))
        self.simulator.dampen(self.arm).using(AnalyticalLinearDamper, damping_constant=damping_randomized, time_step=self.dt)
        self.simulator.finalize()

        self.current_pressure = np.zeros(3, dtype=np.float32)
        self.previous_dist = np.linalg.norm(self.arm.position_collection[..., -1] - self.target_position)
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        target_pressure = np.clip(action, 0.0, 1.0)
        
        # Apply the randomized latency
        self.current_pressure = (1 - self.alpha_randomized) * self.current_pressure + self.alpha_randomized * target_pressure 
        p1, p2, p3 = self.current_pressure
        
        max_curvature = 8.0 
        
        kappa_x = max_curvature * (p1 * 1.0 + p2 * -0.5 + p3 * -0.5) 
        sin_120 = np.sqrt(3) / 2.0
        kappa_y = max_curvature * (p1 * 0.0 + p2 * sin_120 + p3 * -sin_120)
        
        self.arm.rest_kappa[0, :] = kappa_x
        self.arm.rest_kappa[1, :] = kappa_y

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            integrate(PositionVerlet(), self.simulator, self.rl_step_time, self.steps_per_action, progress_bar=False)

        observation = self._get_obs()
        tip_pos = observation[0:3]
        tip_vel = observation[6:9]
        
        dist = np.linalg.norm(tip_pos - self.target_position)
        
        delta_dist = self.previous_dist - dist
        self.previous_dist = dist
        
        # Dense reward for moving closer, slight penalty for total distance
        reward = (delta_dist * 200.0) - dist 
        
        if dist < 0.05: 
            reward += 10.0 
            reward += (0.05 - dist) * 100.0 # Pulls it to the core
            vel_mag = np.linalg.norm(tip_vel)
            reward -= vel_mag * 2.0 # Brakes
            
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        tip_pos = self.arm.position_collection[..., -1]
        tip_vel = self.arm.velocity_collection[..., -1]
        
        # Calculate error vector
        error_vec = self.target_position - tip_pos
        
        return np.concatenate([tip_pos, error_vec, tip_vel, self.current_pressure]).astype(np.float32)

def make_env():
    def _init():
        return Monitor(PneumaticArmEnv())
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    TOTAL_TIMESTEPS = int(2e6) 
    NUM_CORES = 8
    
    # Encourage exploration
    ENTROPY_COEF = 0.01 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"./arm_runs_domain/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    tb_log_dir = os.path.join(run_dir, "tensorboard")
    model_path = os.path.join(run_dir, "pneumatic_arm_ppo")
    vec_norm_path = os.path.join(run_dir, "vec_normalize.pkl")
    
    print(f"Creating isolated hardware environment at: {run_dir}")

    env = SubprocVecEnv([make_env() for _ in range(NUM_CORES)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=StateEncoder,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy", env, policy_kwargs=policy_kwargs, 
        learning_rate=linear_schedule(0.0003), ent_coef=ENTROPY_COEF, 
        verbose=1, device="cpu", tensorboard_log=tb_log_dir
    )

    print(f"Starting Training on {NUM_CORES} cores...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    
    model.save(model_path)
    env.save(vec_norm_path)
    print("Training complete.")
    env.close()