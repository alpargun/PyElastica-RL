import os
from typing import Callable
from contextlib import redirect_stdout
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from elastica import *
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

from encoder import StateEncoder

# -----------------------------------------------------------------------------
# PyElastica Simulator Setup
# -----------------------------------------------------------------------------
class GripperSimulator(BaseSystemCollection, Constraints, Damping):
    """
    Simulator class required by PyElastica. 
    Notice we removed 'Forcing' entirely. We actuate via intrinsic curvature!
    """
    pass

# -----------------------------------------------------------------------------
# Gymnasium RL Environment
# -----------------------------------------------------------------------------
class SoftGripperEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(SoftGripperEnv, self).__init__()
        
        # Action space: Pneumatic pressure for 2 fingers
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 3D coordinates of Finger 1 tip, Finger 2 tip, and Target, and 6 velocities (3 for each tip)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.dt = 1e-4  
        self.rl_step_time = 0.1  
        self.steps_per_action = int(self.rl_step_time / self.dt)
        self.max_episode_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.simulator = GripperSimulator()
        
        # DOMAIN RANDOMIZATION
        base_E = 1e5
        E_randomized = base_E * np.random.uniform(0.8, 1.2) 
        base_damping = 0.1
        damping_randomized = base_damping * np.random.uniform(0.5, 1.5)
        
        # Update the target position
        self.target_position = np.array([
            0.0, 
            np.random.uniform(0.47, 0.49), 
            0.0  # Locked to the 2D bending plane of the fingers
        ])

        # GRIPPER GEOMETRY
        n_elements = 20
        direction = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        base_length = 0.5
        base_radius = 0.02
        density = 1000

        # Finger 1 (Left)
        start_1 = np.array([-0.1, 0.0, 0.0])
        self.finger1 = CosseratRod.straight_rod(
            n_elements, start_1, direction, normal, base_length, base_radius, density,
            youngs_modulus=E_randomized, shear_modulus=E_randomized / 3.0
        )
        self.simulator.append(self.finger1)
        
        # Finger 2 (Right)
        start_2 = np.array([0.1, 0.0, 0.0])
        self.finger2 = CosseratRod.straight_rod(
            n_elements, start_2, direction, normal, base_length, base_radius, density,
            youngs_modulus=E_randomized, shear_modulus=E_randomized / 3.0
        )
        self.simulator.append(self.finger2)

        # BOUNDARY CONDITIONS & DAMPING
        self.simulator.constrain(self.finger1).using(OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,))
        self.simulator.constrain(self.finger2).using(OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,))

        self.simulator.dampen(self.finger1).using(AnalyticalLinearDamper, damping_constant=damping_randomized, time_step=self.dt)
        self.simulator.dampen(self.finger2).using(AnalyticalLinearDamper, damping_constant=damping_randomized, time_step=self.dt)

        self.simulator.finalize()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        act_f1, act_f2 = action[0], action[1]
        
        # INTRINSIC CURVATURE
        max_curvature = 2.0 
        
        # Modify rest_kappa in-place. Index [0, :] bends the rod in the XY plane.
        self.finger1.rest_kappa[0, :] = -max_curvature * act_f1 # bends inward (negative curvature)
        self.finger2.rest_kappa[0, :] = max_curvature * act_f2 # bends inward (positive curvature)

        # Step the Physics Engine
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            integrate(PositionVerlet(), self.simulator, self.rl_step_time, self.steps_per_action, progress_bar=False)

        # REWARD -------------------------------------------------------------------------------------------------------
        # Compute Reward
        observation = self._get_obs()
        tip1 = observation[0:3]
        tip2 = observation[3:6]
        
        # Distance from tips to the target
        dist1 = np.linalg.norm(tip1 - self.target_position)
        dist2 = np.linalg.norm(tip2 - self.target_position)
        
        # Base penalty for distance
        reward = - (dist1 + dist2) 
        
        # The Success Zone
        if dist1 < 0.05 and dist2 < 0.05:
            # Base bonus for entering the zone
            reward += 10.0
            
            # The closer it gets to 0.0, the higher the multiplier pays out
            reward += (0.05 - dist1) * 200.0
            reward += (0.05 - dist2) * 200.0
            
            # Symmetry Penalty
            # tip1's X is negative, tip2's X is positive. If symmetrical, their sum is 0.
            # If the red finger crosses over the center line, this heavily penalizes the score.
            reward -= abs(tip1[0] + tip2[0]) * 50.0
            
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        tip1_pos = self.finger1.position_collection[..., -1]
        tip2_pos = self.finger2.position_collection[..., -1]
        
        # Grab the velocity of the fingertips
        tip1_vel = self.finger1.velocity_collection[..., -1]
        tip2_vel = self.finger2.velocity_collection[..., -1]
        
        return np.concatenate([tip1_pos, tip2_pos, self.target_position, tip1_vel, tip2_vel]).astype(np.float32)


def make_env():
    def _init():
        return Monitor(SoftGripperEnv())
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == "__main__":
    TOTAL_TIMESTEPS = 500000 #1e6
    NUM_CORES = 8
    ENTROPY_COEF = 0.0 

    env = SubprocVecEnv([make_env() for _ in range(NUM_CORES)])

    policy_kwargs = dict(
        features_extractor_class=StateEncoder,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        learning_rate=linear_schedule(0.0003), 
        ent_coef=ENTROPY_COEF, 
        verbose=1, 
        device="cpu",
        tensorboard_log="./ppo_gripper_pneumatic_tensorboard/"
    )

    print(f"Starting training with Domain Randomization enabled on {NUM_CORES} cores...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("soft_gripper_ppo_pneumatic3")
    print("Training complete and model saved.")
    env.close()