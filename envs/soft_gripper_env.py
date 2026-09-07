import os
from contextlib import redirect_stdout

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from elastica import *
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate


class GripperSimulator(BaseSystemCollection, Constraints, Damping):
    # No 'Forcing' mixin: the fingers are actuated through intrinsic curvature.
    pass


class SoftGripperEnv(gym.Env):
    """Two opposing pneumatic fingers reaching for a common target point."""

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(SoftGripperEnv, self).__init__()

        # Action space: Pneumatic pressure for 2 fingers
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: 15 dimensions (9 for pos, 6 for velocities to prevent blind momentum)
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

        # TARGET POSITION (Locked to Z=0 for 1-DOF Actuators)
        self.target_position = np.array([
            0.0,
            np.random.uniform(0.47, 0.49),
            0.0
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

        # Initialize pressure to 0 at the start of every episode
        self.current_pressure = np.zeros(2, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        max_curvature = 2.0

        # Simulate pneumatic airflow delay (Low-Pass Filter)
        alpha = 0.2  # alpha = 0.2 means it takes a few steps for pressure to build up
        self.current_pressure = (1 - alpha) * self.current_pressure + alpha * action

        # Opposite signs so positive pressure closes both fingers inward
        self.finger1.rest_kappa[0, :] = -max_curvature * self.current_pressure[0]
        self.finger2.rest_kappa[0, :] = max_curvature * self.current_pressure[1]

        # Step Physics Engine
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            integrate(PositionVerlet(), self.simulator, self.rl_step_time, self.steps_per_action, progress_bar=False)

        # Observation Extraction
        observation = self._get_obs()
        tip1 = observation[0:3]
        tip2 = observation[3:6]
        tip1_vel = observation[9:12]
        tip2_vel = observation[12:15]

        dist1 = np.linalg.norm(tip1 - self.target_position)
        dist2 = np.linalg.norm(tip2 - self.target_position)

        # 1. Base Distance Penalty
        reward = - (dist1 + dist2)

        if dist1 < 0.05 and dist2 < 0.05:
            # 2. Zone Bonus
            reward += 10.0
            # 3. Continuous Precision Bonus
            reward += (0.05 - dist1) * 200.0
            reward += (0.05 - dist2) * 200.0
            # 4. Symmetry Penalty
            reward -= abs(tip1[0] + tip2[0]) * 50.0

        # 5. Velocity penalty, forces a smooth stop instead of jitter
        reward -= (np.linalg.norm(tip1_vel) + np.linalg.norm(tip2_vel)) * 2.0

        # 6. Hard penalty for fingers crossing over each other
        if tip1[0] > tip2[0]:
            reward -= 100.0

        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        return observation, reward, terminated, truncated, {}

    def _get_obs(self):
        tip1_pos = self.finger1.position_collection[..., -1]
        tip2_pos = self.finger2.position_collection[..., -1]
        tip1_vel = self.finger1.velocity_collection[..., -1]
        tip2_vel = self.finger2.velocity_collection[..., -1]
        return np.concatenate([tip1_pos, tip2_pos, self.target_position, tip1_vel, tip2_vel]).astype(np.float32)
