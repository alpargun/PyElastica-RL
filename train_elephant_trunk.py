import os
from datetime import datetime
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from encoder import StateEncoder
from envs import PneumaticArmEnv


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
    run_dir = f"./runs/trunk/run_{timestamp}"
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
    print(f"Training complete. Run saved to: {run_dir}")
    env.close()