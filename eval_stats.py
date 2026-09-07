import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import PneumaticArmEnv, SoftGripperEnv

TOLERANCE = 0.05

CONFIGS = {
    "trunk": (PneumaticArmEnv, "vec_normalize.pkl", "pneumatic_arm_ppo"),
    "gripper": (SoftGripperEnv, "vec_normalize_final.pkl", "soft_gripper_ppo_final"),
}


def tip_errors(base_env):
    target = base_env.target_position
    if hasattr(base_env, "arm"):
        return [np.linalg.norm(base_env.arm.position_collection[:, -1] - target)]
    return [
        np.linalg.norm(base_env.finger1.position_collection[:, -1] - target),
        np.linalg.norm(base_env.finger2.position_collection[:, -1] - target),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy over many episodes.")
    parser.add_argument("env", choices=CONFIGS.keys())
    parser.add_argument("run_folder")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    env_cls, norm_file, model_file = CONFIGS[args.env]

    env = DummyVecEnv([lambda: env_cls()])
    env = VecNormalize.load(f"{args.run_folder}/{norm_file}", env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(f"{args.run_folder}/{model_file}")

    base_env = env.venv.envs[0]
    final_errors = []
    rewards = []

    for ep in range(args.episodes):
        obs = env.reset()
        history = []
        total_reward = 0.0
        done_flag = False

        while not done_flag:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            history.append(tip_errors(base_env))
            total_reward += reward[0]
            done_flag = done[0]

        # history[-1] is post-reset, so take the last real step
        final_errors.append(history[-2])
        rewards.append(total_reward)

    errors = np.array(final_errors)
    success = (errors < TOLERANCE).all(axis=1)

    print(f"{args.episodes} episodes")
    print(f"mean tip error: {errors.mean() * 100:.1f} cm (worst tip: {errors.max() * 100:.1f} cm)")
    print(f"success rate within {TOLERANCE * 100:.0f} cm: {success.mean() * 100:.0f}%")
    print(f"mean episodic reward: {np.mean(rewards):.0f}")