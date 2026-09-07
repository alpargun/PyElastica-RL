import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import PneumaticArmEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll out a trained trunk policy and animate it.")
    parser.add_argument("run_folder", help="Training run directory, e.g. runs/trunk/run_20260604_154313")
    parser.add_argument("--save", default=None, help="Optional output path for a .gif of the rollout")
    args = parser.parse_args()

    env = DummyVecEnv([lambda: PneumaticArmEnv()])

    env = VecNormalize.load(f"{args.run_folder}/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(f"{args.run_folder}/pneumatic_arm_ppo")

    obs = env.reset()

    base_env = env.venv.envs[0]
    target_pos = base_env.target_position

    arm_history = []
    done_flag = False
    total_reward = 0

    print("Running 100-step simulation...")

    while not done_flag:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        arm_history.append(base_env.arm.position_collection.copy())

        total_reward += reward[0]
        done_flag = done[0]

    print(f"Simulation finished. Cumulative Evaluated Reward: {total_reward:.2f}")

    # 3D Animation block
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    line1, = ax.plot([], [], [], lw=8, color='steelblue', label='Pneumatic Arm')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='red', s=300, marker='o', label='Target')

    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([0.0, 0.6])

    ax.set_title("3-Bellow Pneumatic Arm Simulation")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Depth)")
    ax.set_zlabel("Z (Height)")
    ax.legend()

    def update(frame):
        pos = arm_history[frame]
        line1.set_data(pos[0, :], pos[1, :])
        line1.set_3d_properties(pos[2, :])
        return line1,

    ani = animation.FuncAnimation(fig, update, frames=len(arm_history), interval=100, blit=False)

    if args.save:
        ani.save(args.save, writer=animation.PillowWriter(fps=10))
        print(f"Animation saved to {args.save}")
    else:
        plt.show()