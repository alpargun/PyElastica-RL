import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import SoftGripperEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roll out a trained gripper policy and animate it.")
    parser.add_argument("run_folder", help="Training run directory, e.g. runs/gripper/run_20260528_152332")
    parser.add_argument("--save", default=None, help="Optional output path for a .gif of the rollout")
    args = parser.parse_args()

    env = DummyVecEnv([lambda: SoftGripperEnv()])

    env = VecNormalize.load(f"{args.run_folder}/vec_normalize_final.pkl", env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(f"{args.run_folder}/soft_gripper_ppo_final")

    obs = env.reset()

    # Get the true coordinate directly from the base environment
    base_env = env.venv.envs[0]
    target_pos = base_env.target_position

    f1_history = []
    f2_history = []
    done_flag = False
    total_reward = 0

    while not done_flag:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        f1_history.append(base_env.finger1.position_collection.copy())
        f2_history.append(base_env.finger2.position_collection.copy())

        total_reward += reward[0]
        done_flag = done[0]

    d1 = np.linalg.norm(f1_history[-2][:, -1] - target_pos)
    d2 = np.linalg.norm(f2_history[-2][:, -1] - target_pos)
    print(f"Simulation finished. Reward: {total_reward:.2f}, tip errors: {d1:.3f} m / {d2:.3f} m")

    print("Generating 3D Animation...")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    line1, = ax.plot([], [], [], lw=6, color='blue', label='Finger 1')
    line2, = ax.plot([], [], [], lw=6, color='red', label='Finger 2')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='green', s=300, marker='*', label='Target')

    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([0.0, 0.55])
    ax.set_zlim([-0.05, 0.05])
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_yticks([0.0, 0.2, 0.4])

    ax.set_title("Soft Pneumatic Gripper Animation")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Length/Forward)")
    ax.set_zlabel("Z (Height)")
    ax.legend(loc='upper left')
    ax.view_init(elev=25, azim=-70)

    def update(frame):
        pos1 = f1_history[frame]
        pos2 = f2_history[frame]
        line1.set_data(pos1[0, :], pos1[1, :])
        line1.set_3d_properties(pos1[2, :])
        line2.set_data(pos2[0, :], pos2[1, :])
        line2.set_3d_properties(pos2[2, :])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(f1_history[:-1]), interval=100, blit=False)

    if args.save:
        ani.save(args.save, writer=animation.PillowWriter(fps=10), dpi=70)
        print(f"Animation saved to {args.save}")
    else:
        plt.show()