import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom imports
from encoder import StateEncoder 
from train_gripper_pneumatic import SoftGripperEnv

if __name__ == "__main__":
    print("Loading environment and trained model...")
    
    # Wrap our single environment in a DummyVecEnv
    env = DummyVecEnv([lambda: SoftGripperEnv()])
    
    # Load the normalization filter from training
    env = VecNormalize.load("vec_normalize.pkl", env)
    # We must tell the filter NOT to update its math during evaluation
    env.training = False
    env.norm_reward = False

    model = PPO.load("soft_gripper_ppo_pneumatic4")

    # Reset the environment
    obs = env.reset()
    
    # Because VecEnv batches inputs, extract the un-normalized original observation to get true 3D plotting coordinates
    unnorm_obs = env.get_original_obs()
    target_pos = unnorm_obs[0][6:9]
    
    f1_history = []
    f2_history = []
    
    total_reward = 0
    done_flag = False
    step_count = 0
    
    print("Running 100-step physics simulation...")
    
    while not done_flag:
        action, _states = model.predict(obs, deterministic=True)
        
        # Note: Stable Baselines3 VecEnvs return 4 values, handling truncation automatically
        obs, reward, done, info = env.step(action)
        
        base_env = env.venv.envs[0]
        
        f1_history.append(base_env.finger1.position_collection.copy())
        f2_history.append(base_env.finger2.position_collection.copy())
        
        total_reward += reward[0]
        step_count += 1
        done_flag = done[0]

    print(f"Simulation finished. Cumulative Evaluated Reward: {total_reward:.2f}")

    print("Generating 3D Animation Video...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    line1, = ax.plot([], [], [], lw=6, color='blue', label='Finger 1')
    line2, = ax.plot([], [], [], lw=6, color='red', label='Finger 2')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='green', s=300, marker='*', label='Target')
    
    ax.set_xlim([-0.2, 0.2])   
    ax.set_ylim([0.0, 0.6])    
    ax.set_zlim([-0.1, 0.1])   
    
    ax.set_title("Soft Pneumatic Gripper Animation (Normalized)")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Length/Forward)")
    ax.set_zlabel("Z (Height)")
    ax.legend()
    
    def update(frame):
        pos1 = f1_history[frame]
        pos2 = f2_history[frame]
        line1.set_data(pos1[0, :], pos1[1, :])
        line1.set_3d_properties(pos1[2, :])
        line2.set_data(pos2[0, :], pos2[1, :])
        line2.set_3d_properties(pos2[2, :])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=len(f1_history), interval=100, blit=False)
    plt.show()