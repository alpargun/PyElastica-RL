import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO

from train_gripper_pneumatic import SoftGripperEnv

if __name__ == "__main__":
    print("Loading environment and trained model...")
    
    env = SoftGripperEnv()
    
    model = PPO.load("soft_gripper_ppo_pneumatic3")

    obs, info = env.reset()
    target_pos = obs[6:9]
    
    # Arrays to store the shape of the fingers
    f1_history = []
    f2_history = []
    
    total_reward = 0
    done = False
    
    print("Running physics simulation...")
    
    # ---------------------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------------------
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Save the full 3D array of the rod (shape: 3 x 20 nodes)
        f1_history.append(env.finger1.position_collection.copy())
        f2_history.append(env.finger2.position_collection.copy())
        
        total_reward += reward
        done = terminated or truncated

    print(f"Simulation finished. Total Cumulative Reward: {total_reward:.2f}")

    # 3D Animation using Matplotlib
    print("Generating 3D Animation...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize the lines that will represent our soft fingers
    line1, = ax.plot([], [], [], lw=6, color='blue', label='Finger 1')
    line2, = ax.plot([], [], [], lw=6, color='red', label='Finger 2')
    
    # Plot the stationary Target Object
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='green', s=300, marker='*', label='Target')
    
    # Set static axes limits based on the gripper's physical geometry
    ax.set_xlim([-0.2, 0.2])   # Width
    ax.set_ylim([0.0, 0.6])    # Length of the rods (0.5m)
    ax.set_zlim([-0.1, 0.1])   # Height
    
    ax.set_title("Soft Pneumatic Gripper Animation")
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Length/Forward)")
    ax.set_zlabel("Z (Height)")
    ax.legend()
    
    # The update function called for every frame of the video
    def update(frame):
        # Extract the full rod shape for this specific time step
        pos1 = f1_history[frame]
        pos2 = f2_history[frame]
        
        # Update Finger 1 data
        line1.set_data(pos1[0, :], pos1[1, :])
        line1.set_3d_properties(pos1[2, :])
        
        # Update Finger 2 data
        line2.set_data(pos2[0, :], pos2[1, :])
        line2.set_3d_properties(pos2[2, :])
        
        return line1, line2

    # Create the animation (interval=100 means 10 frames per second)
    ani = animation.FuncAnimation(
        fig, update, frames=len(f1_history), interval=100, blit=False
    )
    
    plt.show()