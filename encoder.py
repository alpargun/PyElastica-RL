import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class StateEncoder(BaseFeaturesExtractor):
    """
    Process the 3D coordinates  of the soft gripper and the target.
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Get the number of inputs (9 coordinates)
        n_input_channels = observation_space.shape[0]
        
        # MLP (shared by both Actor and Critic)
        self.net = nn.Sequential(
            nn.Linear(n_input_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)