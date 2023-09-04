import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Critic Network for PPO algorithm.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.

    Methods:
        forward(state): Forward pass through the network.
    """

    def __init__(self, state_size: int, hidden_size: int) -> None:
        """
        Initializes the critic network.

        Args:
            state_size (int): Dimension of the state space.
            hidden_size (int): Dimension of the hidden layers.

        Returns:
            None.

        """
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network.

        Args:
            state (torch.Tensor): The state.

        Returns:
            x (torch.Tensor): The output of the actor network.

        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
