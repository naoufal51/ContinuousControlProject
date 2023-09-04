import torch
import torch.nn as nn
import torch.nn.Functional as F


class Actor(nn.Module):
    """
    The actor network maps the state to the actions.
    For our environment, the actor's output should be a vector of four values (torques), each between -1 and 1.
    Torque is the amount of force applied to the joint to move it in a particular direction.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.

    Methods:
        forward(state): The forward pass of the actor network.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        """
        Initializes the actor network.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            hidden_size (int): The size of the hidden layers.

        Returns:
            None.

        """
        super(Actor, self).__init__()  # This is the initialization of the parent class.

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the actor network.

        Args:
            state (torch.Tensor): The state.

        Returns:
            x (torch.Tensor): The output of the actor network.

        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
