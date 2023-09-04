import numpy as np


class GaussianNoise:
    """
    Add gaussian noise to the input data.
    This is useful for PPO as it adds explorative noise to actions.
    Which helps with learning in the context of continuous action spaces.

    Attributes:
        action_dim (int): Dimension of the action space.
        mean (float): Mean of the gaussian noise.
        sigma (float): Standard deviation of the gaussian noise.
    """

    def __init__(self, action_dim: int, mean: float = 0.0, sigma: float = 0.1):
        """
        Initialize gaussian noise instance.

        Args:
            action_dim (int): Dimension of the environment action space.
            mean (float): Mean of the gaussian noise to control bias.
            sigma (float): Standard deviation of the gaussian noise.

        Returns:
            None.

        """
        assert (
            isinstance(action_dim, int) and action_dim > 0
        ), "action_dim must be a positive integer"

        self.action_dim = action_dim
        self.mean = mean
        self.sigma = sigma

    def sample_noise(self):
        """
        Samples gaussian noise.

        Returns:
            action (np.array): Continuous action with gaussian noise.

        """
        noise = np.random.normal(loc=self.mean, scale=self.sigma, size=self.action_dim)
        return noise

    def add_noise(self, action: np.array) -> np.array:
        """
        Adds gaussian noise to the input action.

        Args:
            action (np.array): Continuous action.

        Returns:
            noisy_action (np.array): Continuous action with gaussian noise.

        """
        assert (
            action.shape[-1] == self.action_dim
        ), "action must have the same dimension as action_dim"

        noisy_action = action + self.sample_noise()
        return noisy_action
