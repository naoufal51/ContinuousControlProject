import torch
import numpy as np
from torch import optim
from torch.nn.utils import clip_grad_norm_
from actor_critic_model import ContinuousActorCriticNetwork
from utils import generate_batches
import torch.nn.functional as F
from typing import List, Tuple


class PPOContinuous:
    """
    PPO agent.
    It is trained using the PPO (Proximal Policy Optimization) algorithm.
    PPO is part of the policy gradient family, which means it directly optmizes the policy by estimating the gradient of the reward with respect to the policy parameters.
    It uses clipped surrogate loss to train the policy. Which ensures that new policy is not so far away from the old policy.
    It is known for its efficiency and stability.
    Given our Environment type, we choose to support continuous action space.
    """

    def __init__(
        self,
        state_size,
        action_size,
        n_epochs=10,
        gamma=0.99,
        clip_epsilon=0.2,
        lr=1e-4,
        action_std_bound=[1e-2, 1.0],
        normalize_advantage=True,
        tau=0.95,
        logger=None,
        value_coef=0.5,
        entropy_coef=0.01,
        max_norm=0.5,
    ):
        """
        Initialize PPO agent.

        Args:
            state_size (int): The state space dimension.
            action_size (int): The action space dimension.
            n_epochs (int): Number of epochs to train the agent.
            gamma (float): Discount factor for computing the td-loss.
            clip_epsilon (float): Clipped surrogate loss, used for PPO to ensure learning stabilty.
            lr (float): Learning rate.
            action_std_bound (list): Lower and upper bound of the action space.
            normalize_advantage (bool): Whether to normalize the advantage (GAE in our case)
            tau (float): Parameter for GAE computation.
            logger (object): Logger object for logging (Wandb for instance)
            value_coef (float): Coefficient for value loss.
            entropy_coef (float): Coefficient for entropy loss.
            max_norm (float): Maximum gradient norm, used for gradient clipping.

        """
        self.actor_critic = ContinuousActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.tau = tau
        self.logger = logger
        self.clip_epsilon = clip_epsilon
        self.action_std_bound = action_std_bound
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm

        self.normalize_advantage = normalize_advantage
        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        ) = ([], [], [], [], [], [])

    def act(self, state: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the actions and the log probabilities of the given states.

        Args:
            state (List): List of states.

        Returns:
            sampled_action (np.ndarray): numpy array containing the sampled actions.
            action_log_prob (np.ndarray): numpy array containing the action log probabilities.
        """
        state = torch.tensor(state).float()
        with torch.no_grad():
            mu, log_std, _ = self.actor_critic(state)
            action_dist = torch.distributions.Normal(mu, log_std.exp())
            sampled_action = action_dist.sample().detach()
            action_log_prob = action_dist.log_prob(sampled_action).sum(dim=-1)
        return sampled_action.numpy(), action_log_prob.numpy()

    def memorize(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
    ) -> None:
        """
        Memorize the given trajectories/experiences.

        Args:
             state (np.ndarray): numpy array containing the states.
             action (float): float containing the action.
             reward (float): float containing the reward.
             next_state (np.ndarray): numpy array containing the next states.
             done (bool): bool indicating if the state is terminal.
             log_prob (float): float containing the log probability of the action.

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE) for policy gradient optimization.

        GAE is a technique that stabilizes the advantage function by averaging over
        multiple n-step advantage estimates. The goal is to reduce variance in the
        advantage estimate without introducing much bias.

        The GAE for time step 't' is formulated as:

        A^{GAE(γ, τ)}_t = Σ_{l=0}^{∞} (γτ)^l δ_{t+l}
        with δ_t = r_t + γV(s_{t+1}) - V(s_t)

        where:
        - δ_t represents the Temporal Difference (TD) error.
        - r_t is the reward at time t.
        - V(s) is the value function of state s.
        - γ is the discount factor for future rewards.
        - τ is a hyperparameter for GAE, weighting different n-step estimators.

        Args:
            rewards (torch.Tensor): Tensor of rewards for the trajectories.
            values (torch.Tensor): Tensor of value estimates for each state in trajectories.
            next_values (torch.Tensor): Tensor of value estimates for each next state in trajectories.
            dones (torch.Tensor): Tensor indicating if the state is terminal (1 if terminal, 0 otherwise).

        Returns:
            advantages Tensor: GAE advantages for each state.
        """

        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        advantage = torch.tensor(0.0)

        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + self.gamma * self.tau * advantage
            advantages[t] = advantage

        return advantages

    def update(self, batch_size=64):
        """
        Update the ppo agent.
        The actor critic networks are trained on batches of collected data for the current policy.
        We log the relevant metrics in wandb for further analysis.

        Args:
            batch_size (int): The number of samples to use for training.

        """

        # Convert the collected data to tensors
        states = torch.tensor(np.vstack(self.states)).float()
        actions = torch.tensor(np.vstack(self.actions)).float()
        rewards = torch.tensor(np.vstack(self.rewards)).float().squeeze(-1)
        next_states = torch.tensor(np.vstack(self.next_states)).float()
        dones = torch.tensor(np.vstack(self.dones)).float().squeeze(-1)
        old_log_probs = torch.tensor(np.vstack(self.log_probs)).float().squeeze(-1)

        entropy_losses = []
        policy_gradient_losses = []
        value_losses = []
        approx_kls = []
        explained_variances = []
        clip_fractions = []
        stds = []

        # Compute Actor-Critic current and next states values using the actor-critic model
        with torch.no_grad():
            _, _, values = self.actor_critic(states)
            _, _, next_values = self.actor_critic(next_states)

        # Compute GAE (Generalized Advantage Estimation)
        advantages = self.compute_gae(
            rewards, values.squeeze(), next_values.squeeze(), dones
        )

        # Compute TD targets for Value Estimator
        td_targets = advantages + values.squeeze()

        # Normalize advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            for (
                batch_states,
                batch_actions,
                _,
                _,
                _,
                batch_old_log_probs,
                start,
                end,
            ) in generate_batches(
                batch_size, states, actions, rewards, next_states, dones, old_log_probs
            ):
                # compute action log probabilities and entropy loss
                mu, log_std, value_estimate = self.actor_critic(batch_states)
                log_std = torch.clamp(
                    log_std, self.action_std_bound[0], self.action_std_bound[1]
                )
                action_dist = torch.distributions.Normal(mu, log_std.exp())
                action_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                entropy_loss = action_dist.entropy().mean()

                # compute ratio between new and old action log probabilities (new and old policy)
                ratios = (action_log_probs - batch_old_log_probs).exp()

                # compute surrogate loss for PPO (Small policy update)
                batch_advantages = advantages[start:end]
                surrogate_loss = -torch.min(
                    ratios * batch_advantages,
                    torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages,
                ).mean()

                # Compute value loss for Value Estimator
                batch_td_targets = td_targets[start:end]
                value_loss = F.mse_loss(value_estimate.squeeze(), batch_td_targets)

                # Compute gradient and perform a single step of gradient descent on the actor-critic model
                # We clip the norm of the gradients to prevent gradient explosion
                self.optimizer.zero_grad()
                (
                    surrogate_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                ).backward()
                clip_grad_norm_(self.actor_critic.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

                # Compute approximate KL divergence to track the difference between old and new policy
                log_ratio = action_log_probs - batch_old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

                # Compute clip fraction
                clip_fraction = (
                    (
                        (ratios < 1 - self.clip_epsilon)
                        | (ratios > 1 + self.clip_epsilon)
                    )
                    .float()
                    .mean()
                    .item()
                )

                # Explained variance
                y_true = batch_td_targets.numpy()
                y_pred = value_estimate.detach().squeeze().numpy()
                mask = ~np.isnan(y_true)
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)

                # Store metrics
                entropy_losses.append(entropy_loss.item())
                policy_gradient_losses.append(surrogate_loss.item())
                value_losses.append(value_loss.item())
                approx_kls.append(approx_kl)
                explained_variances.append(explained_var)
                clip_fractions.append(clip_fraction)
                if hasattr(self.actor_critic, "actor_log_std"):
                    stds.append(torch.exp(log_std).mean().item())

        if self.logger:
            self.logger.log({"mean_entropy_loss": np.mean(entropy_losses)})
            self.logger.log(
                {"mean_policy_gradient_loss": np.mean(policy_gradient_losses)}
            )
            self.logger.log({"mean_value_loss": np.mean(value_losses)})
            self.logger.log({"mean_approx_kl": np.mean(approx_kls)})
            self.logger.log({"mean_explained_variance": np.mean(explained_variances)})
            self.logger.log({"mean_clip_fraction": np.mean(clip_fractions)})
            if stds:
                self.logger.log({"mean_std": np.mean(stds)})

        (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.dones,
            self.log_probs,
        ) = ([], [], [], [], [], [])
