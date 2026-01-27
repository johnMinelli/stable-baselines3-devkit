"""
FastSAC Policy - Distributional Soft Actor-Critic

This implementation uses distributional Q-learning (C51-style) instead of
point-estimate Q-values, providing better value estimates and more stable training.

Key differences from standard SAC:
- Critic outputs num_atoms logits (distribution over returns) instead of single Q-value
- Uses categorical cross-entropy loss instead of MSE
- Network architecture: SiLU activation, LayerNorm, decreasing dimensions
"""

from typing import Any, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

# CAP the standard deviation of the actor
LOG_STD_MAX = 0.0
LOG_STD_MIN = -5.0


class FastSACActor(BasePolicy):
    """
    Actor network (policy) for FastSAC.

    Uses a decreasing-dimension MLP architecture (hidden -> hidden/2 -> hidden/4)
    with SiLU activation and optional LayerNorm for stable training.
    Output layers are zero-initialized for better initial exploration.

    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_dim: Hidden dimension for the first layer (subsequent layers decrease)
    :param features_dim: Number of input features (if None, computed from observation_space)
    :param use_layer_norm: Whether to use LayerNorm after each linear layer. Defaults to True.
    :param log_std_init: Initial value for the log standard deviation. Defaults to -3.
    :param use_tanh: Whether to use tanh squashing for actions. Defaults to True.
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim: int,
        features_dim: int = None,
        use_layer_norm: bool = True,
        log_std_init: float = -3,
        use_tanh: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            squash_output=True,
        )

        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.log_std_init = log_std_init
        self.use_tanh = use_tanh

        # Calculate input dimension from observation space
        if features_dim is not None:
            input_dim = features_dim
        elif isinstance(observation_space, spaces.Dict):
            input_dim = sum(
                space.shape[0] for space in observation_space.spaces.values() if isinstance(space, spaces.Box)
            )
        else:
            input_dim = observation_space.shape[0]

        action_dim = get_action_dim(self.action_space)

        # Build network with decreasing dimensions: hidden -> hidden/2 -> hidden/4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )

        # Output heads for mean and log_std
        self.mu = nn.Linear(hidden_dim // 4, action_dim)
        self.log_std = nn.Linear(hidden_dim // 4, action_dim)

        # Zero initialization for output layers (important for stable training)
        nn.init.constant_(self.mu.weight, 0.0)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.bias, 0.0)

        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def get_action_dist_params(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Get the parameters for the action distribution."""
        state, state_privileged = obs["state"], obs["privileged"]

        combined_state = torch.cat([state, state_privileged], dim=-1)
        x = self.net(combined_state)

        mean_actions = self.mu(x)
        log_std = self.log_std(x)

        # Clamp log_std using tanh transformation (from SpinUp / Denis Yarats)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action and associated log probability."""
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self(observation, deterministic)


class DistributionalQNetwork(nn.Module):
    """
    Distributional Q-Network (C51-style) that outputs a categorical distribution
    over returns instead of a single Q-value.

    The network outputs num_atoms logits representing the probability distribution
    over a fixed set of "atoms" (support) spanning [v_min, v_max].
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device = None,
    ):
        super().__init__()

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.device = device

        # Network with decreasing dimensions
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms),
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [batch, num_atoms]."""
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)

    def get_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns probability distribution over atoms."""
        logits = self.forward(obs, actions)
        return F.softmax(logits, dim=-1)


class FastSACCritic(BaseModel):
    """
    Distributional Critic network for FastSAC.

    Contains multiple DistributionalQNetwork instances (C51-style) that output
    categorical distributions over returns. Handles the distributional Bellman
    projection for target computation.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_dim: Hidden dimension for the first layer (subsequent layers decrease). Defaults to 512.
    :param num_atoms: Number of atoms for the categorical distribution. Defaults to 51.
    :param v_min: Minimum value of the support. Defaults to -10.0.
    :param v_max: Maximum value of the support. Defaults to 10.0.
    :param n_critics: Number of critic networks to create. Defaults to 2.
    :param use_layer_norm: Whether to use LayerNorm after each linear layer. Defaults to True.
    :param features_dim: Number of input features (if None, computed from observation_space)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        hidden_dim: int = 512,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_critics: int = 2,
        use_layer_norm: bool = True,
        features_dim: int = None,
    ):
        super().__init__(
            observation_space,
            action_space,
        )

        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_critics = n_critics
        self.use_layer_norm = use_layer_norm

        # Calculate input dimension
        if features_dim is not None:
            n_obs = features_dim
        elif isinstance(observation_space, spaces.Dict):
            n_obs = sum(space.shape[0] for space in observation_space.spaces.values() if isinstance(space, spaces.Box))
        else:
            n_obs = observation_space.shape[0]

        action_dim = get_action_dim(action_space)

        # Create Q-networks
        self.q_networks = nn.ModuleList([
            DistributionalQNetwork(
                n_obs=n_obs,
                n_act=action_dim,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(n_critics)
        ])

        # Register support as buffer (fixed set of atoms)
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("support", support)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Returns logits from all Q-networks.
        Shape: [n_critics, batch, num_atoms]
        """
        state, state_privileged = obs["state"], obs["privileged"]

        combined_state = torch.cat([state, state_privileged], dim=-1)
        outputs = [qnet(combined_state, actions) for qnet in self.q_networks]
        return torch.stack(outputs, dim=0)

    def get_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns probability distributions from all Q-networks."""
        logits = self.forward(obs, actions)
        return F.softmax(logits, dim=-1)

    def get_q_values(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q-values from the distributions.
        Shape: [n_critics, batch]
        """
        probs = self.get_probs(obs, actions)
        return torch.sum(probs * self.support, dim=-1)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the distributional Bellman projection for target distribution.

        This projects the next-state distribution onto the fixed support,
        accounting for rewards and discount factor.

        Returns: [n_critics, batch, num_atoms]
        """
        state, state_privileged = obs["state"], obs["privileged"]

        combined_state = torch.cat([state, state_privileged], dim=-1)
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        # Compute projected support: r + gamma * z
        # bootstrap is 0 for terminal states, 1 otherwise
        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * self.support
        target_z = target_z.clamp(self.v_min, self.v_max)

        # Compute projection indices
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        # Handle edge case where b is exactly an integer
        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        # Get next state distributions from all Q-networks
        next_dists = [F.softmax(qnet(combined_state, actions), dim=-1) for qnet in self.q_networks]

        projections = []
        for next_dist in next_dists:
            proj_dist = torch.zeros_like(next_dist)

            # Create offset for batch indexing
            offset = (
                torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .long()
            )

            # Safety check for indices
            lower_indices = (lower + offset).view(-1)
            upper_indices = (upper + offset).view(-1)
            max_index = proj_dist.numel() - 1

            lower_indices = torch.clamp(lower_indices, 0, max_index)
            upper_indices = torch.clamp(upper_indices, 0, max_index)

            # Distribute probability mass
            proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))

            projections.append(proj_dist)

        return torch.stack(projections, dim=0)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass using only the first Q-network."""
        state, state_privileged = obs["state"], obs["privileged"]

        combined_state = torch.cat([state, state_privileged], dim=-1)
        return self.q_networks[0](combined_state, actions)


class FastSACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for FastSAC.

    Combines a FastSACActor with a distributional FastSACCritic (C51-style).
    Uses SiLU activation, LayerNorm, and decreasing-dimension architecture.

    :param observation_space: Observation space.
    :param action_space: Action space.
    :param lr_schedule: Learning rate schedule (could be constant).
    :param actor_hidden_dim: Hidden dimension for actor network. Defaults to 512.
    :param critic_hidden_dim: Hidden dimension for critic network. Defaults to 768.
    :param num_atoms: Number of atoms for the categorical distribution. Defaults to 51.
    :param v_min: Minimum value of the support. Defaults to -10.0.
    :param v_max: Maximum value of the support. Defaults to 10.0.
    :param n_critics: Number of critic networks to create. Defaults to 2.
    :param use_layer_norm: Whether to use LayerNorm after each linear layer. Defaults to True.
    :param use_tanh: Whether to use tanh squashing for actions. Defaults to True.
    :param optimizer_class: The optimizer to use, ``torch.optim.AdamW`` by default.
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer.
    """

    actor: FastSACActor
    critic: FastSACCritic
    critic_target: FastSACCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 768,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        n_critics: int = 2,
        use_layer_norm: bool = True,
        use_tanh: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_critics = n_critics
        self.use_layer_norm = use_layer_norm
        self.use_tanh = use_tanh

        self.actor = FastSACActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_dim=self.actor_hidden_dim,
            use_layer_norm=self.use_layer_norm,
            use_tanh=self.use_tanh,
        ).to(self.device)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **(self.optimizer_kwargs or {}),
        )

        # Build Critic
        self.critic = FastSACCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_dim=self.critic_hidden_dim,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            n_critics=self.n_critics,
            use_layer_norm=self.use_layer_norm,
        ).to(self.device)

        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),
            **(self.optimizer_kwargs or {}),
        )

        # Build Critic Target (copy of critic)
        self.critic_target = FastSACCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_dim=self.critic_hidden_dim,
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            n_critics=self.n_critics,
            use_layer_norm=self.use_layer_norm,
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic)

    @torch.no_grad()
    def predict(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Forward method.

        :param obs: State with shape (batch_size, len_seq, state_len)
        :return: Policy output with shape (batch_size, num_action)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        mode = self.training
        self.set_training_mode(False)
        joint_deltas = self.forward(obs, deterministic=True)
        self.set_training_mode(mode)

        return joint_deltas

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode
