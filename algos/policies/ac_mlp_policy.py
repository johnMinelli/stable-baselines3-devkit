from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from bitsandbytes import optim
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from algos.policies.modules.nn_utils import get_activation_mod


class MlpPolicy(BasePolicy):
    """
    This has been modified to be a simple MLP Policy,
    removing the complex Transformer decoder.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Union[float, Schedule],
        net_arch: List[int] = [256, 256],  # Added for MLP structure
        activation: str = "relu",
        use_log_std: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,  # Note: we garbage-collect additional parameters, so double check the implementation details when using a configuration
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.n_joints = action_space.shape[-1]
        self.use_log_std = use_log_std

        # Get combined state dimension
        self.features_dim = self.observation_space["state"].shape[0] + self.observation_space["privileged"].shape[0]

        # --- Build Policy Network (Actor) ---
        policy_layers: List[nn.Module] = []
        in_dim = self.features_dim
        for layer_size in net_arch:
            policy_layers.append(nn.Linear(in_dim, layer_size))
            policy_layers.append(get_activation_mod(activation))
            in_dim = layer_size

        self.policy_net = nn.Sequential(*policy_layers)
        self.action_dist = DiagGaussianDistribution(self.n_joints)
        self.action_net = nn.Linear(net_arch[-1], self.n_joints)
        if use_log_std:
            self.log_std = nn.Parameter(torch.log(torch.ones(self.n_joints) * 1), requires_grad=True)
        else:
            self.std = nn.Parameter(torch.ones(self.n_joints) * 1, requires_grad=True)

        # --- Build Value Network (Critic) ---
        value_layers: List[nn.Module] = [nn.Linear(self.features_dim, net_arch[0]), get_activation_mod(activation)]
        in_dim = net_arch[0]
        for layer_size in net_arch:
            value_layers.append(nn.Linear(in_dim, layer_size))
            value_layers.append(get_activation_mod(activation))
            in_dim = layer_size

        self.value_net = nn.Sequential(
            *value_layers,
            nn.Linear(net_arch[-1], 1)  # Output a single value
        )

        self.apply(self._init_weights)

        # Orthogonal init for the final action layer
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.constant_(self.action_net.bias, 0)

        # Setup optimizer with initial learning rate
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-8

        actor_params = list(self.policy_net.parameters()) + list(self.action_net.parameters()) + [self.log_std if self.use_log_std else self.std]
        critic_params = list(self.value_net.parameters())

        lr_schedule_list = lr_schedule if isinstance(lr_schedule, list) else [lr_schedule, lr_schedule]

        optim_groups = [
            {"params": actor_params, "lr": lr_schedule_list[0]},
            {"params": critic_params, "lr": lr_schedule_list[1]},
        ]
        self.optimizer = optimizer_class(optim_groups, **optimizer_kwargs)

    def _init_weights(self, module):
        """
        Initialize model weights.

        :param module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, obs: Dict[str, Tensor], deterministic: bool = False, get_distribution=False) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        # expected normalized input and output to be un-normalized
        state, state_privileged = obs["state"], obs["privileged"]

        # Combine state and privileged information
        combined_state = torch.cat([state, state_privileged], dim=-1)

        # --- Actor (Policy) ---
        policy_features = self.policy_net(combined_state)
        joints_means = self.action_net(policy_features)

        # --- Critic (Value) ---
        values = self.value_net(combined_state).squeeze(-1)

        # --- Action Distribution ---
        distribution = self.action_dist.proba_distribution(joints_means, log_std=self.log_std if self.use_log_std else torch.log(self.std))
        joint_deltas = distribution.get_actions(deterministic=deterministic)

        dist_out = distribution if get_distribution else (distribution.log_prob(joint_deltas), distribution.mode(), distribution.distribution.stddev)

        return joint_deltas, values, dist_out

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :return: the estimated values.
        """

        state, state_privileged = obs["state"], obs["privileged"]
        combined_state = torch.cat([state, state_privileged], dim=-1)
        values = self.value_net(combined_state).squeeze(-1)

        return values

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        joint_deltas, values, distribution = self.forward(obs, get_distribution=True)

        dist_out = distribution.log_prob(actions), distribution.mode(), distribution.distribution.stddev
        entropy = distribution.entropy()

        return joint_deltas, values, dist_out, entropy

    def _predict(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        pass

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method.

        :param obs: State with shape (batch_size, len_seq, state_len)
        :return: Tuple of (policy, value) where policy has shape (batch_size, num_action) and value has shape (batch_size, 1)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        mode = self.training
        self.set_training_mode(False)
        joint_deltas, _, _ = self.forward(obs, deterministic=True)
        self.set_training_mode(mode)

        return joint_deltas
