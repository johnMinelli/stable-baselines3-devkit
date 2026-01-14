from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from nn_utils import get_activation_mod
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule

from algos.policies.modules.transformer import GatedTransformerXL


class ACTransformerPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Union[float, Schedule],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation: str = "tanh",
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        embed_dim: int = 256,
        **kwargs: Any,  # Note: we garbage-collect additional parameters, so double check the implementation details when using a configuration
    ):
        """
        Initialize the Actor-Critic Transformer Policy.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (can be a list [actor_lr, critic_lr])
        :param net_arch: Network architecture configuration
        :param activation: Activation function class
        :param ortho_init: Whether to use orthogonal initialization
        :param use_sde: Whether to use state-dependent exploration
        :param log_std_init: Initial value for log standard deviation
        :param full_std: Whether to use full covariance matrix for SDE
        :param use_expln: Whether to use exponential log for SDE
        :param squash_output: Whether to squash output (only with SDE)
        :param features_extractor_class: Features extractor class
        :param features_extractor_kwargs: Features extractor kwargs
        :param normalize_images: Whether to normalize images
        :param optimizer_class: Optimizer class
        :param optimizer_kwargs: Optimizer kwargs
        :param embed_dim: Embedding dimension
        :param kwargs: Additional keyword arguments
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.net_arch = net_arch
        self.activation = activation
        self.ortho_init = ortho_init
        self.embed_dim = embed_dim
        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        # backbone
        # self.fc = self._layer_init(nn.Linear(observation_space.shape[0], hidden_size), std=np.sqrt(2))
        self.fc = nn.Identity()
        self.transformer = GatedTransformerXL(input_size=self.observation_space["state"].shape[0]+self.observation_space["privileged"].shape[0], embed_dim=embed_dim, **kwargs)
        self.tr_num_blocks = self.transformer.num_blocks

        # action head (sampling from distribution in case of LSTM SB3 implementation ow policy network)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=embed_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=embed_dim, latent_sde_dim=embed_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=embed_dim)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # ow

        # self.action_net = nn.Sequential(
        #     self._layer_init(nn.Linear(embed_dim, embed_dim), std=np.sqrt(2)),
        #     nn.GELU(),
        #     self._layer_init(nn.Linear(embed_dim, action_space.shape[0]), std=0.01),
        # )

        # value head
        self.value_net = nn.Sequential(
            self._layer_init(nn.Linear(embed_dim, embed_dim), std=np.sqrt(2)),
            get_activation_mod(activation),
            self._layer_init(nn.Linear(embed_dim, 1), std=1),
        )

        # Xavier-uniform initialization of the transformer parameters as in the original code
        for p in chain(self.transformer.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Setup optimizer with initial learning rate
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        # Support separate learning rates for actor and critic
        actor_params = list(self.transformer.parameters()) + list(self.action_net.parameters())
        if hasattr(self, 'log_std'):
            actor_params.append(self.log_std)
        critic_params = list(self.value_net.parameters())

        lr_schedule_list = lr_schedule if isinstance(lr_schedule, list) else [lr_schedule, lr_schedule]

        optim_groups = [
            {"params": actor_params, "lr": lr_schedule_list[0]},
            {"params": critic_params, "lr": lr_schedule_list[1]},
        ]
        self.optimizer = optimizer_class(optim_groups, **optimizer_kwargs)

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """
        Initialize layer weights and biases with constraints.

        :param layer: Layer to initialize
        :param std: Standard deviation for orthogonal initialization
        :param bias_const: Constant value for bias initialization
        :return: Initialized layer
        """
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        if isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)

        raise ValueError("Invalid action distribution")

    def forward(
        self, obs: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor, memory_indices: torch.Tensor, deterministic: bool = False, get_distribution=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        :param obs: Observation tensor with shape (batch_size, obs_dim)
        :param memory: Transformer memory tensor
        :param memory_mask: Memory mask tensor
        :param memory_indices: Memory indices tensor
        :param deterministic: Whether to use deterministic actions
        :return: Tuple of (actions, values, log_prob, memory, memory_mask)
        """
        state, state_privileged = obs["state"], obs["privileged"]

        # Combine state and privileged information
        combined_state = torch.cat([state, state_privileged], dim=-1)

        x = self.fc(combined_state)
        out, memory = self.transformer(x, memory, memory_mask, memory_indices)
        out = out.squeeze(1)
        values = self.value_net(out).squeeze()
        distribution = self._get_action_dist_from_latent(out)
        actions = distribution.get_actions(deterministic=deterministic)

        dist_out = distribution if get_distribution else (distribution.log_prob(actions), distribution.distribution.mode, distribution.distribution.stddev)

        return actions, values, dist_out, memory, memory_mask

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation tensor
        :param lstm_states: Last hidden and memory states (unused in transformer)
        :param episode_starts: Whether observations correspond to new episodes (unused)
        :return: Action distribution and new hidden states
        """
        # Not implemented - use forward() method instead
        pass

    def predict_values(
        self,
        obs: torch.Tensor,
        memory: torch.Tensor,
        memory_mask,
        memory_indices,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy.

        :param obs: Observation tensor
        :param memory: Transformer memory tensor
        :param memory_mask: Memory mask tensor
        :param memory_indices: Memory indices tensor
        :return: Estimated values
        """
        state, state_privileged = obs["state"], obs["privileged"]

        # Combine state and privileged information
        combined_state = torch.cat([state, state_privileged], dim=-1)

        x = self.fc(combined_state)
        out, memory = self.transformer(x, memory, memory_mask, memory_indices)
        out = out.squeeze(1)
        values = self.value_net(out).squeeze(-1)

        return values

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor, memory_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Evaluate actions according to the current policy, given the observations.

        :param obs: Observation tensor
        :param actions: Actions to evaluate
        :param memory: Transformer memory tensor
        :param memory_mask: Memory mask tensor
        :param memory_indices: Memory indices tensor
        :return: Tuple of (actions, values, (log_prob, action_mean, action_std), entropy)
        """
        state, state_privileged = obs["state"], obs["privileged"]

        # Combine state and privileged information
        combined_state = torch.cat([state, state_privileged], dim=-1)

        x = self.fc(combined_state)
        out, _ = self.transformer(x, memory, memory_mask, memory_indices)
        out = out.squeeze(1)
        values = self.value_net(out).squeeze()
        distribution = self._get_action_dist_from_latent(out)
        dist_out = distribution.log_prob(actions), distribution.distribution.mean, distribution.distribution.stddev
        entropy = distribution.entropy()

        # Return sampled actions for consistency
        sampled_actions = distribution.get_actions(deterministic=False)

        return sampled_actions, values, dist_out, entropy

    def _predict(self):
        pass

    @torch.no_grad()
    def predict(
        self, obs: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor, memory_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict actions for inference (evaluation mode).

        :param obs: Observation tensor
        :param memory: Transformer memory tensor
        :param memory_mask: Memory mask tensor
        :param memory_indices: Memory indices tensor
        :return: Tuple of (actions, updated_memory, updated_memory_mask)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        mode = self.training
        self.set_training_mode(False)
        actions, _, _, memory, memory_mask = self.forward(obs, memory, memory_mask, memory_indices, deterministic=True)
        self.set_training_mode(mode)

        return actions, memory, memory_mask
