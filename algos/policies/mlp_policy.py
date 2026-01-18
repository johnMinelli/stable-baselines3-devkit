from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F  # noqa: N812
from bitsandbytes import optim
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from algos.policies.modules.nn_utils import get_activation_mod


class MlpPolicy(BasePolicy):
    """
    Simple MLP Policy for supervised learning (non-actor-critic).
    Structured similar to LSTMPolicy but uses MLP backbone instead of LSTM.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Union[float, Schedule],
        net_arch: List[int] = [512, 512],
        activation: str = "relu",
        dropout: float = 0.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[torch.optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the MLP policy.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule
        :param net_arch: Network architecture (list of layer sizes)
        :param activation: Activation function name
        :param dropout: Dropout probability
        :param features_extractor_class: Features extractor class
        :param features_extractor_kwargs: Features extractor kwargs
        :param optimizer_class: Optimizer class
        :param optimizer_kwargs: Optimizer kwargs
        :param kwargs: Additional arguments
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.net_arch = net_arch
        self.activation = activation
        self.dropout = dropout
        self.n_joints = action_space.shape[-1]

        # Get combined state dimension
        self.features_dim = self.observation_space["state"].shape[0] + self.observation_space["privileged"].shape[0]

        # Build MLP backbone
        mlp_layers: List[nn.Module] = []
        in_dim = self.features_dim

        for layer_size in net_arch:
            mlp_layers.append(nn.Linear(in_dim, layer_size))
            mlp_layers.append(get_activation_mod(activation))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = layer_size

        self.mlp_backbone = nn.Sequential(*mlp_layers)

        # Action head - outputs action deltas
        self.action_net = nn.Linear(net_arch[-1], self.n_joints)

        self.apply(self._init_weights)

        # Setup optimizer with initial learning rate
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-8

        param_groups = [
            {"params": self.parameters(), "lr": lr_schedule},
        ]
        self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)

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

    def compute_loss(self, obs: Dict[str, Tensor], actions: Tensor) -> Dict[str, Tensor]:
        """
        Compute supervised learning loss (MSE).

        :param obs: Observation dictionary
        :param actions: Target actions
        :return: Dictionary with loss values
        """
        state, state_privileged = obs["state"], obs["privileged"]

        # Combine state and privileged information
        combined_state = torch.cat([state, state_privileged], dim=-1)

        # Forward through MLP
        features = self.mlp_backbone(combined_state)
        joint_deltas = self.action_net(features)

        # Compute MSE loss with masking
        mse_loss = F.mse_loss(joint_deltas.squeeze(), actions.squeeze(), reduction="none")
        mse_mask = (torch.logical_and(obs["expert_mask"], ~obs["action_is_pad"]) if "expert_mask" in obs else ~obs["action_is_pad"]) if "action_is_pad" in obs else torch.ones_like(mse_loss, dtype=torch.bool)

        return {"mse_loss": mse_loss[mse_mask].mean()}

    def _predict(
        self, obs: Dict[str, Tensor]
    ) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        """Required by BasePolicy but not used."""
        pass

    @torch.no_grad()
    def predict(self, obs: Dict[str, Tensor]) -> Tensor:
        """
        Predict action for observation (inference mode).

        :param obs: Observation dictionary
        :return: Predicted actions
        """
        # Switch to eval mode
        mode = self.training
        self.set_training_mode(False)

        state, state_privileged = obs["state"], obs["privileged"]
        combined_state = torch.cat([state, state_privileged], dim=-1)

        # Forward through MLP
        features = self.mlp_backbone(combined_state)
        joint_deltas = self.action_net(features)

        self.set_training_mode(mode)

        return joint_deltas
