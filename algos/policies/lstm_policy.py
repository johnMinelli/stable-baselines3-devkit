from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F  # noqa: N812
from bitsandbytes import optim
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from algos.policies.modules.lstm import LSTM


class LSTMPolicy(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Union[float, Schedule],
        dim_model: int = 512,
        lstm_hidden: int = 512,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        final_ffn: bool = False,
        action_horizon: int = 12,
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

        self.dim_model = dim_model
        self.lstm_hidden = lstm_hidden
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        self.n_joints = action_space.shape[-1]
        self.action_horizon = action_horizon
        # vars for masked inference (robustness test)
        self.min_mask_duration = 2
        self.max_mask_duration = 15
        self.mask_probability = 0.0

        # LSTM backbone network
        self.lstm_backbone = LSTM(
            image_shape=observation_space["images"].shape,
            state_dim=observation_space["state"].shape[0],
            dim_model=dim_model,
            lstm_hidden=lstm_hidden,
            n_lstm_layers=n_lstm_layers,
            dropout=dropout,
            final_ffn=final_ffn,
        )

        # Action head - outputs action deltas
        self.action_net = nn.Linear(self.dim_model, self.n_joints)

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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def compute_loss(self, obs, actions):
        """
        SL agent forward method
        """
        state, images, tokenized_prompt, tokenized_prompt_mask = obs["state"], obs["images"], obs.get("tokenized_prompt"), obs.get("tokenized_prompt_mask")  # noqa: F841

        if len(images.shape) == 5:
            images = images.unsqueeze(1)

        bs, time_seq, num_cameras, c, h, w = images.shape

        lstm_features = self.lstm_backbone(images, state)  # (bs, time_seq, dim_model)
        joint_deltas = self.action_net(lstm_features)  # (bs, time_seq, n_joints)

        mse_loss = F.mse_loss(joint_deltas, actions, reduction="none")
        mse_mask = (torch.logical_and(obs["expert_mask"], ~obs["action_is_pad"]) if "expert_mask" in obs else ~obs["action_is_pad"]) if "action_is_pad" in obs else torch.ones_like(mse_loss, dtype=torch.bool)

        return {"mse_loss": mse_loss[mse_mask].mean()}

    def _predict(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        pass

    def predict(self, obs: Dict[str, Tensor]) -> Tuple[Tensor, Union[Tuple[Tensor, Tensor], Tuple[None, None]]]:
        """
        Predict action for a single observation.
        This is used during inference/evaluation.
        """
        with torch.no_grad():
            state, images = obs["state"], obs["images"]

            if len(images.shape) == 5:
                images = images.unsqueeze(1)
            elif len(images.shape) == 4:
                images = images.unsqueeze(0).unsqueeze(1)  # Add batch and time dims
                state = state.unsqueeze(0).unsqueeze(1)

            # Forward through LSTM
            lstm_features = self.lstm_backbone(images, state)
            joint_deltas = self.action_net(lstm_features)

            # Return the last timestep prediction
            return joint_deltas[:, -1], (None, None)

    def reset(self):
        self.past_state = deque(maxlen=self.action_horizon - 1)
        self.past_img = deque(maxlen=self.action_horizon - 1)
        self.counter = 0
        self.skip = 1  # do not skip steps
        self.masking_counter = 0
        self.masking_duration = 0
        self.is_masking = False

    @torch.no_grad()
    def predict_seq(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward batched (naive/dumb non-optimized method as we recompute every time the whole sequence returning only the last timestep output)
        """
        # Switch to eval mode
        mode = self.training
        self.set_training_mode(False)

        # Queue management
        if not hasattr(self, "past_state"):
            self.reset()

        self.counter += 1

        state, images, tokenized_prompt, tokenized_prompt_mask = obs["state"], obs["images"], obs.get("tokenized_prompt"), obs.get("tokenized_prompt_mask")  # noqa: F841

        if len(images.shape) == 5:
            state = state.unsqueeze(1)
            images = images.unsqueeze(1)

        # Handle masking logic
        if not self.is_masking and torch.rand(1).item() < self.mask_probability:
            self.is_masking = True
            self.masking_duration = torch.randint(self.min_mask_duration, self.max_mask_duration + 1, (1,)).item()
            self.masking_counter = 0
        if self.is_masking:
            self.masking_counter += 1
            images = torch.zeros_like(images)
            if self.masking_counter >= self.masking_duration:
                self.is_masking = False

        state_seq = torch.cat(list(self.past_state)+[state.float()], dim=1)
        images_seq = torch.cat(list(self.past_img)+[images], dim=1)

        if self.counter%self.skip==0:
            self.past_state.append(state.float())
            self.past_img.append(images)

        # Forward through LSTM
        lstm_features = self.lstm_backbone(images_seq, state_seq)
        joint_deltas = self.action_net(lstm_features)

        self.set_training_mode(mode)

        return joint_deltas[:,-1]
