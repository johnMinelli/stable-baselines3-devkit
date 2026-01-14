from typing import Dict, Union

import numpy as np
import torch
from gymnasium import spaces

from common.envs.gym_preprocessor import GymPreprocessor


class Maniskill_2_Mlp(GymPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.task == "StackCube-v1":
            self.proc_observation_space = spaces.Dict({
                "privileged": spaces.Box(low=-1, high=1, shape=(30,), dtype=np.float32),
                "state": spaces.Box(low=-10, high=10, shape=(9,), dtype=np.float32),
                "images": spaces.Box(low=-1, high=1, shape=(2, 3, 94, 94), dtype=np.float32),
            })

    def _prepare_action(self, action=None):
        """
        Process action similar to GR00TTransform's _prepare_action method.
        """
        # Handle case where action is missing
        if action is None:
            action = torch.zeros((self.action_horizon, self.max_action_dim), device=self.device)
            action_mask = torch.zeros((self.action_horizon, self.max_action_dim), device=self.device, dtype=torch.bool)
            return action, action_mask

        # Convert to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)

        # Ensure correct shape
        if action.dim() == 2:  # Already has time dimension
            pass
        else:  # Add time dimension if missing
            action = action.unsqueeze(0)

        n_action_dims = action.shape[-1]

        # Verify action dimensions
        assert n_action_dims <= self.max_action_dim, f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        pad_size = self.max_action_dim - n_action_dims
        if pad_size > 0:
            action = torch.cat([action, torch.zeros((action.shape[0], pad_size), device=self.device)], dim=1)

        # Create mask for real action dims
        action_mask = torch.zeros_like(action, dtype=torch.bool)
        action_mask[:, :n_action_dims] = True

        return action, action_mask

    def forward(self, observations=None):
        assert isinstance(observations, dict), "Processing implemented only for `Dict` observations"
        norm_obs = self.normalize_observations(self.preprocess_observations(observations))

        # Truncate taking only the firsts elements which are the joint positions as requested by the policy
        norm_obs["privileged"] = norm_obs["state"][..., norm_obs["state"].shape[-1]-min(self.proc_observation_space["privileged"].shape[-1], norm_obs["state"].shape[-1]) :]
        if self.task == "TwoRobotStackCube-v1":
            norm_obs["state"] = torch.cat([norm_obs["state"][..., :18], norm_obs["state"][..., 18 : 18 + 18]], -1)
        else:
            norm_obs["state"] = norm_obs["state"][..., : self.proc_observation_space["state"].shape[-1]]
        if "images" in norm_obs:
            norm_obs["images"] = torch.stack([norm_obs["images"][k] for k in norm_obs["images"]], dim=-4)

        return norm_obs

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        actions = super().forward_post(actions)

        return actions
