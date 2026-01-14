from typing import Dict, Union

import numpy as np
import torch
from gymnasium import spaces

from common.datasets.aloha_preprocessor import AlohaPreprocessor


class Aloha_2_Mlp(AlohaPreprocessor):

    def __init__(self, tokenizer_kwargs=None, mask_padding: bool = True, padding_dim: int = 32, use_delta_joint_actions: bool = False, **kwargs):
        self.tokenizer_kwargs = tokenizer_kwargs
        self.mask_padding = mask_padding
        self.padding_dim = padding_dim

        super().__init__(use_delta_joint_actions=use_delta_joint_actions, **kwargs)

        # Hard-imposed spaces
        self.proc_observation_space = spaces.Dict({
            "privileged": spaces.Box(low=-1, high=1, shape=(0,), dtype=np.float32),
            "state": spaces.Box(low=-10, high=10, shape=self.observation_space["state"].shape, dtype=np.float32),
            "images": spaces.Box(low=-1, high=1, shape=(1, 3, 224, 224), dtype=np.float32), })
        self.proc_action_space = spaces.Box(low=-10, high=10, shape=self.action_space.shape, dtype=np.float32)

    def forward(self, observations=None):
        """
        Forward pass through the normalizer. Operation include a preprocessing operation and normalization in this order.

        :param observations: Observation dictionary
        :param actions: Action dictionary
        :return: Tuple of normalized observations and actions
        """
        assert isinstance(observations, dict), "Processing implemented only for `Dict` observations"
        norm_obs = self.normalize_observations(self.preprocess_observations(observations))

        norm_obs["privileged"] = norm_obs["state"][..., norm_obs["state"].shape[-1]-min(self.proc_observation_space["privileged"].shape[-1], norm_obs["state"].shape[-1]) :]
        norm_obs["state"] = norm_obs["state"][..., : self.proc_observation_space["state"].shape[-1]]
        if "images" in norm_obs:
            norm_obs["images"] = torch.stack([norm_obs["images"][k] for k in norm_obs["images"] if k in self.expected_image_keys], dim=-4)
        else:
            norm_obs["images"] = torch.stack([norm_obs[k] for k in norm_obs if k in self.expected_image_keys], dim=-4)
            for k in list(norm_obs.keys()):
                if k in self.expected_image_keys: del norm_obs[k]

        return norm_obs

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        actions = super().forward_post(actions)

        return actions
