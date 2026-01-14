from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import MultiDiscrete
from gymnasium import spaces

from common.preprocessor import Preprocessor


class GymPreprocessor(Preprocessor):
    """
    Normalizer for Aloha dataset features that can be used independently of the dataloader.

    Encoding input: Preprocess (standard + env-specific data transformations) -> Normalize -> (opt) superclass operations (policy-specific)
    Decoding output: (opt) superclass operations (policy-specific) -> De-normalize -> Postprocess
    """

    def __init__(self, squash_output: bool = False, drop_images: bool = False,
                 discrete_actions: Union[bool, int] = False, **kwargs):
        """
        Initialize the normalizer.

        :param squash_output: Whether to compress the output range to the action space min/max instead of applying clipping
        :param drop_images: Whether to drop the images in the observation
        :param discrete_actions: Use a MultiDiscrete action space discretizing each dimension into n independent tokens
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        self.squash_output = squash_output
        self.drop_images = drop_images
        self.discrete_actions = bool(discrete_actions)

        self.action_space_high = torch.tensor(self.action_space.high, device=self.device)
        self.action_space_low = torch.tensor(self.action_space.low, device=self.device)

        if isinstance(self.observation_space, spaces.Box):
            self.proc_observation_space = self.observation_space
        elif not self.drop_images and "images" in self.observation_space.spaces:
            if isinstance(self.observation_space["images"], spaces.Box):
                self.proc_observation_space = spaces.Dict({
                    "state": self.observation_space["state"],
                    "images": self.observation_space["images"]})
            else:
                self.proc_observation_space = spaces.Dict({
                    "state": self.observation_space["state"],
                    "images": spaces.Box(low=0, high=255, shape=(
                        sum([cam_img.shape[2] for cam_img in self.observation_space["images"].values()]),
                        *list(self.observation_space["images"].values())[0].shape[:2]), dtype=np.uint8) if len(self.observation_space["images"].values()) else spaces.Box(low=0, high=255, shape=(0, 0, 0), dtype=np.uint8)})
        else:
            self.proc_observation_space = self.observation_space["state"]

        if discrete_actions:
            self.n_tokens_per_dim = discrete_actions
            self.proc_action_space = MultiDiscrete(
                [self.n_tokens_per_dim] * self.action_space.shape[0])  # joint-dimensions * `n_tokens_per_dim` tokens each
        else:
            self.proc_action_space = self.action_space

    def preprocess_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Aloha-specific preprocessing to raw observations from dataset.

        :param observations: Dictionary with observation data
        :return: Preprocessed observations dictionary
        """

        # Process observations
        processed_obs = {}
        for key, _obs in observations.items():
            if "images" in key:
                if self.drop_images:
                    continue
                if isinstance(_obs, dict):  # handle dictionary of images
                    processed_images = {}
                    for cam_name, img in _obs.items():
                        # cconvert to torch tensor if needed
                        if not isinstance(img, torch.Tensor):
                            img = torch.tensor(img, device=self.device)
                        # ensure channel-first format
                        if img.dim() == 4 and img.shape[1] not in [1, 3, 4] and img.shape[-1] in [1, 3, 4]:
                            img = img.permute(0, 3, 1, 2)
                        if self.normalize_images:
                            if img.max() > 1:
                                img = img.float() / 255.0
                            img = img.float() * 2.0 - 1.0
                        if self.resize_images:
                            assert img.dtype != torch.uint8, "Resizing can not be performed of uint8 data type. Consider setting `normalize_images` true."
                            img = F.interpolate(img, size=self.resize_images, mode="bilinear", align_corners=False)
                        processed_images[cam_name] = img.to(self.device)
                    processed_obs[key] = processed_images
                else:  # handle single image
                    if not isinstance(_obs, torch.Tensor):
                        _obs = torch.tensor(_obs, device=self.device)
                    if _obs.dim() == 4 and _obs.shape[1] in [1,3,4] and _obs.shape[-1] in [1,3,4]:
                        _obs = _obs.permute(0, 3, 1, 2)
                    if self.normalize_images:
                        if _obs.max() > 1:
                            _obs = _obs.float() / 255.0
                        _obs = _obs.float() * 2.0 - 1.0
                    if self.resize_images:
                        _obs = F.interpolate(_obs, size=self.resize_images, mode="bilinear", align_corners=False)
                    processed_obs[key] = _obs.to(self.device)
            elif key == "prompt" and self.tokenizer is not None:
                tokens, token_masks = self.tokenizer.tokenize_batch(_obs)
                if len(tokens.shape) != 2:
                    tokens = tokens.unsqueeze(0).repeat(observations["state"].size(0), 1)
                    token_masks = token_masks.unsqueeze(0).repeat(observations["state"].size(0), 1)
                processed_obs["tokenized_prompt"] = tokens
                processed_obs["tokenized_prompt_mask"] = token_masks
            else:  # If key doesn't exist in spaces, just pass it through
                processed_obs[key] = _obs

        if "prompt" not in processed_obs and self.prompt_generator:
            processed_obs["prompt"] = self.prompt_generator.generate()
            if self.tokenizer:
                tokens, token_masks = self.tokenizer.tokenize(processed_obs["prompt"])
                processed_obs["tokenized_prompt"] = tokens.unsqueeze(0).repeat(observations["state"].size(0), 1)
                processed_obs["tokenized_prompt_mask"] = token_masks.unsqueeze(0).repeat(observations["state"].size(0), 1)

        return processed_obs

    def forward(self, observations=None):
        """
        Forward pass through the normalizer. Operation include a preprocessing operation and normalization in this order.

        :param observations: Observation dictionary
        :return: Tuple of normalized observations and actions
        """

        if not isinstance(observations, dict):
            observations = {"state": observations}

        observations = self.preprocess_observations(observations)
        if self.drop_images:
            observations = observations["state"]
        elif isinstance(observations["images"], dict):
            observations["images"] = torch.stack([v for v in observations["images"].values()], dim=-4)

        return observations

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Action processing on the output of the model. Operation follows in reverse the steps of processing step: un-normalize and post-processing operations back to environment format.
        """
        if self.discrete_actions:
            actions = -10 + (20.0 * actions / (self.n_tokens_per_dim - 1))

        actions = self.unnormalize_actions(actions)

        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        if self.squash_output:  # unscale the actions to match env bounds
            actions = self.action_space_low + (0.5 * (actions + 1.0) * (self.action_space_high - self.action_space_low))
        else:  # ow clip the actions to avoid out of bound error as we are sampling from an unbounded Gaussian distribution
            actions = torch.clip(actions, self.action_space_low, self.action_space_high)

        return actions


class Gym_2_Mlp(GymPreprocessor):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.proc_observation_space = spaces.Dict({
            "privileged": spaces.Box(low=-1, high=1, shape=(0,), dtype=np.float32),
            "state": spaces.Box(low=-10, high=10, shape=(36,), dtype=np.float32),
            # "images": spaces.Box(low=-1, high=1, shape=(2, 3, 94, 94), dtype=np.float32),
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
        assert isinstance(observations, dict), "Processing implemented only for `Dict` observations. Did you wrap correctly the datasource to obtain here the shared data source?"

        norm_obs = self.normalize_observations(self.preprocess_observations(observations))

        norm_obs["privileged"] = norm_obs["state"][..., norm_obs["state"].shape[-1]-min(self.proc_observation_space["privileged"].shape[-1], norm_obs["state"].shape[-1]) :]
        norm_obs["state"] = norm_obs["state"][..., : self.proc_observation_space["state"].shape[-1]]
        if "images" in norm_obs:
            norm_obs["images"] = torch.stack([norm_obs["images"][k] for k in norm_obs["images"]], dim=-4) if isinstance(norm_obs["images"], dict) else norm_obs["images"]

        return norm_obs

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        actions = super().forward_post(actions)

        return actions


# Aliases for other policy types that use the same preprocessing
Gym_2_Lstm = Gym_2_Mlp
Gym_2_Tr = Gym_2_Mlp
