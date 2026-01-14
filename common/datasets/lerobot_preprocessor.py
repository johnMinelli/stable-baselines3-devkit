from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from common.preprocessor import Preprocessor


class LerobotPreprocessor(Preprocessor):
    """
    Normalizer for Aloha dataset features that handles both statistical normalization
    and Aloha-specific data transformations.

    Encoding input: Preprocess (standard + dataset-specific data transformations) -> Normalize -> (opt) superclass operations (policy-specific)
    Decoding output: (opt) superclass operations (policy-specific) -> De-normalize -> Postprocess
    """

    def __init__(self, use_delta_joint_actions: bool = False, **kwargs):
        """
        Initialize the normalizer.

        Args:
            use_delta_joint_actions: Whether to use delta joint actions
            adapt_to_pi: Whether to adapt data to pi space
            kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        self.use_delta_joint_actions = use_delta_joint_actions

        # The expected cameras names
        self.expected_image_keys = [
            "base_camera",
            "hand_camera",
            "head_camera",
            "image",
            "wrist_image",
            "top",
            "image_side_1",
            "image_side_2",
        ]

    def _decode_state(self, state: torch.Tensor) -> torch.Tensor:
        return state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.float32)

    def _decode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.float32)

    def preprocess_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Aloha-specific preprocessing to raw observations from dataset.

        Args:
            observations: Dictionary with observation data

        Returns:
            Preprocessed observations dictionary
        """

        # Process observations
        processed_obs = {}
        for key in observations.keys():
            _obs = observations[key]
            if key == "state":
                processed_obs["state"] = self._decode_state(_obs).to(self.device)
            elif "images" in key:
                processed_images = {}
                for cam_name, img in _obs.items():
                    if cam_name in self.expected_image_keys:
                        # cconvert to torch tensor if needed
                        if not isinstance(img, torch.Tensor):
                            img = torch.tensor(img, device=self.device)
                        img = img.to(self.device)
                        # ensure channel-first format
                        if img.shape[-3] not in [1, 3, 4] and img.shape[-1] in [1, 3, 4]:
                            permutation_indices = list(range(img.dim()))
                            permutation_indices.append(permutation_indices.pop(-3))
                            permutation_indices.append(permutation_indices.pop(-3))
                            img = img.permute(*permutation_indices)
                        if self.normalize_images:
                            if img.max() > 1:
                                img = img.float() / 255.0
                            img = img.float() * 2.0 - 1.0
                        if self.resize_images:
                            img = F.interpolate(
                                img.view(-1, *img.shape[-3:]),
                                size=self.resize_images,
                                mode="bilinear",
                                align_corners=False,
                            ).view(*img.shape[:-2], *self.resize_images)
                        processed_images[cam_name] = img.to(self.device)
                processed_obs[key] = processed_images
            elif key in self.expected_image_keys:
                _obs = _obs.to(self.device)
                if _obs.shape[-3] not in [1, 3, 4] and _obs.shape[-1] not in [1, 3, 4]:
                    _obs = _obs.permute(0, 3, 1, 2)
                if self.normalize_images:
                    if _obs.max() > 1:
                        _obs = _obs.float() / 255.0
                    _obs = _obs.float() * 2.0 - 1.0
                if self.resize_images:
                    _obs = F.interpolate(
                        _obs.view(-1, *_obs.shape[-3:]), size=self.resize_images, mode="bilinear", align_corners=False
                    ).view(*_obs.shape[:-2], *self.resize_images)
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
                processed_obs["tokenized_prompt_mask"] = token_masks.unsqueeze(0).repeat(
                    observations["state"].size(0), 1
                )

        return processed_obs

    def preprocess_actions(
        self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply Aloha-specific preprocessing to raw actions from dataset.

        Args:
            actions: Action tensor of shape [horizon, dim] or [batch, horizon, dim] or dictionary of action tensors

        Returns:
            Preprocessed action
        """
        # Handle dictionary case
        if isinstance(actions, dict):
            return {k: self.preprocess_actions(v) for k, v in actions.items()}

        # Handle tensor case
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device)
        # Apply delta transformation if needed
        if self.use_delta_joint_actions:
            if actions.dim() == 2:  # [horizon, dim]
                delta_mask = torch.ones_like(actions[0])
                delta_mask[[6, 13]] = 0  # Keep gripper dimensions absolute
                actions[1:] = actions[1:] - actions[:-1] * delta_mask
            elif actions.dim() == 3:  # [batch, horizon, dim]
                delta_mask = torch.ones_like(actions[0, 0])
                delta_mask[[6, 13]] = 0  # Keep gripper dimensions absolute
                actions[:, 1:] = actions[:, 1:] - actions[:, :-1] * delta_mask

        # Apply Aloha-specific transformations
        return self._decode_actions(actions)

    def postprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse of Aloha-specific preprocessing to model outputs.

        Args:
            actions: Normalized action tensor

        Returns:
            Denormalized action tensor in original Aloha format
        """
        # Apply Aloha-specific format encoding
        actions = self._encode_actions(actions)

        # Convert deltas back to absolute positions if needed
        if self.use_delta_joint_actions:
            # Create mask for determining which dimensions to accumulate
            delta_mask = torch.ones_like(actions[0, 0] if actions.dim() == 3 else actions[0])
            delta_mask[[6, 13]] = 0  # Gripper dimensions were kept absolute

            # Apply accumulation to convert deltas back to absolute positions
            if actions.dim() == 2:  # [horizon, dim]
                absolute_actions = actions.clone()
                for t in range(1, actions.shape[0]):
                    absolute_actions[t] = absolute_actions[t] + absolute_actions[t - 1] * delta_mask
            elif actions.dim() == 3:  # [batch, horizon, dim]
                absolute_actions = actions.clone()
                for t in range(1, actions.shape[1]):
                    absolute_actions[:, t] = absolute_actions[:, t] + absolute_actions[:, t - 1] * delta_mask

        return actions

    def forward(self, observations=None, actions=None):
        """
        Forward pass through the normalizer. Operation include a preprocessing operation and normalization in this order.

        Args:
            observations: Observation dictionary
            actions: Action dictionary

        Returns:
            Tuple of normalized observations and actions
        """
        normalized_results = []

        if observations is not None:
            norm_obs = self.normalize_observations(self.preprocess_observations(observations))
            norm_obs["images"] = torch.stack([v for v in observations["images"].values()], dim=-4)
            normalized_results.append(norm_obs)
        if actions is not None:
            normalized_results.append(self.normalize_actions(self.preprocess_actions(actions)))

        return normalized_results[0] if len(normalized_results) == 1 else tuple(normalized_results)

    def forward_post(
        self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Action processing on the output of the model. Operation follows in reverse the steps of processing step: un-normalize batch according to the specified modes and post-processing operations back to dataset format.
        """
        actions = self.unnormalize_actions(actions)

        # Apply Aloha-specific postprocessing on the tensors
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"
        actions = self.postprocess_actions(actions)

        return actions


class Lerobot_2_Mlp(LerobotPreprocessor):

    def __init__(self, tokenizer_kwargs=None, mask_padding: bool = True, padding_dim: int = 32, **kwargs):
        self.tokenizer_kwargs = tokenizer_kwargs
        self.mask_padding = mask_padding
        self.padding_dim = padding_dim

        super().__init__(**kwargs)

        # Hard-imposed spaces specific for the dataset stored data and for the algorithm expectations
        obs = self.observation_space["observation"] if "observation" in self.observation_space.spaces else self.observation_space
        images = {im_key: im for im_key, im in (obs["images"] if "images" in obs.spaces else obs).items() if im_key in self.expected_image_keys}
        if len(images):
            imm_shape = (len(images), 3, *(list(images.values())[0].shape[:2] if not self.resize_images else self.resize_images))
            imm_box = spaces.Box(low=-1, high=1, shape=imm_shape, dtype=np.float32)
        else:
            imm_box = spaces.Box(low=-1, high=1, shape=(0, 0, 0, 0), dtype=np.float32)

        self.proc_observation_space = spaces.Dict({
            "privileged": spaces.Box(low=-1, high=1, shape=obs["privileged"].shape, dtype=np.float32),
            "state": spaces.Box(low=-10, high=10, shape=obs["state"].shape, dtype=np.float32),
            "images": imm_box,
        })
        self.proc_action_space = spaces.Box(low=-10, high=10, shape=self.action_space.shape, dtype=np.float32)

    def forward(self, observations=None, actions=None):
        """
        Forward pass through the normalizer. Operation include a preprocessing operation and normalization in this order.

        Args:
            observations: Observation dictionary
            actions: Action dictionary

        Returns:
            Tuple of normalized observations and actions
        """
        normalized_results = []

        if observations is not None:
            if "prompt" in observations:
                del observations["prompt"]
            norm_obs = self.normalize_observations(self.preprocess_observations(observations))

            norm_obs["privileged"] = norm_obs["state"][..., -self.proc_observation_space["privileged"].shape[-1] :]
            norm_obs["state"] = norm_obs["state"][..., : self.proc_observation_space["state"].shape[-1]]
            if "images" in norm_obs:
                norm_obs["images"] = torch.stack([norm_obs["images"][k] for k in norm_obs["images"] if k in self.expected_image_keys], dim=-4)
            else:
                norm_obs["images"] = torch.stack([norm_obs[k] for k in norm_obs if k in self.expected_image_keys], dim=-4)
                for k in list(norm_obs.keys()):
                    if k in self.expected_image_keys: del norm_obs[k]

            normalized_results.append(norm_obs)

        if actions is not None:
            norm_act = self.normalize_actions(self.preprocess_actions(actions))

            normalized_results.append(norm_act)

        return normalized_results[0] if len(normalized_results) == 1 else tuple(normalized_results)

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        # Truncate output
        actions = actions[..., : self.action_space.shape[-1]]
        actions = super().forward_post(actions)

        return actions


# Aliases for other policy types that use the same preprocessing
Lerobot_2_Tcn = Lerobot_2_Mlp
Lerobot_2_Lstm = Lerobot_2_Mlp


"""
Specific preprocessor of Y datasource for X policy is implemented by extending the YPreprocessor class:

e.g.
`
class Y_2_X(YPreprocessor):
    def __init__(self, tokenizer_kwargs=None, **kwargs):
        self.tokenizer_kwargs = tokenizer_kwargs
        super().__init__(**kwargs)

        # Policy Specific spaces relative to the dataset stored (if policy expect specific format/data you redefine the processed space and apply pre-/post-processing)
        obs = self.observation_space["observation"] if "observation" in self.observation_space.spaces else self.observation_space
        images = {im_key: im for im_key, im in (obs["images"] if "images" in obs.spaces else obs).items() if im_key in self.expected_image_keys}
        imm_shape = (len(images), 3, *(list(images.values())[0].shape[:2] if not self.resize_images else self.resize_images))
        self.proc_observation_space = spaces.Dict({
            "privileged": spaces.Box(low=-1, high=1, shape=obs["privileged"].shape, dtype=np.float32),
            "state": spaces.Box(low=-10, high=10, shape=obs["state"].shape, dtype=np.float32),
            "images": spaces.Box(low=-1, high=1, shape=imm_shape, dtype=np.float32), })
        self.proc_action_space = spaces.Box(low=-10, high=10, shape=self.action_space.shape, dtype=np.float32)

        # Prepare tokenizer and wrapper for prompts
        from common.preprocessor import WrapperTokenizer
        self.custom_tokenizer = YourCustomTokenizer(device=self.device, **tokenizer_kwargs)
        self.tokenizer = WrapperTokenizer(self.custom_tokenizer)


    def forward(self, observations=None, actions=None):
        \"""
        Forward pass through the normalizer. Operation include a preprocessing operation and normalization in this order.

        Args:
            observations: Observation dictionary
            actions: Action dictionary

        Returns:
            Tuple of normalized observations and actions
        \"""
        normalized_results = []

        # Example of processing to extract necessary data for the policy
        if observations is not None:
            if "prompt" in observations: del observations["prompt"]
            norm_obs = self.normalize_observations(self.preprocess_observations(observations))

            norm_obs["privileged"] = norm_obs["state"][..., -self.proc_observation_space["privileged"].shape[-1]:]
            if self.task == "TwoRobotStackCube-v1":
                norm_obs["state"] = torch.cat([norm_obs["state"][..., :9], norm_obs["state"][..., 18:18 + 9]], -1)
            else:
                norm_obs["state"] = norm_obs["state"][..., :self.proc_observation_space["state"].shape[-1]]
            if "images" in norm_obs:
                norm_obs["images"] = torch.stack([norm_obs["images"][k] for k in norm_obs["images"] if k in self.expected_image_keys], dim=-4)
            else:
                norm_obs["images"] = norm_obs["image"].unsqueeze(-4)
                del norm_obs["image"]

            normalized_results.append(norm_obs)

        if actions is not None:
            norm_act = self.normalize_actions(self.preprocess_actions(actions))

            normalized_results.append(norm_act)

        return normalized_results[0] if len(normalized_results) == 1 else tuple(normalized_results)

    def forward_post(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        assert isinstance(self.action_space, spaces.Box), "Processing implemented only for `Box` action space"

        # Truncate output
        actions = actions[..., :self.action_space.shape[-1]]
        actions = super().forward_post(actions)
`
"""
