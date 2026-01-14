from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from common.utils import extract_shapes_from_space


class Preprocessor(nn.Module):
    """
    Normalizer base class.

    Encoding input: Preprocess (standard + env-specific data transformations) -> Normalize -> (opt) superclass operations (policy-specific)
    Decoding output: (opt) superclass operations (policy-specific) -> De-normalize -> Postprocess
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, normalize_images: bool = False,
                 resize_images: Optional[tuple] = False, observation_modes: Optional[Dict[str, str]] = None,
                 action_modes: Optional[Dict[str, str]] = None, data_key_map: Optional[Dict[str, str]] = None,
                 stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, overwrite_stats: bool = False,
                 tokenizer=None, max_token_len: int = 48, prompt_generator=None,
                 task: Optional[str]=None, device: torch.device = torch.device('cpu')):
        """
        Initialize the normalizer.

        :param observation_space: Observation space of the environment
        :param action_space: Action space of the environment
        :param normalize_images: Whether to normalize images by dividing by 255 and 0-center i.e. [-1,1] range
        :param resize_images: Resolution to which resize the images
        :param observation_modes: Dict mapping observation keys to normalization modes
        :param action_modes: Dict mapping action keys to normalization modes
        :param stats: Optional dictionary with normalization statistics
        :param overwrite_stats: Whether to replace existing normalization statistics (torch parameters), ideally in a model resume operation
        :param tokenizer: tokenizer object class
        :param max_token_len: Maximum length of tokenized prompts
        :param prompt_generator: prompt generator object class
        :param device: torch device
        """
        super().__init__()

        self.observation_modes = observation_modes if observation_modes else {}
        self.action_modes = action_modes if action_modes else {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.data_key_map = data_key_map if data_key_map else {}
        self.overwrite_stats = overwrite_stats
        self.normalize_images = normalize_images
        self.resize_images = resize_images
        self.max_token_len = max_token_len
        self.tokenizer = eval(tokenizer)(max_token_len, device=device) if tokenizer is not None else None
        self.prompt_generator = eval(prompt_generator) if prompt_generator is not None else None
        self.task = task
        self.device = device

        # Extract shapes from spaces
        observation_shapes = extract_shapes_from_space(observation_space)
        action_shapes = extract_shapes_from_space(action_space)
        # Create buffers for normalization
        self._create_stats_buffers(observation_shapes, self.observation_modes, stats, prefix="obs_")
        self._create_stats_buffers(action_shapes, self.action_modes, stats, prefix="action_")

    def _create_stats_buffers(self, shapes: Dict[str, List[int]], modes: Dict[str, str],
                              stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None, prefix: str = "") -> None:
        """
        Create normalization buffers for observations or actions.

        :param shapes: Dictionary mapping keys to their shapes
        :param modes: Dictionary mapping keys to normalization modes
        :param stats: Optional dictionary with normalization statistics
        :param prefix: Prefix for buffer names
        """
        for key, mode in modes.items():

            shape_key = key if key in shapes else ""  # Handle non-dict spaces

            if shape_key not in shapes:
                print(f"`{shape_key}` can be found between normalization keys but has not been found in the current observation/action space!")
                continue

            shape = tuple(shapes[shape_key])

            if "images" in key:
                # Sanity checks for images
                assert len(shape) == 3, f"number of dimensions of {key} != 3 (shape={shape})"
                c, h, w = shape
                if c > h or c > w:
                    c = w  # probably not channel first... inconsistencies shouldn't be a problem since we apply preprocessing before normalization
                shape = (c, 1, 1)  # shape invariant to height and width

            assert key in self.data_key_map, f"Key `{key}` is a valid normalization key for the current observation/action space but cannot be mapped to a standard nomenclature. Add `{key}` mapping to `data_key_map` parameter."
            buffer_key = prefix + self.data_key_map[key].replace(".", "_")

            if mode == "mean_std":
                if stats is not None and key in stats:
                    mean = torch.from_numpy(stats[key]["mean"]) if isinstance(stats[key]["mean"], np.ndarray) else torch.tensor(stats[key]["mean"])
                    std = torch.from_numpy(stats[key]["std"]) if isinstance(stats[key]["std"], np.ndarray) else torch.tensor(stats[key]["std"])
                else:
                    mean = torch.zeros(shape, dtype=torch.float32, requires_grad=False)
                    std = torch.ones(shape, dtype=torch.float32, requires_grad=False)

                if not hasattr(self, f"{buffer_key}_mean") or self.overwrite_stats:
                    self.register_buffer(f"{buffer_key}_mean", mean.to(self.device).float())
                    self.register_buffer(f"{buffer_key}_std", std.to(self.device).float())

            elif mode == "min_max":
                if stats is not None and key in stats:
                    min_val = torch.from_numpy(stats[key]["min"]) if isinstance(stats[key]["min"], np.ndarray) else torch.tensor(stats[key]["min"])
                    max_val = torch.from_numpy(stats[key]["max"]) if isinstance(stats[key]["max"], np.ndarray) else torch.tensor(stats[key]["max"])
                else:
                    min_val = -torch.ones(shape, dtype=torch.float32, requires_grad=False)
                    max_val = torch.ones(shape, dtype=torch.float32, requires_grad=False)

                if not hasattr(self, f"{buffer_key}_min") or self.overwrite_stats:
                    self.register_buffer(f"{buffer_key}_min", min_val.to(self.device).float())
                    self.register_buffer(f"{buffer_key}_max", max_val.to(self.device).float())

    def _check_stats_initialized(self, name: str) -> None:
        """Check if stats are initialized and raise an error if they're not."""
        if torch.isinf(getattr(self, name)).any():
            raise ValueError(f"`{name}` is infinity. You should either initialize with `stats` as an argument, "
                             "or use a pretrained model.")

    def normalize_observations(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize observation batch according to the specified modes.

        :param observations: Dictionary of observation tensors
        :return: Dictionary of normalized observation tensors
        """
        # Handle tensor case
        if isinstance(observations, torch.Tensor):
            # for a single tensor, use default observation mode else skip normalization
            key = next(iter(self.observation_modes.keys())) if self.observation_modes else "default"
            return self.normalize_observations({key: observations})[key]

        # Deep copy the observations to avoid modifying the original
        observations = {k: v for k, v in observations.items()}

        # Apply statistical normalization based on input keys
        for key in observations.keys():
            # Find matching mode key
            mode_key = next((mk for mk in self.observation_modes.keys() if key in mk), None)
            if mode_key is None:
                continue

            mode = self.observation_modes[mode_key]
            buffer_key = "obs_" + self.data_key_map[mode_key].replace(".", "_")

            if mode == "mean_std":
                self._check_stats_initialized(f"{buffer_key}_mean")
                self._check_stats_initialized(f"{buffer_key}_std")
                mean = getattr(self, f"{buffer_key}_mean")
                std = getattr(self, f"{buffer_key}_std")
                if "images" in key and isinstance(observations[key], dict):
                    # Handle dictionary of images
                    for cam_name, img in observations[key].items():
                        observations[key][cam_name] = (img - mean) / (std + 1e-8)
                else:
                    observations[key] = (observations[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                self._check_stats_initialized(f"{buffer_key}_min")
                self._check_stats_initialized(f"{buffer_key}_max")
                min_val = getattr(self, f"{buffer_key}_min")
                max_val = getattr(self, f"{buffer_key}_max")
                if "images" in key and isinstance(observations[key], dict):
                    for cam_name, img in observations[key].items():
                        observations[key][cam_name] = (img - min_val) / (max_val - min_val + 1e-8) * 2 - 1  # [-1, 1]
                else:
                    observations[key] = (observations[key] - min_val) / (max_val - min_val + 1e-8) * 2 - 1  # [-1, 1]

        return observations

    def normalize_actions(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Normalize action batch according to the specified modes.

        :param actions: Dictionary of action tensors
        :return: Dictionary of normalized action tensors
        """
        # Handle tensor case
        if isinstance(actions, torch.Tensor):
            # for a single tensor, use default action mode else skip normalization
            key = next(iter(self.action_modes.keys())) if self.action_modes else "default"
            return self.normalize_actions({key: actions})[key]

        actions = {k: v for k, v in actions.items()}

        # Apply statistical normalization based on input keys
        for key in actions.keys():
            # Find matching mode key
            mode_key = next((mk for mk in self.action_modes.keys() if key in mk), None)
            if mode_key is None:
                continue

            mode = self.action_modes[mode_key]
            buffer_key = "action_" + self.data_key_map[mode_key].replace(".", "_")

            if mode == "mean_std":
                self._check_stats_initialized(f"{buffer_key}_mean")
                self._check_stats_initialized(f"{buffer_key}_std")
                mean = getattr(self, f"{buffer_key}_mean")
                std = getattr(self, f"{buffer_key}_std")
                actions[key] = (actions[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                self._check_stats_initialized(f"{buffer_key}_min")
                self._check_stats_initialized(f"{buffer_key}_max")
                min_val = getattr(self, f"{buffer_key}_min")
                max_val = getattr(self, f"{buffer_key}_max")
                actions[key] = (actions[key] - min_val) / (max_val - min_val + 1e-8)  # normalize to [0,1]
                actions[key] = actions[key] * 2 - 1  # normalize to [-1, 1]

        return actions

    def unnormalize_actions(self, actions: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[None, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Un-normalize actions.

        :param actions: Either a tensor or dictionary of normalized action tensors
        :return: Un-normalized actions in the same format as input (tensor or dictionary)
        """
        if actions is None:
            return None

        # Handle tensor case
        if isinstance(actions, torch.Tensor):
            # for a single tensor, use default action mode else skip normalization
            key = next(iter(self.action_modes.keys())) if self.action_modes else "default"
            return self.unnormalize_actions({key: actions})[key]

        actions_result = {k: v.clone() for k, v in actions.items()}

        # Apply statistical normalization based on input keys
        for key in actions.keys():
            # Find matching mode key
            mode_key = next((mk for mk in self.action_modes.keys() if key in mk), None)
            if mode_key is None:
                continue

            mode = self.action_modes[mode_key]
            buffer_key = "action_" + self.data_key_map[mode_key].replace(".", "_")

            if mode == "mean_std":
                self._check_stats_initialized(f"{buffer_key}_mean")
                self._check_stats_initialized(f"{buffer_key}_std")
                mean = getattr(self, f"{buffer_key}_mean")
                std = getattr(self, f"{buffer_key}_std")
                actions_result[key] = actions_result[key] * std + mean
            elif mode == "min_max":
                self._check_stats_initialized(f"{buffer_key}_min")
                self._check_stats_initialized(f"{buffer_key}_max")
                min_val = getattr(self, f"{buffer_key}_min")
                max_val = getattr(self, f"{buffer_key}_max")
                actions_result[key] = (actions_result[key] + 1) / 2  # [-1, 1] to [0, 1]
                actions_result[key] = actions_result[key] * (max_val - min_val) + min_val  # [0, 1] to original range

        return actions_result

    def forward(self):
        raise NotImplementedError

    def forward_post(self):
        raise NotImplementedError


class WrapperTokenizer:
    def __init__(self, processor, num_images):
        self.processor = processor
        self.num_images = num_images

    def prompt2chat(self, prompt):
        return {"prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt, "num_images": self.num_images}
        ]}

    def tokenize(self, prompt):
        # Use the EAGLE processor's tokenizer
        return self.processor.process_chat(self.prompt2chat(prompt))

    def tokenize_batch(self, prompts):
        # Process a batch of prompts
        return self.processor.process_batch([self.prompt2chat(prompt) for prompt in prompts])
