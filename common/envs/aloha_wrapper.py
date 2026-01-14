"""
Wrapper to configure Aloha environment to standard format compatible with Sb3EnvStdWrapper.

This wrapper transforms Aloha's observation structure to match the expected format:
- Converts 'agent_pos' to 'policy' key
- Flattens nested 'pixels' dict to top-level camera keys
- Ensures observations are torch tensors
- Handles batching and auto-reset logic
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


class AlohaStdWrapper(gym.Wrapper):
    """
    A wrapper that standardizes Aloha gym environment for use with Sb3EnvStdWrapper.

    This wrapper:
    1. Converts observation space from Aloha format to standard format
       - 'agent_pos' -> 'policy'
       - 'pixels'/'top' -> 'top' (flattened camera keys)
    2. Ensures all observations are returned as batched tensors
    3. Handles auto-reset on termination/truncation
    4. Standardizes the interface to remove conditionals in Sb3EnvStdWrapper

    Aloha observation space structure:
        Dict('agent_pos': Box(...), 'pixels': Dict('top': Box(...)))

    Output observation space structure:
        Dict('policy': Box(...), 'top': Box(...))
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.

        :param env: The base Aloha gymnasium environment
        """
        super().__init__(env)
        self.num_envs = 1  # Aloha is not vectorized
        self.sim_device = "cpu"  # Aloha runs on CPU

        # Transform observation space from Aloha format to standard format
        # Original: Dict('agent_pos': Box(...), 'pixels': Dict('top': Box(...)))
        # Target: Dict('policy': Box(...), 'top': Box(...))
        original_obs_space = self.env.observation_space

        if not isinstance(original_obs_space, spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(original_obs_space)}")

        # Extract agent_pos as policy
        if "agent_pos" not in original_obs_space.spaces:
            raise ValueError("Expected 'agent_pos' in observation space")

        agent_pos_space = original_obs_space["agent_pos"]

        # Build standardized observation space
        new_obs_space = {"policy": spaces.Box(
            low=-math.inf,
            high=math.inf,
            shape=agent_pos_space.shape,
            dtype=np.float32
        )}

        # Extract camera observations from nested pixels dict
        if "pixels" in original_obs_space.spaces:
            pixels_space = original_obs_space["pixels"]
            if isinstance(pixels_space, spaces.Dict):
                # Flatten nested camera structure
                for cam_name, cam_space in pixels_space.spaces.items():
                    new_obs_space[cam_name] = cam_space
            else:
                # If pixels is not a dict, use it directly
                new_obs_space["pixels"] = pixels_space

        self.single_observation_space = spaces.Dict(new_obs_space)
        self.single_action_space = self.env.action_space

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Reset the environment and return batched observations.

        :return: Tuple of (obs_dict, info) where obs_dict contains batched tensor observations
        """
        obs, info = self.env.reset(**kwargs)

        # Convert observations to standardized dict format
        obs = self._process_observation(obs)

        return obs, info

    def step(self, action) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Step the environment and return batched results.

        :param action: Action to execute (can be batched or single)
        :return: Tuple of (obs_dict, reward, terminated, truncated, info)
        """
        # Unbatch action if needed (Aloha expects single action)
        if isinstance(action, torch.Tensor):
            if action.dim() > 1:
                action = action.squeeze(0)
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert to tensors and batch
        reward = self._batch_scalar(reward)
        terminated = self._batch_scalar(terminated).bool()
        truncated = self._batch_scalar(truncated).bool()

        # Handle auto-reset
        if terminated.item() or truncated.item():
            obs, _ = self.reset(**{"seed": None})
        else:
            obs = self._process_observation(obs)

        return obs, reward, terminated, truncated, info

    def _process_observation(self, obs) -> dict[str, torch.Tensor]:
        """
        Convert observation from Aloha format to standardized dict format.

        Transforms:
            {'agent_pos': array, 'pixels': {'top': array}}
        to:
            {'policy': tensor, 'top': tensor}

        :param obs: Raw observation from environment
        :return: Processed observation dictionary with torch tensors
        """
        if not isinstance(obs, dict):
            raise ValueError(f"Expected dict observation, got {type(obs)}")

        processed_obs = {}

        # Convert agent_pos to policy
        if "agent_pos" in obs:
            agent_pos = obs["agent_pos"]
            if isinstance(agent_pos, np.ndarray):
                agent_pos = torch.from_numpy(agent_pos).float()
            elif not isinstance(agent_pos, torch.Tensor):
                agent_pos = torch.tensor(agent_pos, dtype=torch.float32)

            # Ensure batched shape (1, ...)
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)

            processed_obs["policy"] = agent_pos.to(self.sim_device)

        # Extract camera observations from nested pixels dict
        if "pixels" in obs:
            pixels = obs["pixels"]
            if isinstance(pixels, dict):
                # Flatten nested camera structure
                for cam_name, cam_data in pixels.items():
                    if isinstance(cam_data, np.ndarray):
                        cam_tensor = torch.from_numpy(cam_data)
                    elif not isinstance(cam_data, torch.Tensor):
                        cam_tensor = torch.tensor(cam_data)
                    else:
                        cam_tensor = cam_data

                    # Ensure batched shape (1, H, W, C) or (1, C, H, W)
                    if cam_tensor.dim() == 3:
                        cam_tensor = cam_tensor.unsqueeze(0)

                    processed_obs[cam_name] = cam_tensor.to(self.sim_device)
            else:
                # If pixels is not a dict, convert directly
                if isinstance(pixels, np.ndarray):
                    pixels = torch.from_numpy(pixels)
                elif not isinstance(pixels, torch.Tensor):
                    pixels = torch.tensor(pixels)

                if pixels.dim() == 3:
                    pixels = pixels.unsqueeze(0)

                processed_obs["pixels"] = pixels.to(self.sim_device)

        return processed_obs

    def _batch_scalar(self, value: float | int | bool | torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Convert scalar value to batched tensor.

        :param value: Scalar value to batch
        :return: Batched tensor with shape (num_envs,)
        """
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.unsqueeze(0).to(self.sim_device)
            return value.to(self.sim_device)

        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
            if value.dim() == 0:
                return value.unsqueeze(0).to(self.sim_device)
            return value.to(self.sim_device)

        # Scalar value (float, int, bool)
        return torch.tensor([float(value)], device=self.sim_device)

    def seed(self, seed: int | None = None) -> None:
        """
        Seed the environment.

        Note: Aloha environments handle seeding through reset() kwargs,
        so this method doesn't need to do anything.

        :param seed: Random seed (unused, handled via reset)
        """
        # Aloha environments don't have a separate seed() method
        # Seeding is handled through reset(seed=seed)
        pass

    @property
    def unwrapped(self):
        """
        Return the base environment.

        :return: The unwrapped base environment
        """
        return self.env.unwrapped
