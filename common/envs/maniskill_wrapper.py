"""
Wrapper to configure an environment instance to Stable-Baselines3 vectorized environment to a standard format (IsaacLab)
before being converted to a standard that is common to envs and datasets by Sb3VecEnvWrapper.

"""

from __future__ import annotations

import copy
import math
from typing import Any

import gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from common import utils


class ManiSkillEnvStdWrapper(gym.Wrapper):
    """
    A wrapper that standardizes gym environments for use with Sb3VecEnvWrapper.

    This wrapper:
    1. Sets up single_observation_space with 'policy' key for state observations
    2. Ensures all observations, rewards, and info are returned as batched tensors
    3. Handles both dict and non-dict observation spaces
    4. Standardizes the interface to remove conditionals in Sb3VecEnvWrapper
    """

    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.

        Args:
            env: The base gymnasium environment
            num_envs: Number of parallel environments (for batching)
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        super().__init__(env)
        self.num_envs = env.num_envs
        self.sim_device = getattr(self.unwrapped, "device", "cpu")

        # Setup spaces
        if isinstance(self.env.observation_space, spaces.Dict):
            self.single_observation_space = spaces.Dict({
                "policy": spaces.Box(-math.inf, math.inf, self.env.single_observation_space["state"].shape),
                **{cam_name: cam["rgb"] for cam_name, cam in self.env.single_observation_space["sensor_data"].items()}
            })
        else:
            self.single_observation_space = spaces.Dict({"policy": self.env.single_observation_space})
        self.single_action_space = self.env.single_action_space

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Reset the environment and return batched observations.

        Returns:
            obs_dict: Dictionary of batched tensor observations
            info: Dictionary of batched info
        """
        obs, info = self.env.reset(**kwargs)

        # Convert observations to standardized dict format
        obs = self._process_observation(obs)

        return obs, info

    def step(self, action) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Step the environment and return batched results.

        Args:
            action: Action tensor (batched or single)

        Returns:
            obs_dict: Dictionary of batched tensor observations
            reward: Batched reward tensor
            terminated: Batched terminated tensor
            truncated: Batched truncated tensor
            info: Dictionary of batched info
        """
        # Unbatch action if needed
        if ((isinstance(action, torch.Tensor) and action.dim() > 0) or (isinstance(action, np.ndarray) and action.ndim > 0)) and self.num_envs == 1:
            action = action.squeeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.render()

        reward = self._batch_scalar(reward)
        terminated = self._batch_scalar(terminated).bool()
        truncated = self._batch_scalar(truncated).bool()

        if torch.all(terminated | truncated):
            obs, _ = self.reset(**{"seed": None})
        else:
            # Restart necessary environments and update observations
            env_idx = (terminated | truncated).nonzero(as_tuple=True)[0]
            reset_obs = None
            if env_idx.numel() > 0:
                reset_obs, _ = self.reset(options={"env_idx": env_idx}, **{"seed": None})

            processed_obs = self._process_observation(obs)
            if reset_obs is not None:
                for key in processed_obs.keys():
                    processed_obs[key][env_idx] = reset_obs[key][env_idx]
            obs = processed_obs

        return obs, reward, terminated, truncated, info

    def _process_observation(self, obs_dict) -> dict[str, torch.Tensor | np.ndarray]:
        """Convert observation to standardized dict format with 'policy' key."""
        # new_obs = self.env.base_env.scene.get_sim_state()
        # 'policy': torch.cat([ new_obs['articulations']['panda_wristcam'][...,-18:], torch.tensor([[0.]*9]), obs_dict['state']], -1).detach().to(self.device),
        obs_dict = obs_dict if isinstance(obs_dict, dict) else {"state": obs_dict}
        return {
            "policy": obs_dict["state"].detach(),
            **{k: v["rgb"].detach() for k, v in obs_dict.get("sensor_data", {}).items()},
        }

    def _batch_observations(self, obs_dict: dict[str, torch.Tensor | np.ndarray]) -> dict[str, torch.Tensor]:
        """Batch observations to match expected format."""
        batched_obs = {}

        for key, value in obs_dict.items():
            if isinstance(value, torch.Tensor):
                # Ensure tensor is batched
                if value.dim() == 0:  # scalar
                    batched_value = value.unsqueeze(0).expand(self.num_envs)
                elif value.dim() == 1 and value.shape[0] != self.num_envs:
                    # Add batch dimension
                    batched_value = value.unsqueeze(0).expand(self.num_envs, -1)
                elif value.shape[0] != self.num_envs:
                    # Add batch dimension
                    batched_value = value.unsqueeze(0).expand(self.num_envs, *value.shape)
                else:
                    batched_value = value
            else:
                # Convert numpy/other to tensor and batch
                tensor_value = torch.from_numpy(np.asarray(value))
                if tensor_value.dim() == 0:  # scalar
                    batched_value = tensor_value.unsqueeze(0).expand(self.num_envs)
                elif tensor_value.shape[0] != self.num_envs:
                    # Add batch dimension
                    batched_value = tensor_value.unsqueeze(0).expand(self.num_envs, *tensor_value.shape)
                else:
                    batched_value = tensor_value

            batched_obs[key] = batched_value

        return batched_obs

    def _batch_scalar(self, value: float | int | torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert scalar value to batched tensor."""
        if isinstance(value, torch.Tensor):
            if ((value.dim() == 0) or (value.shape[0] != self.num_envs)):
                return value.unsqueeze(0).expand(self.num_envs)
            return value
        return torch.full((self.num_envs,), float(value), device=self.sim_device)

    def seed(self, rnd: int) -> None:
        """Return the base environment."""
        pass

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped


class FlattenActionSpaceWrapper(gym.ActionWrapper):
    """
    Flattens the action space. The original action space must be spaces.Dict
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._orig_single_action_space = copy.deepcopy(self.base_env.single_action_space)
        self.single_action_space = gymnasium.spaces.utils.flatten_space(self.base_env.single_action_space)
        if self.base_env.num_envs > 1:
            self.action_space = batch_space(self.single_action_space, n=self.base_env.num_envs)
        else:
            self.action_space = self.single_action_space

    @property
    def base_env(self):
        return self.env.unwrapped

    def action(self, action):
        if self.base_env.num_envs == 1 and action.shape == self.single_action_space.shape:
            action = utils.batch(action)
        if isinstance(self._orig_single_action_space, gym.spaces.Box):
            unflattened_action = action
        else:
            unflattened_action = dict()
            start, end = 0, 0
            for k, space in self._orig_single_action_space.items():
                end += space.shape[0]
                unflattened_action[k] = action[:, start:end]
                start += space.shape[0]

        return unflattened_action
