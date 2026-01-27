"""
Wrapper to convert MuJoCo Playground's State-based JAX API to Gymnasium-compatible interface.

MuJoCo Playground uses JAX/MJX for GPU-accelerated physics simulation. This wrapper:
- Loads environments via registry.load() instead of gym.make()
- Converts State-based API to tuple-based Gymnasium API
- Converts JAX arrays to PyTorch tensors (via DLPack for zero-copy GPU transfer)

Reference: https://github.com/google-deepmind/mujoco_playground
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
import torch
from gymnasium import spaces
from mujoco_playground import registry


def jax_to_torch_gpu(x: jax.Array) -> torch.Tensor:
    """Zero-copy JAX to PyTorch conversion on GPU via DLPack."""
    return torch.from_dlpack(x)


def jax_to_torch_cpu(x: jax.Array) -> torch.Tensor:
    """JAX to PyTorch conversion on CPU via numpy (no copy if contiguous)."""
    return torch.from_numpy(np.asarray(x))


def torch_to_jax_gpu(x: torch.Tensor) -> jax.Array:
    """Zero-copy PyTorch to JAX conversion on GPU via DLPack."""
    return jax.dlpack.from_dlpack(x.detach())


def torch_to_jax_cpu(x: torch.Tensor) -> jax.Array:
    """PyTorch to JAX conversion on CPU via numpy."""
    return jnp.asarray(x.detach().numpy())


class MjxPlaygroundGymWrapper(gym.Env):
    """
    Gymnasium-compatible wrapper for MuJoCo Playground environments.

    Note: differently from Gym environments, MuJoCo Playground State dataclass contains the following data that should be converted:
     - data: mjx.Data (physics state)
     - obs: Union[jax.Array, Dict[str, jax.Array]] (observations)
     - reward: jax.Array (scalar or batched reward)
     - done: jax.Array (termination flag)
     - metrics: Dict[str, jax.Array]
     - info: Dict[str, Any]
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cuda",
        config_overrides: dict[str, Any] | None = None,
        max_episode_steps: int | None = 1000,
        render_mode: str | None = None,
        camera_resolution: tuple[int, int] = (640, 480),
    ):
        """
        Initialize the MuJoCo Playground wrapper.

        :param env_name: Name of the environment (e.g., "CartpoleBalance", "CheetahRun")
        :param num_envs: Number of parallel environments (batched via vmap)
        :param seed: Random seed for JAX PRNG
        :param device: PyTorch device for output tensors ("cuda" or "cpu")
        :param config_overrides: Optional config overrides for the environment
        :param max_episode_steps: Maximum steps before truncation
        :param render_mode: Render mode ("rgb_array", "human", or None)
        :param camera_resolution: Resolution for camera rendering (width, height)
        """
        super().__init__()

        self.env_name = env_name
        self.num_envs = num_envs
        self.seed_value = seed
        self.sim_device = device
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.camera_resolution = camera_resolution

        # Cameras enabled if render_mode is set
        self.render_cameras = render_mode in ("rgb_array", "human")
        self.is_vector_env = True

        # Load environment config and create environment
        self.config = registry.get_default_config(env_name)
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.env = registry.load(env_name, config=self.config)

        # Store dt for rendering fps calculation
        self.dt = self.env.dt if hasattr(self.env, "dt") else 0.02

        # Initialize JAX PRNG key
        self._rng = jax.random.PRNGKey(seed)

        # Store current state (needed for State-based API)
        self._state = None
        self._step_count = None  # Track steps for truncation

        # Setup renderer for visualization (CPU-based MuJoCo renderer)
        # Must be done before _setup_spaces since spaces depend on camera resolution
        self._renderer = None
        self._viewer = None
        self._mj_model = None
        self._mj_data = None
        if self.render_cameras:
            self._setup_renderer()

        # Setup observation and action spaces
        self._setup_spaces()

        # Set device-specific conversion functions (avoid branching in hot path)
        self._is_gpu = "cuda" in device
        if self._is_gpu:
            self._jax_to_torch = jax_to_torch_gpu
            self._torch_to_jax = torch_to_jax_gpu
        else:
            self._jax_to_torch = jax_to_torch_cpu
            self._torch_to_jax = torch_to_jax_cpu

        # JIT compile step and reset for performance
        if num_envs > 1:
            self._jit_reset = jax.jit(jax.vmap(self.env.reset))
            self._jit_step = jax.jit(jax.vmap(self.env.step))
        else:
            self._jit_reset = jax.jit(self.env.reset)
            self._jit_step = jax.jit(self.env.step)

    def _setup_spaces(self):
        """Setup observation and action spaces based on environment."""
        # Get observation and action sizes from MuJoCo Playground env
        obs_size = self.env.observation_size
        action_size = self.env.action_size

        # Create observation space
        if isinstance(obs_size, int):
            state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
            obs_spaces = {"state": state_space}
        elif isinstance(obs_size, dict):
            # Handle dict observations
            obs_spaces = {k: spaces.Box(low=-np.inf, high=np.inf, shape=(v,), dtype=np.float32) for k, v in obs_size.items()}
        else:
            # Try to infer from a sample reset
            self._rng, reset_rng = jax.random.split(self._rng)
            sample_state = self.env.reset(reset_rng)
            sample_obs = np.asarray(sample_state.obs)
            state_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32)
            obs_spaces = {"state": state_space}

        # Add camera space if render_cameras is enabled
        if self.render_cameras:
            width, height = self.camera_resolution
            obs_spaces["camera"] = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

        self.single_observation_space = spaces.Dict(obs_spaces)

        # Action space - typically bounded [-1, 1] for control
        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)

        # Batched spaces for vectorized envs (used by Sb3EnvStdWrapper)
        if self.num_envs > 1:
            batched_spaces = {}
            for key, space in self.single_observation_space.spaces.items():
                batched_shape = (self.num_envs,) + space.shape
                batched_spaces[key] = spaces.Box(
                    low=space.low.flat[0], high=space.high.flat[0], shape=batched_shape, dtype=space.dtype
                )
            self.observation_space = spaces.Dict(batched_spaces)
            batched_action_shape = (self.num_envs,) + self.single_action_space.shape
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=batched_action_shape, dtype=np.float32)
        else:
            self.observation_space = self.single_observation_space
            self.action_space = self.single_action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset the environment.

        :param seed: Optional new seed for PRNG
        :param options: Optional reset options (e.g., env_idx for partial reset)
        :return: Tuple of (observations, info)
        """
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)

        # Handle partial reset for specific environment indices
        env_idx = options.get("env_idx") if options else None

        if env_idx is not None and self._state is not None and self.num_envs > 1:
            # Partial reset - only reset specified environments (vectorized)
            # Convert indices to JAX array (use DLPack on GPU, numpy on CPU)
            if isinstance(env_idx, torch.Tensor):
                env_idx_jax = self._torch_to_jax(env_idx.int())
            else:
                env_idx_jax = jnp.array(env_idx)
            n_resets = env_idx_jax.shape[0]

            reset_rngs = jax.random.split(self._rng, n_resets + 1)
            self._rng = reset_rngs[0]

            # Vectorized reset for all indices at once
            new_states = jax.vmap(self.env.reset)(reset_rngs[1:])

            # Vectorized state update using scatter
            self._state = self._update_state_at_indices(self._state, new_states, env_idx_jax)
            self._step_count[env_idx] = 0
        else:
            # Full reset
            if self.num_envs > 1:
                reset_rngs = jax.random.split(self._rng, self.num_envs + 1)
                self._rng = reset_rngs[0]
                self._state = self._jit_reset(reset_rngs[1:])
            else:
                self._rng, reset_rng = jax.random.split(self._rng)
                self._state = self._jit_reset(reset_rng)

            self._step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)

        # Convert observations to PyTorch
        obs = self._convert_observation(self._state.obs)
        info = self._convert_info(self._state.info) if hasattr(self._state, "info") and self._state.info else {}

        return obs, info

    def step(self, action: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Step the environment.

        :param action: Action tensor (batched if num_envs > 1)
        :return: Tuple of (obs, reward, terminated, truncated, info)
        """
        # Convert action to JAX (stay on same device)
        if isinstance(action, torch.Tensor):
            action_jax = self._torch_to_jax(action)
        else:
            action_jax = jnp.asarray(action)

        # Ensure correct shape for single env
        if self.num_envs == 1 and action_jax.ndim == 2:
            action_jax = action_jax.squeeze(0)

        # Step the environment
        self._state = self._jit_step(self._state, action_jax)
        self._step_count += 1

        # Convert outputs to PyTorch (stay on same device)
        obs = self._convert_observation(self._state.obs)
        reward = self._jax_to_torch(self._state.reward)
        terminated = self._jax_to_torch(self._state.done).bool()

        # Handle truncation (max episode steps)
        truncated = (self._step_count >= self.max_episode_steps) & ~terminated

        # Ensure batched shape for single env
        if self.num_envs == 1:
            if reward.ndim == 0:
                reward = reward.unsqueeze(0)
            if terminated.ndim == 0:
                terminated = terminated.unsqueeze(0)
            if truncated.ndim == 0:
                truncated = truncated.unsqueeze(0)

        # Build info dict
        info = {}
        if hasattr(self._state, "info") and self._state.info:
            info = self._convert_info(self._state.info)
        if hasattr(self._state, "metrics") and self._state.metrics:
            info["metrics"] = {k: self._jax_to_torch(v) for k, v in self._state.metrics.items()}

        return obs, reward, terminated, truncated, info

    def _convert_observation(self, obs: jax.Array | dict[str, jax.Array]) -> torch.Tensor:
        """Convert JAX observations to PyTorch tensors (stays on same device)."""
        if isinstance(obs, dict):
            return {k: self._jax_to_torch(v) for k, v in obs.items()}

        obs_torch = self._jax_to_torch(obs)

        # Ensure batched shape (num_envs, obs_dim)
        if self.num_envs == 1 and obs_torch.ndim == 1:
            obs_torch = obs_torch.unsqueeze(0)

        return obs_torch

    def _convert_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Convert info dict, handling JAX arrays (stays on same device)."""
        if info is None:
            return {}
        result = {}
        for k, v in info.items():
            if isinstance(v, jax.Array):
                result[k] = self._jax_to_torch(v)
            elif isinstance(v, dict):
                result[k] = self._convert_info(v)
            else:
                result[k] = v
        return result

    @staticmethod
    def _update_state_at_index(state, new_state, idx: int):
        """Update state at specific index for partial reset (single index)."""

        def update_at_idx(old, new):
            if isinstance(old, jax.Array) and old.ndim > 0:
                return old.at[idx].set(new)
            return old

        return jax.tree_util.tree_map(update_at_idx, state, new_state)

    @staticmethod
    def _update_state_at_indices(state, new_states, indices: jax.Array):
        """Vectorized update of state at multiple indices using scatter.

        :param state: Current batched state (num_envs, ...)
        :param new_states: New states for reset environments (n_resets, ...)
        :param indices: JAX array of indices to update (n_resets,)
        :return: Updated state with new values at specified indices
        """

        def scatter_update(old, new):
            if isinstance(old, jax.Array) and old.ndim > 0:
                # Use scatter to update multiple indices at once
                return old.at[indices].set(new)
            return old

        return jax.tree_util.tree_map(scatter_update, state, new_states)

    def _setup_renderer(self):
        """Setup MuJoCo renderer for visualization.

        MJX runs physics on GPU but rendering uses the standard MuJoCo renderer on CPU.
        Supports both "rgb_array" (offscreen) and "human" (GUI window) modes.
        """
        # Get the MuJoCo model from the environment
        # MuJoCo Playground envs store the model in different ways
        if hasattr(self.env, "mj_model"):
            self._mj_model = self.env.mj_model
        elif hasattr(self.env, "model"):
            self._mj_model = self.env.model
        elif hasattr(self.env, "_model"):
            self._mj_model = self.env._model
        else:
            # Try to get from sys (brax/mjx style)
            if hasattr(self.env, "sys") and hasattr(self.env.sys, "mj_model"):
                self._mj_model = self.env.sys.mj_model
            else:
                raise AttributeError(
                    f"Cannot find MuJoCo model in environment {self.env_name}. "
                    "Expected 'mj_model', 'model', '_model', or 'sys.mj_model' attribute."
                )

        # Create MuJoCo data for CPU rendering
        self._mj_data = mujoco.MjData(self._mj_model)

        # Setup renderer based on mode using camera_resolution
        width, height = self.camera_resolution
        if self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
            self._viewer = None
        elif self.render_mode == "human":
            self._renderer = None
            # Launch passive viewer (non-blocking GUI window)
            self._viewer = mujoco.viewer.launch_passive(self._mj_model, self._mj_data)

    def _sync_mjx_to_mujoco(self):
        """Copy MJX state to CPU MuJoCo data for rendering.

        MJX Data contains batched JAX arrays. For rendering, we extract the first
        environment's state and copy to CPU MuJoCo data.
        """
        if self._state is None or self._mj_data is None:
            return

        mjx_data = self._state.data

        # For batched envs, take first environment
        if self.num_envs > 1:
            qpos = np.asarray(mjx_data.qpos[0])
            qvel = np.asarray(mjx_data.qvel[0])
        else:
            qpos = np.asarray(mjx_data.qpos)
            qvel = np.asarray(mjx_data.qvel)

        # Copy to MuJoCo data
        self._mj_data.qpos[:] = qpos
        self._mj_data.qvel[:] = qvel

        # Forward kinematics to update positions
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def render(self) -> np.ndarray | None:
        """Render the environment.

        For render_mode="rgb_array": Returns RGB array (height, width, 3) uint8.
        For render_mode="human": Updates GUI window, returns None.
        """
        if self._state is None or self._mj_data is None:
            return None

        # Sync MJX state to CPU MuJoCo data
        self._sync_mjx_to_mujoco()

        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self._mj_data)
            return self._renderer.render()

        if self.render_mode == "human" and self._viewer is not None:
            # Sync viewer with updated data
            self._viewer.sync()
            return None

        return None

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    @property
    def device(self) -> str:
        """Return the simulation device."""
        return self.sim_device

    @property
    def unwrapped(self):
        """Return the unwrapped gym environment.

        Note: Returns self since MuJoCo Playground envs are not Gymnasium envs.
        This wrapper IS the base Gymnasium-compatible environment.
        """
        return self


class MjxPlaygroundStdWrapper(gym.Wrapper):
    """
    Wrapper that standardizes MuJoCo Playground observations to the format expected by Sb3EnvStdWrapper and handle auto-resets for terminated environments.

    Output format:
        {"policy": state_tensor, "camera": image_tensor (if cameras enabled in env)}
    """

    def __init__(self, env: MjxPlaygroundGymWrapper):
        """
        Initialize the standardization wrapper.

        :param env: MjxPlaygroundGymWrapper instance
        """
        super().__init__(env)

        self.num_envs = env.num_envs
        self.sim_device = env.sim_device

        # Build standardized observation space
        self._setup_observation_space()

    def _setup_observation_space(self):
        """Setup standardized observation space with 'policy' key."""
        original_space = self.env.single_observation_space

        if isinstance(original_space, spaces.Dict):
            new_spaces = {}
            for key, space in original_space.spaces.items():
                if key in ["state", "observation", "obs"]:
                    new_spaces["policy"] = space
                else:
                    new_spaces[key] = space
            self.single_observation_space = spaces.Dict(new_spaces)
        else:
            self.single_observation_space = spaces.Dict({"policy": original_space})

        self.single_action_space = self.env.single_action_space

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset and return standardized observations."""
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action: torch.Tensor | np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step and return standardized observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.render()

        # Handle auto-reset for done environments
        done = terminated | truncated
        if done.any():
            reset_idx = done.nonzero(as_tuple=True)[0]
            if reset_idx.numel() > 0:
                reset_obs, _ = self.env.reset(options={"env_idx": reset_idx})
                obs = self._merge_observations(obs, reset_obs, reset_idx)

        return self._process_observation(obs), reward, terminated, truncated, info

    def _process_observation(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert observation to standardized dict format with 'policy' key."""
        if isinstance(obs, torch.Tensor):
            return {"policy": obs}

        # Dict observation - rename state to policy
        processed = {}
        for key, value in obs.items():
            if key in ["state", "observation", "obs"]:
                processed["policy"] = value
            else:
                processed[key] = value

        return processed

    @staticmethod
    def _merge_observations(
        obs: torch.Tensor | dict[str, torch.Tensor],
        reset_obs: torch.Tensor | dict[str, torch.Tensor],
        reset_idx: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Merge reset observations into current observations at specific indices."""
        if isinstance(obs, dict):
            merged = {}
            for key in obs.keys():
                merged[key] = obs[key].clone()
                if isinstance(reset_obs, dict) and key in reset_obs:
                    merged[key][reset_idx] = reset_obs[key][reset_idx]
                elif not isinstance(reset_obs, dict):
                    merged[key][reset_idx] = reset_obs[reset_idx]
            return merged

        merged = obs.clone()
        merged[reset_idx] = reset_obs[reset_idx]
        return merged

    def seed(self, seed: int | None = None) -> None:
        """Seed the environment."""
        if seed is not None:
            self.env.reset(seed=seed)

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env.unwrapped

    @property
    def device(self) -> str:
        """Return the simulation device."""
        return self.sim_device
