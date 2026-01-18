"""Mock environments for testing."""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv as SB3DummyVecEnv


class SimpleDummyEnv(gym.Env):
    """
    Simple dummy environment for testing.

    Mimics a basic continuous control task.
    Returns numpy observations (standard gym convention).
    """

    def __init__(self, obs_dim=10, action_dim=4, episode_length=100):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        obs = self.observation_space.sample()
        reward = float(np.random.randn())
        terminated = self.current_step >= self.episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}


class DictObsDummyEnv(gym.Env):
    """
    Dummy environment with dictionary observations.

    Useful for testing multi-modal inputs.
    Returns numpy observations (standard gym convention).
    """

    def __init__(self, state_dim=8, priv_dim=6, image_size=94, action_dim=4, episode_length=100):
        super().__init__()
        self.episode_length = episode_length
        self.current_step = 0

        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32),
            "images": spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = {
            "state": self.observation_space["state"].sample(),
            "privileged": self.observation_space["privileged"].sample(),
            "images": self.observation_space["images"].sample(),
        }
        return obs, {}

    def step(self, action):
        self.current_step += 1
        obs = {
            "state": self.observation_space["state"].sample(),
            "privileged": self.observation_space["privileged"].sample(),
            "images": self.observation_space["images"].sample(),
        }
        reward = float(np.random.randn())
        terminated = self.current_step >= self.episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}


class DiscreteActionEnv(gym.Env):
    """
    Dummy environment with discrete actions.

    Returns numpy observations (standard gym convention).
    """

    def __init__(self, obs_dim=10, num_actions=5, episode_length=100):
        super().__init__()
        self.episode_length = episode_length
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        obs = self.observation_space.sample()
        reward = float(np.random.randn())
        terminated = self.current_step >= self.episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}


def make_dummy_env(env_type="simple", **kwargs):
    """
    Factory function to create dummy environments.

    Args:
        env_type: Type of environment ("simple", "dict_obs", "discrete")
        **kwargs: Additional arguments passed to the environment

    Returns:
        Gym environment instance
    """
    env_classes = {
        "simple": SimpleDummyEnv,
        "dict_obs": DictObsDummyEnv,
        "discrete": DiscreteActionEnv,
    }

    if env_type not in env_classes:
        raise ValueError(f"Unknown env_type: {env_type}. Choose from {list(env_classes.keys())}")

    return env_classes[env_type](**kwargs)


class DummyVecEnv(SB3DummyVecEnv):
    """
    Custom DummyVecEnv that converts numpy observations to tensors.

    Standard SB3 DummyVecEnv works with numpy arrays.
    This wrapper converts observations to tensors (mimicking wrapped envs like Sb3EnvStdWrapper).
    Environments return numpy (standard gym convention), VecEnv handles tensorization.
    """

    def __init__(self, env_fns, device="cpu"):
        """
        Initialize DummyVecEnv with device support.

        Args:
            env_fns: List of functions that create environments
            device: Device to place tensors on (e.g., "cpu", "cuda", torch.device)
        """
        super().__init__(env_fns)
        self.device = torch.device(device) if isinstance(device, str) else device

    def reset(self):
        """Reset all environments and return tensor observations."""
        obs = []
        for env_idx in range(self.num_envs):
            obs_tuple = self.envs[env_idx].reset()
            # Handle both (obs,) and (obs, info) return formats
            if isinstance(obs_tuple, tuple):
                obs.append(obs_tuple[0])
            else:
                obs.append(obs_tuple)

        # Convert numpy to tensors and stack on device
        if isinstance(obs[0], dict):
            # Dict observations - convert each key from numpy to tensor and stack
            stacked_obs = {}
            for key in obs[0].keys():
                numpy_arrays = [o[key] for o in obs]
                stacked_obs[key] = torch.stack([torch.from_numpy(arr) for arr in numpy_arrays]).to(self.device)
            return stacked_obs
        else:
            # Simple numpy observations - convert to tensors and stack
            return torch.stack([torch.from_numpy(o) for o in obs]).to(self.device)

    def step_wait(self):
        """Wait for step to complete and return tensor observations."""
        for env_idx, action in enumerate(self.actions):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(action)
            # Handle new and old gym done semantics
            done = terminated or truncated
            if done:
                # Save final observation where user can get it
                self.buf_infos[env_idx]["terminal_obs"] = obs
                # Reset and get new observation
                obs = self.envs[env_idx].reset()[0]
            if isinstance(obs, dict):
                for key in obs.keys():
                    self.buf_obs[key][env_idx] = obs[key]
            else:
                self.buf_obs[env_idx] = obs
            self.buf_dones[env_idx] = done

        self.actions = None

        # Convert numpy arrays to tensors and move to device
        if isinstance(self.buf_obs, dict):
            # Dict observations - self.buf_obs is OrderedDict with keys -> numpy arrays
            obs_tensors = {key: torch.from_numpy(val).to(self.device) for key, val in self.buf_obs.items()}
        else:
            # Simple observations - self.buf_obs is a numpy array
            obs_tensors = torch.from_numpy(self.buf_obs).to(self.device)

        # Restructure infos from list of dicts to dict of lists
        infos_dict = {}
        if len(self.buf_infos) > 0:
            # Collect all keys from all info dicts
            all_keys = set()
            for info in self.buf_infos:
                all_keys.update(info.keys())

            # For each key, create a list of values from each env
            for key in all_keys:
                values = [info.get(key, None) for info in self.buf_infos]
                # Convert terminal_obs to tensor format for SAC compatibility
                if key == "terminal_obs":
                    # SAC expects terminal_obs as dict of tensors (for dict obs) or tensor (for simple obs)
                    first_valid = next((v for v in values if v is not None), None)
                    if first_valid is not None:
                        if isinstance(first_valid, dict):
                            # Dict observations - stack each key into tensor
                            infos_dict[key] = {
                                obs_key: torch.stack([
                                    torch.from_numpy(v[obs_key]) if v is not None else torch.zeros_like(torch.from_numpy(first_valid[obs_key]))
                                    for v in values
                                ]).to(self.device)
                                for obs_key in first_valid.keys()
                            }
                        else:
                            # Simple observations - stack into tensor
                            infos_dict[key] = torch.stack([
                                torch.from_numpy(v) if v is not None else torch.zeros_like(torch.from_numpy(first_valid))
                                for v in values
                            ]).to(self.device)
                    else:
                        # No terminal obs this step - create placeholder with NaN based on observation space
                        if isinstance(self.buf_obs, dict):
                            infos_dict[key] = {
                                obs_key: torch.full((self.num_envs, *arr.shape[1:]), float("nan"), dtype=torch.float32).to(self.device)
                                for obs_key, arr in self.buf_obs.items()
                            }
                        else:
                            infos_dict[key] = torch.full((self.num_envs, *self.buf_obs.shape[1:]), float("nan"), dtype=torch.float32).to(self.device)
                else:
                    infos_dict[key] = values

        # Ensure terminal_obs always exists for SAC compatibility (use NaN for invalid/placeholder values)
        if "terminal_obs" not in infos_dict:
            if isinstance(self.buf_obs, dict):
                infos_dict["terminal_obs"] = {
                    obs_key: torch.full((self.num_envs, *arr.shape[1:]), float("nan"), dtype=torch.float32).to(self.device)
                    for obs_key, arr in self.buf_obs.items()
                }
            else:
                infos_dict["terminal_obs"] = torch.full((self.num_envs, *self.buf_obs.shape[1:]), float("nan"), dtype=torch.float32).to(self.device)

        return (
            obs_tensors,
            torch.from_numpy(self.buf_rews).to(self.device),
            torch.from_numpy(self.buf_dones).to(self.device),
            infos_dict,
        )