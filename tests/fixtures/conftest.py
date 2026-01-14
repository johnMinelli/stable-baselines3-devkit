"""Shared pytest fixtures for the test suite."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Return CPU device for testing."""
    return torch.device("cuda")


@pytest.fixture
def simple_observation_space():
    """Simple flat observation space."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)



@pytest.fixture
def dict_observation_space():
    """Dictionary observation space with multiple modalities."""
    return spaces.Dict({
        "state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        "images": spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8),
    })


@pytest.fixture
def image_observation_space():
    """Image-based observation space."""
    return spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)


@pytest.fixture
def simple_action_space():
    """Simple flat action space."""
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)


@pytest.fixture
def discrete_action_space():
    """Discrete action space."""
    return spaces.Discrete(5)


@pytest.fixture
def multi_discrete_action_space():
    """Multi-discrete action space."""
    return spaces.MultiDiscrete([3, 4, 2])


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 4


@pytest.fixture
def sequence_length():
    """Default sequence length for recurrent models."""
    return 8


@pytest.fixture
def num_envs():
    """Number of parallel environments."""
    return 2


@pytest.fixture
def minimal_ppo_config():
    """Minimal PPO configuration for testing."""
    return {
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }


@pytest.fixture
def minimal_policy_kwargs():
    """Minimal policy kwargs for testing."""
    return {
        "net_arch": {"pi": [32, 32], "vf": [32, 32]},
        "activation_fn": torch.nn.Tanh,
    }


@pytest.fixture(autouse=True)
def reset_random_seeds(seed):
    """Reset random seeds before each test."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@pytest.fixture
def dummy_batch_data(batch_size, simple_observation_space, simple_action_space):
    """Generate dummy batch data for testing."""
    obs = np.random.randn(batch_size, *simple_observation_space.shape).astype(np.float32)
    actions = np.random.uniform(
        simple_action_space.low,
        simple_action_space.high,
        size=(batch_size, *simple_action_space.shape)
    ).astype(np.float32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    dones = np.random.randint(0, 2, size=batch_size).astype(bool)

    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


@pytest.fixture
def dummy_dict_batch_data(batch_size, dict_observation_space, simple_action_space):
    """Generate dummy batch data with dict observations."""
    obs = {
        "state": np.random.randn(batch_size, 8).astype(np.float32),
        "proprio": np.random.randn(batch_size, 6).astype(np.float32),
        "images": np.random.randint(0, 255, size=(batch_size, 3, 84, 84), dtype=np.uint8),
    }
    actions = np.random.uniform(
        simple_action_space.low,
        simple_action_space.high,
        size=(batch_size, *simple_action_space.shape)
    ).astype(np.float32)

    return {
        "observations": obs,
        "actions": actions,
    }


@pytest.fixture
def dummy_sequence_data(batch_size, sequence_length, simple_observation_space, simple_action_space):
    """Generate dummy sequence data for recurrent models."""
    obs = np.random.randn(batch_size, sequence_length, *simple_observation_space.shape).astype(np.float32)
    actions = np.random.uniform(
        simple_action_space.low,
        simple_action_space.high,
        size=(batch_size, sequence_length, *simple_action_space.shape)
    ).astype(np.float32)

    return {
        "observations": obs,
        "actions": actions,
    }
