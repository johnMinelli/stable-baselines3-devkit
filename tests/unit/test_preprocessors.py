"""Unit tests for preprocessor classes."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from common.preprocessor import Preprocessor
from tests.fixtures.conftest import *


class TestPreprocessor:
    """Test suite for the base Preprocessor class."""

    def test_initialization_simple_spaces(self, simple_observation_space, simple_action_space, device):
        """Test preprocessor initialization with simple spaces."""
        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            device=device,
        )

        assert preprocessor.observation_space == simple_observation_space
        assert preprocessor.action_space == simple_action_space
        assert preprocessor.device == device

    def test_initialization_with_normalization_modes(self, simple_observation_space, simple_action_space, device):
        """Test preprocessor with normalization modes."""
        observation_modes = {"": "mean_std"}  # Empty key for non-dict spaces
        action_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            action_modes=action_modes,
            data_key_map=data_key_map,
            device=device,
        )

        # Check that buffers are created
        assert hasattr(preprocessor, "obs_default_mean")
        assert hasattr(preprocessor, "obs_default_std")
        assert hasattr(preprocessor, "action_default_mean")
        assert hasattr(preprocessor, "action_default_std")

        # Check buffer shapes
        assert preprocessor.obs_default_mean.shape == simple_observation_space.shape
        assert preprocessor.obs_default_std.shape == simple_observation_space.shape

    def test_normalize_observations_mean_std(self, simple_observation_space, simple_action_space, device):
        """Test observation normalization with mean-std mode."""
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        # Create stats
        stats = {
            "": {
                "mean": np.array([1.0] * 10, dtype=np.float32),
                "std": np.array([2.0] * 10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Test normalization
        obs = torch.tensor([3.0] * 10, dtype=torch.float32).to(device)
        normalized = preprocessor.normalize_observations(obs)

        # Expected: (3 - 1) / 2 = 1.0
        expected = torch.ones(10, dtype=torch.float32).to(device)
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_normalize_observations_min_max(self, simple_observation_space, simple_action_space, device):
        """Test observation normalization with min-max mode."""
        observation_modes = {"": "min_max"}
        data_key_map = {"": "default"}

        # Create stats
        stats = {
            "": {
                "min": np.array([0.0] * 10, dtype=np.float32),
                "max": np.array([10.0] * 10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Test normalization
        obs = torch.tensor([5.0] * 10, dtype=torch.float32).to(device)
        normalized = preprocessor.normalize_observations(obs)

        # Expected: (5 - 0) / 10 * 2 - 1 = 0.0
        expected = torch.zeros(10, dtype=torch.float32).to(device)
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_normalize_actions_mean_std(self, simple_observation_space, simple_action_space, device):
        """Test action normalization with mean-std mode."""
        action_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        stats = {
            "": {
                "mean": np.array([0.5] * 4, dtype=np.float32),
                "std": np.array([0.5] * 4, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            action_modes=action_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        actions = torch.tensor([1.0] * 4, dtype=torch.float32).to(device)
        normalized = preprocessor.normalize_actions(actions)

        # Expected: (1.0 - 0.5) / 0.5 = 1.0
        expected = torch.ones(4, dtype=torch.float32).to(device)
        assert torch.allclose(normalized, expected, atol=1e-5)

    def test_unnormalize_actions_mean_std(self, simple_observation_space, simple_action_space, device):
        """Test action un-normalization with mean-std mode."""
        action_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        stats = {
            "": {
                "mean": np.array([0.5] * 4, dtype=np.float32),
                "std": np.array([0.5] * 4, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            action_modes=action_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Normalized actions
        normalized_actions = torch.ones(4, dtype=torch.float32).to(device)
        unnormalized = preprocessor.unnormalize_actions(normalized_actions)

        # Expected: 1.0 * 0.5 + 0.5 = 1.0
        expected = torch.ones(4, dtype=torch.float32).to(device)
        assert torch.allclose(unnormalized, expected, atol=1e-5)

    def test_unnormalize_actions_min_max(self, simple_observation_space, simple_action_space, device):
        """Test action un-normalization with min-max mode."""
        action_modes = {"": "min_max"}
        data_key_map = {"": "default"}

        stats = {
            "": {
                "min": np.array([0.0] * 4, dtype=np.float32),
                "max": np.array([2.0] * 4, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            action_modes=action_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Normalized actions (in [-1, 1] range)
        normalized_actions = torch.zeros(4, dtype=torch.float32).to(device)
        unnormalized = preprocessor.unnormalize_actions(normalized_actions)

        # Expected: ((0 + 1) / 2) * 2 + 0 = 1.0
        expected = torch.ones(4, dtype=torch.float32).to(device)
        assert torch.allclose(unnormalized, expected, atol=1e-5)

    def test_dict_observations(self, dict_observation_space, simple_action_space, device):
        """Test preprocessor with dictionary observations."""
        observation_modes = {
            "state": "mean_std",
            "privileged": "mean_std",
        }
        data_key_map = {
            "state": "state",
            "privileged": "privileged",
        }

        stats = {
            "state": {
                "mean": np.zeros(8, dtype=np.float32),
                "std": np.ones(8, dtype=np.float32),
            },
            "privileged": {
                "mean": np.zeros(6, dtype=np.float32),
                "std": np.ones(6, dtype=np.float32),
            },
        }

        preprocessor = Preprocessor(
            observation_space=dict_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Test with dict observations
        obs_dict = {
            "state": torch.randn(8).to(device),
            "privileged": torch.randn(6).to(device),
            "images": torch.randint(0, 255, (3, 84, 84), dtype=torch.uint8).to(device),
        }

        normalized = preprocessor.normalize_observations(obs_dict)

        assert "state" in normalized
        assert "privileged" in normalized
        assert "images" in normalized

    def test_overwrite_stats(self, simple_observation_space, simple_action_space, device):
        """Test overwriting normalization statistics."""
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        # Initial stats
        stats1 = {
            "": {
                "mean": np.array([1.0] * 10, dtype=np.float32),
                "std": np.array([1.0] * 10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats1,
            device=device,
        )

        initial_mean = preprocessor.obs_default_mean.clone()

        # New stats
        stats2 = {
            "": {
                "mean": np.array([2.0] * 10, dtype=np.float32),
                "std": np.array([2.0] * 10, dtype=np.float32),
            }
        }

        # Overwrite
        preprocessor2 = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats2,
            overwrite_stats=True,
            device=device,
        )

        # Mean should be different
        assert not torch.allclose(initial_mean, preprocessor2.obs_default_mean)
        assert torch.allclose(preprocessor2.obs_default_mean, torch.tensor([2.0] * 10).to(device))

    def test_normalize_images(self, image_observation_space, simple_action_space, device):
        """Test image normalization."""
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "images"}

        stats = {
            "": {
                "mean": np.zeros((3, 1, 1), dtype=np.float32),
                "std": np.ones((3, 1, 1), dtype=np.float32) * 255.0,
            }
        }

        preprocessor = Preprocessor(
            observation_space=image_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            normalize_images=True,
            device=device,
        )

        # Create image observation
        img = torch.randint(0, 255, (3, 64, 64), dtype=torch.float32).to(device)
        normalized = preprocessor.normalize_observations(img)

        # Values should be roughly in [-1, 1] range
        assert normalized.min() >= -2.0
        assert normalized.max() <= 2.0

    def test_stats_not_initialized_error(self, simple_observation_space, simple_action_space, device):
        """Test that error is raised when stats are not initialized."""
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            device=device,
        )

        # Manually set stats to infinity to simulate uninitialized
        preprocessor.obs_default_mean = torch.full(simple_observation_space.shape, float('inf')).to(device)

        obs = torch.randn(10).to(device)

        # Should raise ValueError
        with pytest.raises(ValueError, match="infinity"):
            preprocessor.normalize_observations(obs)

    def test_batch_normalization(self, simple_observation_space, simple_action_space, batch_size, device):
        """Test normalization with batched data."""
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        stats = {
            "": {
                "mean": np.zeros(10, dtype=np.float32),
                "std": np.ones(10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Batched observations
        obs_batch = torch.randn(batch_size, 10).to(device)
        normalized = preprocessor.normalize_observations(obs_batch)

        assert normalized.shape == (batch_size, 10)

    def test_device_placement(self, simple_observation_space, simple_action_space):
        """Test that buffers are placed on correct device."""
        device = torch.device("cpu")
        observation_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        stats = {
            "": {
                "mean": np.zeros(10, dtype=np.float32),
                "std": np.ones(10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            observation_modes=observation_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        assert preprocessor.obs_default_mean.device == device
        assert preprocessor.obs_default_std.device == device