"""
Smoke tests for quick validation.

These tests run quickly and verify basic functionality.
Run with: pytest tests/test_smoke.py
"""

import pytest
import torch
from gymnasium import spaces


@pytest.mark.smoke
class TestBasicImports:
    """Test that all critical modules can be imported."""

    def test_import_algos(self):
        """Test importing algorithm modules."""
        try:
            from algos import ppo_agent, ppo_recurrent_agent, sl_agent
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import algos: {e}")

    def test_import_policies(self):
        """Test importing policy modules."""
        try:
            from algos.policies import ac_mlp_policy, lstm_policy, tcn_policy
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import policies: {e}")

    def test_import_common(self):
        """Test importing common modules."""
        try:
            from common import preprocessor, utils, callbacks
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import common: {e}")

    def test_import_buffers(self):
        """Test importing buffer modules."""
        try:
            from algos.storage.buffers import RolloutBuffer, ReplayBuffer
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import buffers: {e}")


@pytest.mark.smoke
class TestBasicFunctionality:
    """Test basic functionality of key components."""

    def test_create_simple_policy(self):
        """Test creating a simple policy."""
        from algos.policies.ac_mlp_policy import MlpPolicy

        obs_space = spaces.Dict({
            "state": spaces.Box(low=-1.0, high=1.0, shape=(10,)),
            "privileged": spaces.Box(low=-1.0, high=1.0, shape=(5,)),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        policy = MlpPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=3e-4,
            net_arch=[32, 32],
        )

        assert policy is not None

    def test_create_preprocessor(self):
        """Test creating a preprocessor."""
        from common.preprocessor import Preprocessor

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,))
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        preprocessor = Preprocessor(
            observation_space=obs_space,
            action_space=action_space,
        )

        assert preprocessor is not None

    def test_create_buffer(self):
        """Test creating a buffer."""
        from algos.storage.buffers import RolloutBuffer

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,))
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        buffer = RolloutBuffer(
            buffer_size=128,
            n_envs=4,
            observation_space=obs_space,
            action_space=action_space,
            device=torch.device("cpu"),
            gae_lambda=0.95,
            gamma=0.99,
        )

        assert buffer is not None

    def test_mock_environment(self):
        """Test creating a mock environment."""
        from tests.fixtures.mock_envs import make_dummy_env

        env = make_dummy_env("simple", obs_dim=10, action_dim=4)

        obs, info = env.reset()
        assert obs is not None

        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert obs is not None


@pytest.mark.smoke
class TestDataFlow:
    """Test basic data flow through components."""

    def test_policy_forward_pass(self):
        """Test policy forward pass."""
        from algos.policies.ac_mlp_policy import MlpPolicy

        obs_space = spaces.Dict({
            "state": spaces.Box(low=-1.0, high=1.0, shape=(10,)),
            "privileged": spaces.Box(low=-1.0, high=1.0, shape=(5,)),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        policy = MlpPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=3e-4,
            net_arch=[32, 32],
        )

        obs = {
            "state": torch.randn(8, 10),
            "privileged": torch.randn(8, 5),
        }

        action, value, dist_out = policy.forward(obs)
        log_probs, mean, std = dist_out

        assert action.shape == (8, 4)
        assert value.shape == (8,)
        assert log_probs.shape == (8,)
        assert mean.shape == (8, 4)
        assert std.shape == (8, 4)

    def test_preprocessor_normalization(self):
        """Test preprocessor normalization."""
        from common.preprocessor import Preprocessor
        import numpy as np

        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10,))
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        stats = {
            "": {
                "mean": np.zeros(10, dtype=np.float32),
                "std": np.ones(10, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=obs_space,
            action_space=action_space,
            observation_modes={"": "mean_std"},
            data_key_map={"": "default"},
            stats=stats,
        )

        obs = torch.randn(8, 10)
        normalized = preprocessor.normalize_observations(obs)

        assert normalized.shape == (8, 10)


@pytest.mark.smoke
class TestConfigLoading:
    """Test configuration loading."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading a YAML config."""
        import yaml

        config = {
            "policy": "MlpPolicy",
            "n_steps": 128,
            "learning_rate": 3e-4,
        }

        config_file = tmp_path / "test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        with open(config_file, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded == config

if __name__ == "__main__":
    # Allow running smoke tests directly
    pytest.main([__file__, "-v", "-m", "smoke"])