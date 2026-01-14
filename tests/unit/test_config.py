"""Unit tests for configuration loading and validation."""

import os
import tempfile

import pytest
import yaml

from common.envs.sb3_env_wrapper import process_sb3_cfg


class TestAgentConfigFiles:
    """Test existing agent configuration files."""

    def test_all_agent_configs_loadable(self):
        """Test that all agent configs can be loaded."""

        original_cwd = os.getcwd()

        # Check if src directory exists in current working directory
        src_dir = os.path.join(original_cwd, "src")
        if os.path.isdir(src_dir):
            os.chdir(src_dir)
            print(f"Changed working directory from {original_cwd} to {src_dir}")

        config_dir = "configs/agents"

        if not os.path.exists(config_dir):
            pytest.skip(f"Config directory not found: {config_dir}")

        # Find all YAML files
        config_files = []
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                if file.endswith(".yaml"):
                    config_files.append(os.path.join(root, file))

        # Load each config
        for config_file in config_files:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                assert config is not None, f"Failed to load {config_file}"


class TestProcessSb3Cfg:
    """Test configuration processing for SB3."""

    def test_process_simple_config(self):
        """Test processing simple configuration."""
        cfg = {
            "learning_rate": 3e-4,
            "n_steps": 128,
            "batch_size": 64,
        }

        processed = process_sb3_cfg(cfg)

        assert processed["learning_rate"] == 3e-4
        assert processed["n_steps"] == 128
        assert processed["batch_size"] == 64

    def test_process_clip_range_constant(self):
        """Test processing constant clip_range."""
        cfg = {
            "clip_range": 0.2,
        }

        processed = process_sb3_cfg(cfg)

        # Should be converted to a callable
        assert callable(processed["clip_range"])
        assert processed["clip_range"](0.5) == 0.2

    def test_process_clip_range_schedule(self):
        """Test processing scheduled clip_range."""
        cfg = {
            "clip_range": "lin_0.2",
        }

        processed = process_sb3_cfg(cfg)

        # Should be converted to a lambda
        assert callable(processed["clip_range"])
        # At progress_remaining=1.0, should be 0.2
        assert processed["clip_range"](1.0) == 0.2
        # At progress_remaining=0.5, should be 0.1
        assert processed["clip_range"](0.5) == 0.1

    def test_process_policy_kwargs_string(self):
        """Test processing policy_kwargs as string."""
        cfg = {
            "policy_kwargs": "dict(net_arch=[256, 256])",
        }

        processed = process_sb3_cfg(cfg)

        assert isinstance(processed["policy_kwargs"], dict)
        assert "net_arch" in processed["policy_kwargs"]

    def test_process_policy_kwargs_dict(self):
        """Test processing policy_kwargs as dict."""
        cfg = {
            "policy_kwargs": {"net_arch": [128, 128], "activation_fn": "relu"},
        }

        processed = process_sb3_cfg(cfg)

        assert processed["policy_kwargs"]["net_arch"] == [128, 128]
        assert processed["policy_kwargs"]["activation_fn"] == "relu"

    def test_process_nested_config(self):
        """Test processing nested configuration."""
        cfg = {
            "agent": {
                "learning_rate": 3e-4,
                "clip_range": 0.2,
            },
            "env": {
                "num_envs": 4,
            }
        }

        processed = process_sb3_cfg(cfg)

        assert processed["agent"]["learning_rate"] == 3e-4
        assert callable(processed["agent"]["clip_range"])
        assert processed["env"]["num_envs"] == 4

    def test_process_negative_clip_range(self):
        """Test that negative clip_range is ignored."""
        cfg = {
            "clip_range": -1,
        }

        processed = process_sb3_cfg(cfg)

        # Negative values should be kept as-is or skipped
        assert "clip_range" in processed

    def test_process_none_clip_range(self):
        """Test that None clip_range is handled."""
        cfg = {
            "clip_range": None,
        }

        processed = process_sb3_cfg(cfg)

        # None should be handled gracefully
        assert "clip_range" in processed
