"""Integration tests for preprocessor-policy data flow."""

import pytest
import torch
from gymnasium import spaces

from common.preprocessor import Preprocessor
from tests.fixtures.conftest import device

class TestPreprocessorPolicyDataFlow:
    """Test data flow between preprocessor and policy."""

    @pytest.fixture
    def mlp_setup(self, device):
        """Setup for MLP policy test."""
        obs_space = spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "privileged": spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,)),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        return obs_space, action_space, device

    @pytest.fixture
    def lr_schedule(self):
        """Learning rate schedule."""
        from algos.policies.modules.nn_utils import get_constant_schedule, get_schedule_fn
        return get_schedule_fn(get_constant_schedule(3e-4))

    def test_preprocessor_normalization_flow(self, device):
        """Test that normalization statistics flow correctly."""
        obs_space = spaces.Dict({"state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,))})
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        # Create preprocessor with normalization
        observation_modes = {"state": "mean_std"}
        action_modes = {"": "mean_std"}
        data_key_map = {"state": "default_obs", "": "default_obs"}

        import numpy as np
        stats = {
            "state": {
                "mean": np.array([1.0] * 10, dtype=np.float32),
                "std": np.array([2.0] * 10, dtype=np.float32),
            },
            "": {
                "mean": np.array([1.0] * 4, dtype=np.float32),
                "std": np.array([2.0] * 4, dtype=np.float32),
            },
        }

        preprocessor = Preprocessor(
            observation_space=obs_space,
            action_space=action_space,
            observation_modes=observation_modes,
            action_modes=action_modes,
            data_key_map=data_key_map,
            overwrite_stats=True,
            stats=stats,
            device=device,
        )

        # Test observation normalization
        obs = torch.tensor([3.0] * 10, dtype=torch.float32).to(device)
        normalized_obs = preprocessor.normalize_observations(obs)

        # Expected: (3 - 1) / 2 = 1.0
        expected = torch.ones(10, dtype=torch.float32).to(device)
        assert torch.allclose(normalized_obs, expected, atol=1e-5)

        # Test action normalization
        actions = torch.tensor([1.5] * 4, dtype=torch.float32).to(device)
        normalized_actions = preprocessor.normalize_actions(actions)

        # Test action un-normalization
        unnormalized_actions = preprocessor.unnormalize_actions(normalized_actions)
        assert torch.allclose(unnormalized_actions, actions, atol=1e-5)

    def test_image_preprocessing_flow(self, device):
        """Test image preprocessing through the pipeline."""
        try:
            from common.envs.aloha_preprocessor import Aloha_2_Mlp

            obs_space = spaces.Dict({
                "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
                "images": spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype='uint8'),
            })
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))

            preprocessor = Aloha_2_Mlp(
                observation_space=obs_space,
                action_space=action_space,
                normalize_images=True,
                device=device,
            )

            # Generate image data
            batch_size = 2
            raw_obs = {
                "state": torch.randn(batch_size, 10).to(device),
                "image": torch.randint(0, 255, (batch_size, 3, 84, 84), dtype=torch.uint8).to(device),
            }

            # Preprocess
            processed_obs = preprocessor.preprocess_observations(raw_obs)

            # Check that images are normalized
            if "images" in processed_obs:
                assert processed_obs["images"].dtype == torch.float32
                # After normalization, values should be in a reasonable range
                assert processed_obs["images"].min() >= -2.0
                assert processed_obs["images"].max() <= 2.0

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_mlp_preprocessor_to_policy(self, mlp_setup):
        """Test data flow from preprocessor to MLP policy."""
        try:
            from common.envs.gym_preprocessor import Gym_2_Mlp
            from algos.policies.ac_mlp_policy import MlpPolicy

            obs_space, action_space, device = mlp_setup

            # Create preprocessor
            preprocessor = Gym_2_Mlp(
                observation_space=obs_space,
                action_space=action_space,
                device=device,
            )

            # Create policy
            policy = MlpPolicy(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=3e-4,
                net_arch=[64, 64],
            ).to(device)

            # Generate dummy data
            batch_size = 8
            raw_obs = {
                "state": torch.randn(batch_size, 10).to(device),
                "privileged": torch.randn(batch_size, 5).to(device),
            }

            # Preprocess
            processed_obs = preprocessor.preprocess_observations(raw_obs)

            # Forward through policy
            action, value, dist_out = policy.forward(processed_obs)
            log_probs, mean, std = dist_out

            # Check outputs
            assert action.shape == (batch_size, 4)
            assert value.shape == (batch_size,)
            assert log_probs.shape == (batch_size,)
            assert mean.shape == (batch_size, 4)
            assert std.shape == (batch_size, 4)
        
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_lstm_preprocessor_to_policy(self, lr_schedule, device):
        """Test data flow from preprocessor to LSTM policy."""
        try:
            from common.envs.gym_preprocessor import Gym_2_Lstm
            from algos.policies.ac_lstm_policy import MlpLstmPolicy

            obs_space = spaces.Dict({
                "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            })
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

            # Create preprocessor
            preprocessor = Gym_2_Lstm(
                observation_space=obs_space,
                action_space=action_space,
                device=device,
            )

            # Create policy
            policy = MlpLstmPolicy(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=lr_schedule,
                lstm_hidden_size=64,
                n_lstm_layers=1,
            ).to(device)

            # Generate dummy sequential data
            batch_size = 4
            raw_obs = {
                "state": torch.randn(batch_size, 10).to(device),
            }

            # Preprocess
            processed_obs = preprocessor.preprocess_observations(raw_obs)
            hidden = torch.zeros(batch_size, *policy.hidden_state_shape, device=device)

            # Forward through policy
            action, value, dist_out, new_hidden = policy.forward(processed_obs, hidden)
            log_probs, mean, std = dist_out
        
            # Check outputs
            assert action.shape == (batch_size, 4)
            assert value.shape == (batch_size,)
            assert log_probs.shape == (batch_size,)
            assert mean.shape == (batch_size, 4)
            assert std.shape == (batch_size, 4)
            assert new_hidden is not None

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestPreprocessorActionPostprocessing:
    """Test action postprocessing from policy outputs."""

    def test_action_denormalization(self, device):
        """Test that actions are correctly denormalized."""
        obs_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,))
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        action_modes = {"": "mean_std"}
        data_key_map = {"": "default"}

        import numpy as np
        stats = {
            "": {
                "mean": np.array([0.5] * 4, dtype=np.float32),
                "std": np.array([0.5] * 4, dtype=np.float32),
            }
        }

        preprocessor = Preprocessor(
            observation_space=obs_space,
            action_space=action_space,
            action_modes=action_modes,
            data_key_map=data_key_map,
            stats=stats,
            device=device,
        )

        # Normalized policy output (in normalized space)
        normalized_actions = torch.zeros(4, dtype=torch.float32).to(device)

        # Denormalize
        denormalized = preprocessor.unnormalize_actions(normalized_actions)

        # Expected: 0 * 0.5 + 0.5 = 0.5
        expected = torch.tensor([0.5] * 4, dtype=torch.float32).to(device)
        assert torch.allclose(denormalized, expected, atol=1e-5)
