"""Unit tests for policy classes."""

import pytest
import torch
from gymnasium import spaces

from algos.policies.ac_mlp_policy import MlpPolicy
from algos.policies.ac_lstm_policy import MlpLstmPolicy
from algos.policies.lstm_policy import LSTMPolicy
from algos.policies.tcn_policy import TCNPolicy
from tests.fixtures.conftest import *


class TestMlpPolicy:
    """Test suite for MLP policy."""

    @pytest.fixture
    def mlp_observation_space(self):
        """Observation space for MLP policy (dict with state and privileged)."""
        return spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "privileged": spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,)),
        })

    @pytest.fixture
    def action_space(self):
        """Action space."""
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))

    @pytest.fixture
    def lr_schedule(self):
        """Learning rate schedule."""
        return 3e-4

    def test_mlp_policy_forward(self, mlp_observation_space, action_space, lr_schedule, device):
        """Test MLP policy forward pass."""
        policy = MlpPolicy(
            observation_space=mlp_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[64, 64],
        ).to(device)

        batch_size = 8
        out_size = action_space.shape[-1]
        obs = {
            "state": torch.randn(batch_size, 10).to(device),
            "privileged": torch.randn(batch_size, 5).to(device),
        }

        # Test forward pass
        action, value, dist_out = policy.forward(obs)
        log_probs, mean, std = dist_out

        assert action.shape == (batch_size, out_size)
        assert value.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert mean.shape == (batch_size, out_size)
        assert std.shape == (batch_size, out_size)

    def test_mlp_policy_predict_values(self, mlp_observation_space, action_space, lr_schedule, device):
        """Test value prediction."""
        policy = MlpPolicy(
            observation_space=mlp_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[64, 64],
        ).to(device)

        batch_size = 8
        obs = {
            "state": torch.randn(batch_size, 10).to(device),
            "privileged": torch.randn(batch_size, 5).to(device),
        }

        values = policy.predict_values(obs)

        assert values.shape == (batch_size,)


class TestLSTMPolicyAC:
    """Test suite for LSTM Actor-Critic policy."""

    @pytest.fixture
    def lstm_observation_space(self):
        """Observation space for LSTM policy."""
        return spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
        })

    @pytest.fixture
    def action_space(self):
        """Action space."""
        return spaces.Box(low=-1.0, high=1.0, shape=(4,))

    @pytest.fixture
    def lr_schedule(self):
        """Learning rate schedule."""
        from algos.policies.modules.nn_utils import get_constant_schedule, get_schedule_fn
        return get_schedule_fn(get_constant_schedule(3e-4))


    def test_lstm_policy_forward(self, lstm_observation_space, action_space, lr_schedule, device):
        """Test LSTM policy forward pass."""
        policy = MlpLstmPolicy(
            observation_space=lstm_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=64,
            n_lstm_layers=1,
        ).to(device)

        batch_size = 4
        obs = {
            "state": torch.randn(batch_size, 10).to(device),
        }

        # Initialize hidden state
        hidden = torch.zeros(batch_size, *policy.hidden_state_shape, device=device)

        # Test forward pass
        action, value, dist_out, new_hidden = policy.forward(obs, hidden)
        log_probs, mean, std = dist_out

        assert action.shape == (batch_size, 4)
        assert value.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert mean.shape == (batch_size, 4)
        assert std.shape == (batch_size, 4)
        assert new_hidden is not None

    def test_lstm_policy_hidden_state(self, lstm_observation_space, action_space, lr_schedule, device):
        """Test hidden state propagation."""
        policy = MlpLstmPolicy(
            observation_space=lstm_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=64,
            n_lstm_layers=2,
        ).to(device)

        batch_size = 4
        obs = {
            "state": torch.randn(batch_size, 1, 10).to(device),
        }
        hidden = torch.zeros(batch_size, *policy.hidden_state_shape, device=device)

        # First step
        _, _, _, hidden1 = policy.forward(obs, hidden)

        # Second step with hidden state
        _, _, _, hidden2 = policy.forward(obs, hidden1)

        # Hidden states should be different
        assert not torch.allclose(hidden1[0], hidden2[0])


class TestLSTMPolicy:
    """Test suite for LSTM policy (SL version)."""

    @pytest.fixture
    def lstm_sl_observation_space(self):
        """Observation space for LSTM SL policy."""
        return spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "images": spaces.Box(low=0, high=1, shape=(2, 3, 94, 94), dtype='float32'),
        })

    @pytest.fixture
    def action_space(self):
        """Action space."""
        return spaces.Box(low=-1.0, high=1.0, shape=(7,))

    @pytest.fixture
    def lr_schedule(self):
        """Learning rate schedule."""
        return 3e-4

    def test_lstm_sl_policy_compute_loss(self, lstm_sl_observation_space, action_space, lr_schedule, device):
        """Test loss computation for LSTM SL policy."""
        policy = LSTMPolicy(
            observation_space=lstm_sl_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            dim_model=128,
            lstm_hidden=128,
            n_lstm_layers=1,
            action_horizon=4,
        ).to(device)

        batch_size = 2
        seq_len = 10
        obs = {
            "state": torch.randn(batch_size, seq_len, 10).to(device),
            "images": torch.randint(0, 1, (batch_size, seq_len, 2, 3, 94, 94), dtype=torch.float32).to(device),
            "action_is_pad": torch.ones(batch_size, seq_len, 7, dtype=torch.bool).to(device),
        }
        actions = torch.randn(batch_size, seq_len, 7).to(device)

        # Compute loss
        loss = policy.compute_loss(obs, actions)

        assert isinstance(loss, dict)
        assert list(loss.values())[0].ndim == 0  # Tensor scalar loss


class TestTCNPolicy:
    """Test suite for TCN policy."""

    @pytest.fixture
    def tcn_observation_space(self):
        """Observation space for TCN policy."""
        return spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "images": spaces.Box(low=0, high=1, shape=(2, 3, 94, 94), dtype='float32'),
        })

    @pytest.fixture
    def action_space(self):
        """Action space."""
        return spaces.Box(low=-1.0, high=1.0, shape=(7,))

    @pytest.fixture
    def lr_schedule(self):
        """Learning rate schedule."""
        return 3e-4

    def test_tcn_policy_compute_loss(self, tcn_observation_space, action_space, lr_schedule, device):
        """Test loss computation for TCN policy."""
        policy = TCNPolicy(
            observation_space=tcn_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            dim_model=128,
            num_channels=[64, 64],
            kernel_size=3,
            action_horizon=4,
        ).to(device)

        batch_size = 2
        seq_len = 10
        obs = {
            "state": torch.randn(batch_size, seq_len, 10).to(device),
            "images": torch.randint(0, 1, (batch_size, seq_len, 2, 3, 94, 94), dtype=torch.float32).to(device),
            "action_is_pad": torch.ones(batch_size, seq_len, 7, dtype=torch.bool).to(device),
        }
        actions = torch.randn(batch_size, seq_len, 7).to(device)

        # Compute loss
        loss = policy.compute_loss(obs, actions)

        assert isinstance(loss, dict)
        assert list(loss.values())[0].ndim == 0  # Tensor scalar loss


class TestPolicyCommonBehavior:
    """Test common behavior across all policies."""

    def test_policy_device_placement(self, device):
        """Test that policies can be moved to different devices."""
        obs_space = spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "privileged": spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,)),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        policy = MlpPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=3e-4,
            net_arch=[32, 32],
        )

        policy = policy.to(device)

        # Check parameters are on correct device
        for param in policy.parameters():
            assert param.device == device

    def test_policy_gradient_flow(self, device):
        """Test that gradients flow through the policy."""
        obs_space = spaces.Dict({
            "state": spaces.Box(low=-float('inf'), high=float('inf'), shape=(10,)),
            "privileged": spaces.Box(low=-float('inf'), high=float('inf'), shape=(5,)),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        policy = MlpPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=3e-4,
            net_arch=[32, 32],
        ).to(device)

        obs = {
            "state": torch.randn(4, 10).to(device),
            "privileged": torch.randn(4, 5).to(device),
        }

        action, value, log_prob = policy.forward(obs)
        loss = value.mean()
        loss.backward()

        # Check gradients exist
        for param in policy.value_net.parameters():
            assert param.grad is not None
