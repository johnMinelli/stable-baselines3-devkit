"""Unit tests for buffer classes."""

import pytest
import torch
from gymnasium import spaces

from algos.storage.buffers import ReplayBuffer, RolloutBuffer
from tests.fixtures.conftest import *


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""

    @pytest.fixture
    def buffer_config(self, simple_observation_space, simple_action_space, device):
        """Basic buffer configuration."""
        return {
            "buffer_size": 100,
            "n_envs": 4,
            "observation_space": simple_observation_space,
            "action_space": simple_action_space,
            "device": device,
        }

    def test_replay_buffer_initialization(self, buffer_config):
        """Test replay buffer initialization."""
        buffer = ReplayBuffer(**buffer_config)

        assert buffer.buffer_size == 100
        assert buffer.n_envs == 4
        assert buffer.pos == 0
        assert buffer.full is False

    def test_replay_buffer_add(self, buffer_config, device):
        """Test adding experiences to replay buffer."""
        buffer = ReplayBuffer(**buffer_config)

        n_envs = buffer_config["n_envs"]
        obs_shape = buffer_config["observation_space"].shape
        action_shape = buffer_config["action_space"].shape

        obs = torch.randn(n_envs, *obs_shape).to(device)
        next_obs = torch.randn(n_envs, *obs_shape).to(device)
        actions = torch.randn(n_envs, *action_shape).to(device)
        rewards = torch.randn(n_envs).to(device)
        dones = torch.zeros(n_envs).to(device)

        buffer.add(obs, next_obs, actions, rewards, dones)

        assert buffer.pos == 1
        assert buffer.full is False

    def test_fill_replay_buffer(self, buffer_config, device):
        """Test buffer wraparound when full."""
        buffer_config["buffer_size"] = 10
        buffer = ReplayBuffer(**buffer_config)

        n_envs = buffer_config["n_envs"]
        obs_shape = buffer_config["observation_space"].shape
        action_shape = buffer_config["action_space"].shape

        # Fill the buffer
        for _ in range(10):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            next_obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            buffer.add(obs, next_obs, actions, rewards, dones)

        assert buffer.pos == 0  # pos restarted 
        assert buffer.full is True

        # Add one more
        obs = torch.randn(n_envs, *obs_shape).to(device)
        next_obs = torch.randn(n_envs, *obs_shape).to(device)
        actions = torch.randn(n_envs, *action_shape).to(device)
        rewards = torch.randn(n_envs).to(device)
        dones = torch.zeros(n_envs).to(device)
        buffer.add(obs, next_obs, actions, rewards, dones)

        assert buffer.pos == 1
        assert buffer.full is True

    def test_replay_buffer_get_batches(self, buffer_config, device):
        """Test sampling from replay buffer."""
        buffer = ReplayBuffer(**buffer_config)

        n_envs = buffer_config["n_envs"]
        obs_shape = buffer_config["observation_space"].shape
        action_shape = buffer_config["action_space"].shape

        # Add some experiences
        for _ in range(8):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            next_obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            buffer.add(obs, next_obs, actions, rewards, dones)

        # Sample
        batch_size = 2
        samples = next(buffer.get(batch_size))

        assert samples.observations.shape == (batch_size, *obs_shape)
        assert samples.actions.shape == (batch_size, *action_shape)
        assert samples.rewards.shape == (batch_size,)
        assert samples.dones.shape == (batch_size,)

    def test_replay_buffer_dict_obs(self, dict_observation_space, simple_action_space, device):
        """Test replay buffer with dictionary observations."""
        buffer = ReplayBuffer(
            buffer_size=100,
            n_envs=4,
            observation_space=dict_observation_space,
            action_space=simple_action_space,
            device=device,
        )

        assert buffer.is_dict_obs is True

        # Add experience with dict obs
        obs = {
            "state": torch.randn(4, 8).to(device),
            "privileged": torch.randn(4, 6).to(device),
            "images": torch.randint(0, 255, (4, 3, 84, 84), dtype=torch.uint8).to(device),
        }
        next_obs = {
            "state": torch.randn(4, 8).to(device),
            "privileged": torch.randn(4, 6).to(device),
            "images": torch.randint(0, 255, (4, 3, 84, 84), dtype=torch.uint8).to(device),
        }
        action = torch.randn(4, 4).to(device)
        reward = torch.randn(4).to(device)
        done = torch.zeros(4).to(device)

        buffer.add(obs, next_obs, action, reward, done)

        # Sample
        samples = next(buffer.get(1))

        assert isinstance(samples.observations, dict)
        assert "state" in samples.observations
        assert "privileged" in samples.observations
        assert "images" in samples.observations

    def test_replay_buffer_cpu_offload(self, simple_observation_space, simple_action_space):
        """Test replay buffer with CPU offload."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        buffer = ReplayBuffer(
            buffer_size=100,
            n_envs=4,
            observation_space=simple_observation_space,
            action_space=simple_action_space,
            device=torch.device("cuda"),
            cpu_offload=True,
        )

        # Buffer data should be on CPU
        assert buffer.observations.device.type == "cpu"

        # Add data (on GPU)
        obs = torch.randn(4, *simple_observation_space.shape).cuda()
        next_obs = torch.randn(4, *simple_observation_space.shape).cuda()
        actions = torch.randn(4, *simple_action_space.shape).cuda()
        rewards = torch.randn(4).cuda()
        dones = torch.zeros(4).cuda()

        buffer.add(obs, next_obs, actions, rewards, dones)

        # Sample should return data on output device
        samples = next(buffer.get(1))
        assert samples.observations.device.type == "cuda"


class TestRolloutBuffer:
    """Test suite for RolloutBuffer."""

    @pytest.fixture
    def rollout_config(self, simple_observation_space, simple_action_space, device):
        """Basic rollout buffer configuration."""
        return {
            "buffer_size": 128,
            "n_envs": 4,
            "observation_space": simple_observation_space,
            "action_space": simple_action_space,
            "device": device,
            "gae_lambda": 0.95,
            "gamma": 0.99,
        }

    def test_rollout_buffer_initialization(self, rollout_config):
        """Test rollout buffer initialization."""
        buffer = RolloutBuffer(**rollout_config)

        assert buffer.buffer_size == 128
        assert buffer.n_envs == 4
        assert buffer.pos == 0
        assert buffer.full is False

    def test_rollout_buffer_add(self, rollout_config, device):
        """Test adding transitions to rollout buffer."""
        buffer = RolloutBuffer(**rollout_config)
        buffer.initialize_buffer()

        n_envs = rollout_config["n_envs"]
        obs_shape = rollout_config["observation_space"].shape
        action_shape = rollout_config["action_space"].shape

        obs = torch.randn(n_envs, *obs_shape).to(device)
        action = torch.randn(n_envs, *action_shape).to(device)
        reward = torch.randn(n_envs).to(device)
        done = torch.zeros(n_envs).to(device)
        value = torch.randn(n_envs).to(device)
        log_prob = torch.randn(n_envs).to(device)
        mean = torch.randn(n_envs, *action_shape).to(device)
        std = torch.randn(n_envs, *action_shape).to(device)
        action_mask = torch.randn(n_envs, *action_shape).to(device)

        buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        assert buffer.pos == 1

    def test_rollout_buffer_compute_returns_and_advantage(self, rollout_config, device):
        """Test GAE computation."""
        buffer = RolloutBuffer(**rollout_config)
        buffer.initialize_buffer()

        n_envs = rollout_config["n_envs"]
        obs_shape = rollout_config["observation_space"].shape
        action_shape = rollout_config["action_space"].shape

        # Add several transitions
        for _ in range(10):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            action = torch.randn(n_envs, *action_shape).to(device)
            reward = torch.randn(n_envs).to(device)
            done = torch.zeros(n_envs).to(device)
            value = torch.randn(n_envs).to(device)
            log_prob = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.randn(n_envs, *action_shape).to(device)
            buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        last_values = torch.randn(n_envs).to(device)
        last_dones = torch.zeros(n_envs).to(device)

        buffer.compute_returns_and_advantage(last_values, last_dones)

        assert hasattr(buffer, "advantages")
        assert hasattr(buffer, "returns")
        assert buffer.advantages.shape == (n_envs, rollout_config["buffer_size"])
        assert buffer.returns.shape == (n_envs, rollout_config["buffer_size"])

    def test_rollout_buffer_get_batches(self, rollout_config, device):
        """Test generating mini-batches from rollout buffer."""
        buffer = RolloutBuffer(**rollout_config)
        buffer.initialize_buffer()

        n_envs = rollout_config["n_envs"]
        buffer_size = rollout_config["buffer_size"]
        obs_shape = rollout_config["observation_space"].shape
        action_shape = rollout_config["action_space"].shape

        # Fill buffer
        for _ in range(buffer_size):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            action = torch.randn(n_envs, *action_shape).to(device)
            reward = torch.randn(n_envs).to(device)
            done = torch.zeros(n_envs).to(device)
            value = torch.randn(n_envs).to(device)
            log_prob = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.randn(n_envs, *action_shape).to(device)
            buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        last_values = torch.randn(n_envs).to(device)
        last_dones = torch.zeros(n_envs).to(device)
        buffer.compute_returns_and_advantage(last_values, last_dones)

        # Get batches
        batch_size = 64
        batches = list(buffer.get(batch_size))

        assert len(batches) > 0

        for batch in batches:
            assert batch.observations.shape[0] <= batch_size
            assert batch.actions.shape[0] <= batch_size
            assert batch.returns.shape[0] <= batch_size
            assert batch.advantages.shape[0] <= batch_size

    def test_rollout_buffer_reset(self, rollout_config, device):
        """Test buffer reset."""
        buffer = RolloutBuffer(**rollout_config)
        buffer.initialize_buffer()

        n_envs = rollout_config["n_envs"]
        obs_shape = rollout_config["observation_space"].shape
        action_shape = rollout_config["action_space"].shape

        # Add some data
        for _ in range(10):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            action = torch.randn(n_envs, *action_shape).to(device)
            reward = torch.randn(n_envs).to(device)
            done = torch.zeros(n_envs).to(device)
            value = torch.randn(n_envs).to(device)
            log_prob = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.randn(n_envs, *action_shape).to(device)
            buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        assert buffer.pos == 10

        # Reset
        buffer.reset()

        assert buffer.pos == 0
        assert buffer.full is False

    def test_rollout_buffer_advantage_normalization(self, rollout_config, device):
        """Test advantage normalization."""
        rollout_config["normalize_advantage"] = True
        buffer = RolloutBuffer(**rollout_config)
        buffer.initialize_buffer()

        n_envs = rollout_config["n_envs"]
        obs_shape = rollout_config["observation_space"].shape
        action_shape = rollout_config["action_space"].shape

        # Add transitions
        for _ in range(20):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            action = torch.randn(n_envs, *action_shape).to(device)
            reward = torch.randn(n_envs).to(device)
            done = torch.zeros(n_envs).to(device)
            value = torch.randn(n_envs).to(device)
            log_prob = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.randn(n_envs, *action_shape).to(device)
            buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        last_values = torch.randn(n_envs).to(device)
        last_dones = torch.zeros(n_envs).to(device)
        buffer.compute_returns_and_advantage(last_values, last_dones)

        # Advantages should be approximately normalized (mean ~0, std ~1)
        if rollout_config.get("normalize_advantage", False):
            mean_adv = buffer.advantages.mean()
            std_adv = buffer.advantages.std()
            assert abs(mean_adv) < 0.1  # Close to 0
            assert abs(std_adv - 1.0) < 0.1  # Close to 1

    def test_rollout_buffer_dict_observations(self, dict_observation_space, simple_action_space, device):
        """Test rollout buffer with dictionary observations."""
        buffer = RolloutBuffer(
            buffer_size=64,
            n_envs=2,
            observation_space=dict_observation_space,
            action_space=simple_action_space,
            device=device,
            gae_lambda=0.95,
            gamma=0.99,
        )
        buffer.initialize_buffer()

        assert buffer.is_dict_obs is True

        # Add data
        obs = {
            "state": torch.randn(2, 8).to(device),
            "privileged": torch.randn(2, 6).to(device),
            "images": torch.randint(0, 255, (2, 3, 84, 84), dtype=torch.uint8).to(device),
        }
        action = torch.randn(2, 4).to(device)
        reward = torch.randn(2).to(device)
        done = torch.zeros(2).to(device)
        value = torch.randn(2).to(device)
        log_prob = torch.randn(2).to(device)
        mean = torch.randn(2, 4).to(device)
        std = torch.randn(2, 4).to(device)
        action_mask = torch.ones(2, 4).to(device)

        buffer.add(obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob, mean=mean, std=std, action_mask=action_mask)

        assert buffer.pos == 1


class TestTransformerRolloutBuffer:
    """Test suite for TransformerRolloutBuffer."""

    @pytest.fixture
    def transformer_config(self, simple_observation_space, simple_action_space, device):
        """Basic transformer rollout buffer configuration."""
        return {
            "buffer_size": 128,
            "n_envs": 4,
            "observation_space": simple_observation_space,
            "action_space": simple_action_space,
            "device": device,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "seq_length": 4,
            "memory_length": 32,
            "num_blocks": 2,
            "embed_dim": 64,
            "num_heads": 4,
            "memorize_cache": False,
        }

    def test_transformer_buffer_initialization(self, transformer_config):
        """Test transformer buffer with memory tracking."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)

        assert buffer.buffer_size == 128
        assert buffer.n_envs == 4
        assert buffer.pos == 0
        assert buffer.full is False
        assert buffer.memory_length == 32
        assert buffer.seq_length == 4

    def test_transformer_buffer_memory_initialization(self, transformer_config, device):
        """Test that memory tensors are initialized correctly."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        # Check memory tensor shape (n_envs, buffer_size+1, num_blocks, embed_dim)
        assert buffer.memories.shape == (4, 129, 2, 64)
        assert buffer.memories.device == device

        # Check memory mask batch shape (n_envs, buffer_size+1, seq_length, memory_length)
        assert buffer.memory_mask_batch.shape == (4, 129, 4, 32)

    def test_transformer_buffer_add(self, transformer_config, device):
        """Test adding transitions to transformer buffer."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]
        obs_shape = transformer_config["observation_space"].shape
        action_shape = transformer_config["action_space"].shape

        obs = torch.randn(n_envs, *obs_shape).to(device)
        actions = torch.randn(n_envs, *action_shape).to(device)
        rewards = torch.randn(n_envs).to(device)
        dones = torch.zeros(n_envs).to(device)
        values = torch.randn(n_envs).to(device)
        log_probs = torch.randn(n_envs).to(device)
        mean = torch.randn(n_envs, *action_shape).to(device)
        std = torch.randn(n_envs, *action_shape).to(device)
        action_mask = torch.ones(n_envs, *action_shape, dtype=torch.bool).to(device)
        memory = torch.randn(n_envs, 2, 64).to(device)  # (n_envs, num_blocks, embed_dim)

        buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        assert buffer.pos == 1

    def test_transformer_buffer_memory_storage(self, transformer_config, device):
        """Test memory storage and retrieval."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]
        obs_shape = transformer_config["observation_space"].shape
        action_shape = transformer_config["action_space"].shape

        # Add multiple steps
        for step in range(10):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            values = torch.randn(n_envs).to(device)
            log_probs = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.ones(n_envs, *action_shape, dtype=torch.bool).to(device)
            memory = torch.randn(n_envs, 2, 64).to(device) * (step + 1)  # Different values per step

            buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        assert buffer.pos == 10
        # Check that memories were stored
        assert not torch.allclose(buffer.memories[:, 1], buffer.memories[:, 2])

    def test_transformer_buffer_get_last(self, transformer_config, device):
        """Test getting the last memory state."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]
        obs_shape = transformer_config["observation_space"].shape
        action_shape = transformer_config["action_space"].shape

        # Add some data
        for _ in range(5):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            values = torch.randn(n_envs).to(device)
            log_probs = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.ones(n_envs, *action_shape, dtype=torch.bool).to(device)
            memory = torch.randn(n_envs, 2, 64).to(device)

            buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        # Get last memory
        last_memory, last_mask, last_indices = buffer.get_last()

        # Check shape (n_envs, memory_length, num_blocks, embed_dim)
        assert last_memory.shape == (4, 32, 2, 64)
        assert last_mask.shape == (4, 4, 32)
        assert last_indices.shape == (4, 32)

    def test_transformer_buffer_compute_returns_and_advantage(self, transformer_config, device):
        """Test GAE computation for transformer buffer."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]
        obs_shape = transformer_config["observation_space"].shape
        action_shape = transformer_config["action_space"].shape

        # Add several transitions
        for _ in range(20):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            values = torch.randn(n_envs).to(device)
            log_probs = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.ones(n_envs, *action_shape, dtype=torch.bool).to(device)
            memory = torch.randn(n_envs, 2, 64).to(device)

            buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        last_values = torch.randn(n_envs).to(device)
        last_dones = torch.zeros(n_envs).to(device)

        buffer.compute_returns_and_advantage(last_values, last_dones)

        assert hasattr(buffer, "advantages")
        assert hasattr(buffer, "returns")
        assert buffer.advantages.shape == (n_envs, transformer_config["buffer_size"])
        assert buffer.returns.shape == (n_envs, transformer_config["buffer_size"])

    def test_transformer_buffer_restart_memory(self, transformer_config, device):
        """Test memory restart on episode end."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]

        # Add some memories
        for _ in range(5):
            obs = torch.randn(n_envs, 10).to(device)
            actions = torch.randn(n_envs, 4).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            values = torch.randn(n_envs).to(device)
            log_probs = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, 4).to(device)
            std = torch.randn(n_envs, 4).to(device)
            action_mask = torch.ones(n_envs, 4, dtype=torch.bool).to(device)
            memory = torch.randn(n_envs, 2, 64).to(device)

            buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        # Restart memory for some environments
        dones = torch.tensor([True, False, True, False]).to(device)
        buffer.restart_memory(dones, steps=10)

        # Check that memories were reset for done environments
        assert torch.allclose(buffer.memories[0, buffer.pos], torch.zeros(2, 64).to(device))
        assert not torch.allclose(buffer.memories[1, buffer.pos], torch.zeros(2, 64).to(device))

    def test_transformer_buffer_get_batches(self, transformer_config, device):
        """Test generating mini-batches from transformer buffer."""
        from algos.storage.buffers import TransformerRolloutBuffer

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        n_envs = transformer_config["n_envs"]
        buffer_size = transformer_config["buffer_size"]
        obs_shape = transformer_config["observation_space"].shape
        action_shape = transformer_config["action_space"].shape

        # Fill buffer
        for _ in range(buffer_size):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            actions = torch.randn(n_envs, *action_shape).to(device)
            rewards = torch.randn(n_envs).to(device)
            dones = torch.zeros(n_envs).to(device)
            values = torch.randn(n_envs).to(device)
            log_probs = torch.randn(n_envs).to(device)
            mean = torch.randn(n_envs, *action_shape).to(device)
            std = torch.randn(n_envs, *action_shape).to(device)
            action_mask = torch.ones(n_envs, *action_shape, dtype=torch.bool).to(device)
            memory = torch.randn(n_envs, 2, 64).to(device)

            buffer.add(obs, actions, rewards, dones, values, log_probs, mean, std, action_mask, memory)

        last_values = torch.randn(n_envs).to(device)
        last_dones = torch.zeros(n_envs).to(device)
        buffer.compute_returns_and_advantage(last_values, last_dones)

        # Get batches
        batch_size = 64
        batches = list(buffer.get(batch_size))

        assert len(batches) > 0

        for batch in batches:
            assert batch.observations.shape[0] <= batch_size
            assert batch.actions.shape[0] <= batch_size
            assert batch.returns.shape[0] <= batch_size
            assert batch.advantages.shape[0] <= batch_size
            assert hasattr(batch, "memories")
            assert hasattr(batch, "memory_mask")

    def test_transformer_buffer_cache_mode(self, transformer_config, device):
        """Test transformer buffer with cache memorization mode."""
        from algos.storage.buffers import TransformerRolloutBuffer

        # Enable cache mode
        transformer_config["memorize_cache"] = True
        transformer_config["max_tokens"] = 128
        transformer_config["max_tokens_per_step"] = 16
        transformer_config["self_attn_cache"] = True
        transformer_config["cross_attn_cache"] = False

        buffer = TransformerRolloutBuffer(**transformer_config)
        buffer.initialize_buffer()

        assert hasattr(buffer, "last_mask")
        assert hasattr(buffer, "last_memory")
        assert hasattr(buffer, "attn_cache_incremental")

        # Test that cache tensors have correct shapes
        # last_memory: (n_envs, num_blocks, attn_stores, max_tokens, num_heads, embed_dim // num_heads)
        attn_stores = 3  # num_attn_projections * (self_attn_cache + cross_attn_cache) = 3 * 1
        assert buffer.last_memory.shape == (4, 2, attn_stores, 128, 4, 16)
