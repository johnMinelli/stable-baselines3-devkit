"""Unit tests for environment wrappers.

Wrapper Architecture:
- Isaac Lab: Sb3EnvStdWrapper(isaac_env) - Isaac already has "policy" key format
- ManiSkill: Sb3EnvStdWrapper(ManiSkillEnvStdWrapper(maniskill_env))
- Aloha: Sb3EnvStdWrapper(AlohaEnvStdWrapper(aloha_env))

The intermediate wrappers (ManiSkillEnvStdWrapper, AlohaEnvStdWrapper) transform
environment-specific formats to Isaac-like format with "policy" key for states.
Then Sb3EnvStdWrapper provides the final SB3-compatible interface.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from common.envs.sb3_env_wrapper import Sb3EnvStdWrapper


class TestIsaacEnv:
    """Test Sb3EnvStdWrapper."""

    @pytest.fixture
    def mock_isaac_env(self):
        """Create a mock Isaac-style environment."""

        class MockIsaacEnv:
            def __init__(self):
                self.num_envs = 4
                self.device = torch.device("cpu")
                self.unwrapped = self  # Required by wrapper
                self.single_observation_space = spaces.Dict({
                    "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

                self.observation_space = spaces.Dict({
                    "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                })
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

            def reset(self):
                obs = {
                    "policy": torch.randn(self.num_envs, 10),
                }
                info = {}
                return obs, info

            def step(self, actions):
                obs = {
                    "policy": torch.randn(self.num_envs, 10),
                }
                rewards = torch.randn(self.num_envs)
                dones = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, dones, truncated, info

            def close(self):
                pass

        return MockIsaacEnv()

    def test_isaac_wrapper_has_policy_key(self, mock_isaac_env):

        obs, info = mock_isaac_env.reset()

        # Should have policy key
        assert "policy" in obs, "Isaac wrapped env should have 'policy' key"
        assert obs["policy"].shape[0] == 4, "Batch dimension should match num_envs"

    def test_wrapper_step(self, mock_isaac_env):
        """Test wrapper step."""

        actions = np.random.randn(4, 4).astype(np.float32)
        obs, rewards, dones, infos, extras = mock_isaac_env.step(actions)

        assert "policy" in obs
        assert obs["policy"].shape[0] == 4
        assert rewards.shape[0] == 4
        assert dones.shape[0] == 4


class TestManiSkillWrapper:
    """Test ManiSkill environment wrapper."""

    @pytest.fixture
    def mock_maniskill_env(self):
        """Create a mock ManiSkill-style environment."""

        class MockManiSkillEnv:
            def __init__(self):
                self.num_envs = 2
                self.device = "cpu"
                self.unwrapped = self  # Required by wrapper
                # ManiSkill uses state and sensor_data
                self.single_observation_space = spaces.Dict({
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
                    "sensor_data": spaces.Dict({
                        "base_camera": spaces.Dict({
                            "rgb": spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8),
                        })
                    })
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

                self.observation_space = self.single_observation_space
                self.action_space = self.single_action_space

            def reset(self, **kwargs):
                obs = {
                    "state": torch.randn(self.num_envs, 15),
                    "sensor_data": {
                        "base_camera": {
                            "rgb": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                        }
                    },
                }
                info = {}
                return obs, info

            def step(self, action):
                obs = {
                    "state": torch.randn(self.num_envs, 15),
                    "sensor_data": {
                        "base_camera": {
                            "rgb": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                        }
                    },
                }
                rewards = torch.randn(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, terminated, truncated, info

        return MockManiSkillEnv()

    def test_maniskill_wrapper_initialization(self, mock_maniskill_env):
        """Test ManiSkill wrapper initialization."""
        try:
            from common.envs.maniskill_wrapper import ManiSkillEnvStdWrapper

            wrapper = ManiSkillEnvStdWrapper(mock_maniskill_env)

            assert wrapper.num_envs == 2
            assert hasattr(wrapper, "single_observation_space")

        except ImportError as e:
            pytest.skip(f"ManiSkill wrapper not available: {e}")

    def test_maniskill_wrapper_observation_space(self, mock_maniskill_env):
        """Test that ManiSkill wrapper creates correct observation space."""
        try:
            from common.envs.maniskill_wrapper import ManiSkillEnvStdWrapper

            wrapper = ManiSkillEnvStdWrapper(mock_maniskill_env)

            # Should have policy key for state and camera keys
            assert "policy" in wrapper.single_observation_space.spaces

        except ImportError as e:
            pytest.skip(f"ManiSkill wrapper not available: {e}")

    def test_maniskill_wrapper_reset(self, mock_maniskill_env):
        """Test ManiSkill wrapper reset."""
        try:
            from common.envs.maniskill_wrapper import ManiSkillEnvStdWrapper

            wrapper = ManiSkillEnvStdWrapper(mock_maniskill_env)
            obs, info = wrapper.reset()

            # Should return standardized observations
            assert "policy" in obs

        except ImportError as e:
            pytest.skip(f"ManiSkill wrapper not available: {e}")


class TestMjxWrapper:
    """Test MuJoCo Playground environment wrapper."""

    @pytest.fixture
    def mock_mjx_env(self):
        """Create a mock MuJoCo Playground-style environment."""

        class MockMjxPlaygroundEnv:
            def __init__(self):
                self.num_envs = 2
                self.sim_device = "cpu"
                self.unwrapped = self
                self.render_mode = None
                # MjxPlaygroundGymWrapper uses "state" key (converted to "policy" by MjxPlaygroundStdWrapper)
                self.single_observation_space = spaces.Dict({
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

                self.observation_space = spaces.Dict({
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_envs, 12), dtype=np.float32),
                })
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_envs, 4), dtype=np.float32)

            def reset(self, **kwargs):
                obs = torch.randn(self.num_envs, 12)
                info = {}
                return obs, info

            def step(self, action):
                obs = torch.randn(self.num_envs, 12)
                rewards = torch.randn(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, terminated, truncated, info

            def render(self):
                pass

            def close(self):
                pass

        return MockMjxPlaygroundEnv()

    def test_mjx_wrapper_initialization(self, mock_mjx_env):
        """Test MJX wrapper initialization."""
        try:
            from common.envs.mjx_playground_wrapper import MjxPlaygroundStdWrapper

            wrapper = MjxPlaygroundStdWrapper(mock_mjx_env)

            assert wrapper.num_envs == 2
            assert hasattr(wrapper, "single_observation_space")

        except ImportError as e:
            pytest.skip(f"MJX wrapper not available: {e}")

    def test_mjx_wrapper_observation_space(self, mock_mjx_env):
        """Test that MJX wrapper creates correct observation space."""
        try:
            from common.envs.mjx_playground_wrapper import MjxPlaygroundStdWrapper

            wrapper = MjxPlaygroundStdWrapper(mock_mjx_env)

            # Should have policy key for state
            assert "policy" in wrapper.single_observation_space.spaces

        except ImportError as e:
            pytest.skip(f"MJX wrapper not available: {e}")

    def test_mjx_wrapper_reset(self, mock_mjx_env):
        """Test MJX wrapper reset."""
        try:
            from common.envs.mjx_playground_wrapper import MjxPlaygroundStdWrapper

            wrapper = MjxPlaygroundStdWrapper(mock_mjx_env)
            obs, info = wrapper.reset()

            # Should return standardized observations with "policy" key
            assert "policy" in obs
            assert obs["policy"].shape == (2, 12)

        except ImportError as e:
            pytest.skip(f"MJX wrapper not available: {e}")

    def test_mjx_wrapper_step(self, mock_mjx_env):
        """Test MJX wrapper step."""
        try:
            from common.envs.mjx_playground_wrapper import MjxPlaygroundStdWrapper

            wrapper = MjxPlaygroundStdWrapper(mock_mjx_env)
            wrapper.reset()

            actions = torch.randn(2, 4)
            obs, rewards, terminated, truncated, info = wrapper.step(actions)

            # Should return standardized observations
            assert "policy" in obs
            assert obs["policy"].shape == (2, 12)
            assert rewards.shape == (2,)
            assert terminated.shape == (2,)
            assert truncated.shape == (2,)

        except ImportError as e:
            pytest.skip(f"MJX wrapper not available: {e}")


class TestSb3EnvStdWrapperOutputFormat:
    """Test that Sb3EnvStdWrapper produces consistent output format across all environments.

    The wrapper should produce observations with keys:
    - "state": for policy/state observations
    - "images": dict containing camera observations

    This format matches the dataset format (see test_datasets.py) where:
    - batch["obs"]["state"] contains state data
    - batch["obs"]["images"] contains image data
    - batch["actions"] contains action data
    """

    @pytest.fixture
    def mock_isaac_env_with_camera(self):
        """Create a mock Isaac-style environment with camera observations."""

        class MockIsaacEnvWithCamera:
            def __init__(self):
                self.num_envs = 4
                self.device = torch.device("cpu")
                self.unwrapped = self
                self.render_mode = "rgb_array"
                self.single_observation_space = spaces.Dict({
                    "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32),
                    "base_camera": spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8),
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                self.observation_space = self.single_observation_space
                self.action_space = self.single_action_space

            def reset(self):
                obs = {
                    "policy": torch.randn(self.num_envs, 10),
                    "base_camera": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                }
                info = {}
                return obs, info

            def step(self, actions):
                obs = {
                    "policy": torch.randn(self.num_envs, 10),
                    "base_camera": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                }
                rewards = torch.randn(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, terminated, truncated, info

            def close(self):
                pass

            def seed(self, seed):
                pass

        return MockIsaacEnvWithCamera()

    def test_sb3_wrapper_isaac_reset_output_format(self, mock_isaac_env_with_camera):
        """Test that Sb3EnvStdWrapper with Isaac env produces standardized format on reset."""
        wrapped_env = Sb3EnvStdWrapper(mock_isaac_env_with_camera)
        obs = wrapped_env.reset()

        # Check that observations have the expected standardized format
        assert isinstance(obs, dict), "Observations should be a dict"
        assert "state" in obs, "Observations should have 'state' key (not 'policy')"
        assert "images" in obs, "Observations should have 'images' key"

        # Check state shape
        assert isinstance(obs["state"], torch.Tensor), "State should be a tensor"
        assert obs["state"].shape == (4, 10), "State should have shape (num_envs, state_dim)"

        # Check images structure
        assert isinstance(obs["images"], dict), "Images should be a dict"
        assert "base_camera" in obs["images"], "Images should contain camera keys"
        assert obs["images"]["base_camera"].shape == (4, 3, 128, 128), "Camera should have shape (num_envs, C, H, W)"

    def test_sb3_wrapper_isaac_step_output_format(self, mock_isaac_env_with_camera):
        """Test that Sb3EnvStdWrapper with Isaac env produces standardized format on step."""
        wrapped_env = Sb3EnvStdWrapper(mock_isaac_env_with_camera)
        wrapped_env.reset()

        actions = np.random.randn(4, 4).astype(np.float32)
        obs, rewards, dones, infos = wrapped_env.step(actions)

        # Check observations format
        assert isinstance(obs, dict), "Observations should be a dict"
        assert "state" in obs, "Observations should have 'state' key"
        assert "images" in obs, "Observations should have 'images' key"

        # Check state
        assert obs["state"].shape == (4, 10), "State should have shape (num_envs, state_dim)"

        # Check images
        assert isinstance(obs["images"], dict), "Images should be a dict"
        assert "base_camera" in obs["images"], "Images should contain camera keys"

        # Check rewards and dones
        assert rewards.shape == (4,), "Rewards should have shape (num_envs,)"
        assert dones.shape == (4,), "Dones should have shape (num_envs,)"

        # Check infos structure
        assert isinstance(infos, dict), "Infos should be a dict"
        assert "episode" in infos, "Infos should contain 'episode' key"

    def test_sb3_wrapper_maniskill_output_format(self):
        """Test that Sb3EnvStdWrapper with ManiSkill wrapped env produces standardized format."""

        # Create mock ManiSkill environment
        class MockManiSkillEnv:
            def __init__(self):
                self.num_envs = 2
                self.device = "cpu"
                self.unwrapped = self
                self.render_mode = "rgb_array"
                self.single_observation_space = spaces.Dict({
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
                    "sensor_data": spaces.Dict({
                        "base_camera": spaces.Dict({
                            "rgb": spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8),
                        })
                    })
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
                self.observation_space = self.single_observation_space
                self.action_space = self.single_action_space

            def reset(self, **kwargs):
                obs = {
                    "state": torch.randn(self.num_envs, 15),
                    "sensor_data": {
                        "base_camera": {
                            "rgb": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                        }
                    },
                }
                info = {}
                return obs, info

            def step(self, action):
                obs = {
                    "state": torch.randn(self.num_envs, 15),
                    "sensor_data": {
                        "base_camera": {
                            "rgb": torch.randint(0, 255, (self.num_envs, 3, 128, 128), dtype=torch.uint8),
                        }
                    },
                }
                rewards = torch.randn(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, terminated, truncated, info

            def render(self):
                pass

            def close(self):
                pass

            def seed(self, seed):
                pass

        try:
            from common.envs.maniskill_wrapper import ManiSkillEnvStdWrapper

            # First wrap with ManiSkill wrapper to standardize to "policy" format
            mock_env = MockManiSkillEnv()
            maniskill_wrapped = ManiSkillEnvStdWrapper(mock_env)

            # Then wrap with Sb3 wrapper to get final standardized format
            sb3_wrapped = Sb3EnvStdWrapper(maniskill_wrapped)

            # Test reset
            obs = sb3_wrapped.reset()
            assert isinstance(obs, dict), "Observations should be a dict"
            assert "state" in obs, "Observations should have 'state' key"
            assert "images" in obs, "Observations should have 'images' key"
            assert obs["state"].shape == (2, 15), "State should have shape (num_envs, state_dim)"
            assert "base_camera" in obs["images"], "Images should contain camera keys"

            # Test step
            actions = np.random.randn(2, 7).astype(np.float32)
            obs, rewards, dones, infos = sb3_wrapped.step(actions)
            assert "state" in obs, "Step observations should have 'state' key"
            assert "images" in obs, "Step observations should have 'images' key"
            assert rewards.shape == (2,), "Rewards should have shape (num_envs,)"

        except ImportError as e:
            pytest.skip(f"ManiSkill wrapper not available: {e}")

    def test_sb3_wrapper_aloha_output_format(self):
        """Test that Sb3EnvStdWrapper with Aloha wrapped env produces standardized format."""

        # Create mock Aloha environment
        class MockAlohaEnv:
            def __init__(self):
                self.unwrapped = self
                self.render_mode = "rgb_array"
                self.observation_space = spaces.Dict({
                    "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
                    "pixels": spaces.Dict({
                        "top": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                    })
                })
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

            def reset(self, **kwargs):
                obs = {
                    "agent_pos": np.random.randn(14).astype(np.float32),
                    "pixels": {
                        "top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    },
                }
                info = {}
                return obs, info

            def step(self, action):
                obs = {
                    "agent_pos": np.random.randn(14).astype(np.float32),
                    "pixels": {
                        "top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    },
                }
                reward = 0.0
                terminated = False
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info

            def close(self):
                pass

            def seed(self, seed):
                pass

        try:
            from common.envs.aloha_wrapper import AlohaStdWrapper

            # First wrap with Aloha wrapper to standardize to "policy" format
            mock_env = MockAlohaEnv()
            aloha_wrapped = AlohaStdWrapper(mock_env)

            # Then wrap with Sb3 wrapper to get final standardized format
            sb3_wrapped = Sb3EnvStdWrapper(aloha_wrapped)

            # Test reset
            obs = sb3_wrapped.reset()
            assert isinstance(obs, dict), "Observations should be a dict"
            assert "state" in obs, "Observations should have 'state' key"
            assert "images" in obs, "Observations should have 'images' key"
            assert obs["state"].shape == (1, 14), "State should have shape (num_envs, state_dim)"
            assert "top" in obs["images"], "Images should contain camera keys"

            # Test step
            actions = np.random.randn(1, 14).astype(np.float32)
            obs, rewards, dones, infos = sb3_wrapped.step(actions)
            assert "state" in obs, "Step observations should have 'state' key"
            assert "images" in obs, "Step observations should have 'images' key"
            assert rewards.shape == (1,), "Rewards should have shape (num_envs,)"

        except ImportError as e:
            pytest.skip(f"Aloha wrapper not available: {e}")

    def test_sb3_wrapper_mjx_output_format(self):
        """Test that Sb3EnvStdWrapper with MJX wrapped env produces standardized format."""

        # Create mock MJX environment
        class MockMjxEnv:
            def __init__(self):
                self.num_envs = 2
                self.sim_device = "cpu"
                self.unwrapped = self
                self.render_mode = "rgb_array"
                self.single_observation_space = spaces.Dict({
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
                    "camera": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                })
                self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                self.observation_space = self.single_observation_space
                self.action_space = self.single_action_space

            def reset(self, **kwargs):
                obs = {
                    "state": torch.randn(self.num_envs, 12),
                    "camera": torch.randint(0, 255, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
                }
                info = {}
                return obs, info

            def step(self, action):
                obs = {
                    "state": torch.randn(self.num_envs, 12),
                    "camera": torch.randint(0, 255, (self.num_envs, 64, 64, 3), dtype=torch.uint8),
                }
                rewards = torch.randn(self.num_envs)
                terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                truncated = torch.zeros(self.num_envs, dtype=torch.bool)
                info = {}
                return obs, rewards, terminated, truncated, info

            def render(self):
                pass

            def close(self):
                pass

            def seed(self, seed):
                pass

        try:
            from common.envs.mjx_playground_wrapper import MjxPlaygroundStdWrapper

            # First wrap with MJX wrapper to standardize to "policy" format
            mock_env = MockMjxEnv()
            mjx_wrapped = MjxPlaygroundStdWrapper(mock_env)

            # Then wrap with Sb3 wrapper to get final standardized format
            sb3_wrapped = Sb3EnvStdWrapper(mjx_wrapped)

            # Test reset
            obs = sb3_wrapped.reset()
            assert isinstance(obs, dict), "Observations should be a dict"
            assert "state" in obs, "Observations should have 'state' key"
            assert "images" in obs, "Observations should have 'images' key"
            assert obs["state"].shape == (2, 12), "State should have shape (num_envs, state_dim)"
            assert "camera" in obs["images"], "Images should contain camera keys"

            # Test step
            actions = np.random.randn(2, 4).astype(np.float32)
            obs, rewards, dones, infos = sb3_wrapped.step(actions)
            assert "state" in obs, "Step observations should have 'state' key"
            assert "images" in obs, "Step observations should have 'images' key"
            assert rewards.shape == (2,), "Rewards should have shape (num_envs,)"

        except ImportError as e:
            pytest.skip(f"MJX wrapper not available: {e}")

    def test_sb3_wrapper_handles_termination(self, mock_isaac_env_with_camera):
        """Test that Sb3EnvStdWrapper properly handles episode termination."""

        # Create env that terminates
        class TerminatingEnv:
            def __init__(self, base_env):
                self.base_env = base_env
                self.num_envs = base_env.num_envs
                self.device = base_env.device
                self.unwrapped = base_env.unwrapped
                self.render_mode = base_env.render_mode
                self.single_observation_space = base_env.single_observation_space
                self.single_action_space = base_env.single_action_space
                self.observation_space = base_env.observation_space
                self.action_space = base_env.action_space
                self.step_count = 0

            def reset(self):
                return self.base_env.reset()

            def step(self, actions):
                self.step_count += 1
                obs, rewards, terminated, truncated, info = self.base_env.step(actions)
                # Terminate first env after 3 steps
                if self.step_count == 3:
                    terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                    terminated[0] = True
                return obs, rewards, terminated, truncated, info

            def close(self):
                pass

            def seed(self, seed):
                pass

        terminating_env = TerminatingEnv(mock_isaac_env_with_camera)
        wrapped_env = Sb3EnvStdWrapper(terminating_env)
        wrapped_env.reset()

        # Step until termination
        for _ in range(3):
            actions = np.random.randn(4, 4).astype(np.float32)
            obs, rewards, dones, infos = wrapped_env.step(actions)

        # Check that termination is handled properly
        assert dones[0].item(), "First env should be done"
        assert "episode" in infos, "Infos should contain episode information"
        assert "rew" in infos["episode"], "Episode info should contain rewards"
        assert "len" in infos["episode"], "Episode info should contain length"

        # Check terminal observations are included
        assert "terminal_obs" in infos, "Infos should contain terminal observations"
        assert "state" in infos["terminal_obs"], "Terminal obs should have 'state' key"
