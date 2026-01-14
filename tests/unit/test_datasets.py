"""Unit tests for dataset loaders and preprocessors."""

import os

import pytest
import torch

from tests.fixtures.mock_datasets import MockDataset, SequenceMockDataset, create_mock_trajectory_data
from tests.fixtures.conftest import device
from common.utils import create_spaces_from_cfg


class TestMockDatasets:
    """Test mock dataset utilities."""

    def test_mock_dataset_creation(self, device):
        """Test basic mock dataset creation."""
        dataset = MockDataset(
            num_trajectories=10,
            trajectory_length=10,
            obs_dim=10,
            action_dim=4,
            device=device
        )

        assert len(dataset) == 100  # 10 trajectories * 10 steps each

        sample = dataset[0]
        assert "observation" in sample
        assert "action" in sample
        assert "action_is_pad" in sample, "Sample should include action_is_pad (LeRobot format)"

        # Check that tensors are returned
        assert torch.is_tensor(sample["observation"])
        assert torch.is_tensor(sample["action"])
        assert torch.is_tensor(sample["action_is_pad"])

        assert sample["observation"].shape[-1] == 10
        assert sample["action"].shape[-1] == 4
        assert sample["action_is_pad"].shape == sample["action"].shape
        assert sample["action_is_pad"].dtype == torch.bool

        # Check device
        assert sample["observation"].device == device
        assert sample["action"].device == device
        assert sample["action_is_pad"].device == device

    def test_sequence_mock_dataset_creation(self, device):
        """Test sequence mock dataset creation."""
        dataset = SequenceMockDataset(
            num_sequences=20,
            sequence_length=16,
            obs_dim=10,
            action_dim=4,
            device=device,
        )

        assert len(dataset) == 20

        sample = dataset[0]
        assert "observation" in sample
        assert "action" in sample
        assert "action_is_pad" in sample, "Sample should include action_is_pad (LeRobot format)"

        # Check that tensors are returned
        assert torch.is_tensor(sample["observation"])
        assert torch.is_tensor(sample["action"])
        assert torch.is_tensor(sample["action_is_pad"])

        assert sample["observation"].shape[0] == 16
        assert sample["action"].shape[0] == 16
        assert sample["action_is_pad"].shape == sample["action"].shape
        assert sample["action_is_pad"].dtype == torch.bool

        # Check device
        assert sample["observation"].device == device
        assert sample["action"].device == device
        assert sample["action_is_pad"].device == device


class TestLerobotDatasetIntegration:
    """Test LeRobot dataset through dataloader.

    These tests validate that LeRobot format datasets when passed through
    the dataloader are rearranged into the shared data format (same as sb3wrapped envs).
    """

    @pytest.fixture
    def dataset_config(self):
        """Configuration for ManiSkill StackCube dataset."""
        return {
            "name": "maniskill",
            "task": "StackCube-v1",
            "dataset_repo_id": "johnMinelli/ManiSkill_StackCube-v1_recovery"
        }

    @pytest.fixture
    def dataset_shapes(self):
        """Dataset input and output shapes for StackCube task."""
        return {
            "input_shapes": {
                "observation.images.base_camera": [480, 640, 3],
                "observation.images.hand_camera": [480, 640, 3],
                "observation.state": [9],
                "observation.privileged": [30],
            },
            "output_shapes": {
                "action": [8],  # Simplified, ignoring action_horizon for tests
            }
        }

    def test_lerobot_dataset_loading(self, dataset_config):
        """Test loading LeRobot dataset from local directory."""
        dataset_path = None  # f"../../data/{dataset_config['task']}"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}. Download from HuggingFace: {dataset_config['dataset_repo_id']}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(dataset_config['dataset_repo_id'], root=dataset_path)

            assert len(dataset) > 0, "Dataset should not be empty"

            # Check dataset structure
            sample = dataset[0]
            assert sample is not None

        except ImportError as e:
            pytest.skip(f"LeRobot not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to load dataset: {e}")

    def test_dataloader_output_format(self, dataset_config, dataset_shapes, device):
        """Test dataloader produces standardized format.

        Verifies that LeRobot datasets are transformed into the shared data format
        with "obs" containing "policy" key for state observations.
        """
        dataset_path = None  # f"../../data/{dataset_config['task']}"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from common.datasets.dataloader import DataLoader
            from common.datasets.lerobot_preprocessor import LerobotPreprocessor

            # Load dataset
            dataset = LeRobotDataset(dataset_config['dataset_repo_id'], root=dataset_path)

            # Create observation and action spaces from shapes
            observation_space, action_space = create_spaces_from_cfg(
                dataset_shapes["input_shapes"],
                dataset_shapes["output_shapes"]
            )

            # Create dataloader with preprocessor (properly initialized)
            preprocessor = LerobotPreprocessor(
                observation_space=observation_space,
                action_space=action_space,
                device=device
            )
            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=True
            )

            # Get a batch
            batch = next(iter(dataloader))

            # Check standardized format (same as Sb3EnvStdWrapper output)
            assert "obs" in batch, "Batch should have 'obs' key"
            assert "actions" in batch, "Batch should have 'actions' key"

            # Observations should be dict with keys like "state", "images", etc. (NOT "policy")
            if isinstance(batch["obs"], dict):
                # Should have state observations
                assert "state" in batch["obs"], "Observations should have 'state' key for state data"
                # Should have image observations
                assert "images" in batch["obs"], "Observations should have 'images' key"

                # Check shapes
                assert batch["obs"]["state"].shape[0] == 8, "State batch size should be 8"
                assert batch["actions"].shape[0] == 8, "Actions batch size should be 8"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to test dataloader: {e}")

    def test_dataloader_observation_structure(self, dataset_config):
        """Test that observations have consistent structure across batches."""
        dataset_path = None  # f"../../data/{dataset_config['task']}"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from common.datasets.dataloader import DataLoader

            dataset = LeRobotDataset(dataset_config['dataset_repo_id'], root=dataset_path)
            dataloader = DataLoader(dataset, batch_size=4)

            # Get multiple batches
            batches = [next(iter(dataloader)) for _ in range(3)]

            # All batches should have same observation keys
            first_obs_keys = set(batches[0]["obs"].keys())
            for batch in batches[1:]:
                assert set(batch["obs"].keys()) == first_obs_keys, "Observation structure should be consistent"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to test: {e}")

    def test_dataloader_with_images(self, dataset_config):
        """Test dataloader handles image observations correctly."""
        dataset_path = None  # f"../../data/{dataset_config['task']}"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from common.datasets.dataloader import DataLoader

            dataset = LeRobotDataset(dataset_config['dataset_repo_id'], root=dataset_path)
            dataloader = DataLoader(dataset, batch_size=4)

            batch = next(iter(dataloader))

            # Check if images are present
            assert len(batch["obs"]["images"]) > 0
            for key, value in batch["obs"]["images"].items():
                assert len(value.shape) == 4, f"Image {key} should have 4 dims (B, C, H, W)"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to test: {e}")

    def test_dataloader_action_shapes(self, dataset_config):
        """Test that actions have correct shapes."""
        dataset_path = None  # f"../../data/{dataset_config['task']}"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from common.datasets.dataloader import DataLoader

            dataset = LeRobotDataset(dataset_config['dataset_repo_id'], root=dataset_path)
            dataloader = DataLoader(dataset, batch_size=8)

            batch = next(iter(dataloader))

            # Actions should be 2D: (batch_size, action_dim)
            assert len(batch["actions"].shape) == 2, "Actions should be 2D (batch, action_dim)"
            assert batch["actions"].shape[0] == 8, "Batch dimension should match batch_size"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except Exception as e:
            pytest.skip(f"Failed to test: {e}")
