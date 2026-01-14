"""Mock datasets for testing supervised learning pipelines."""

import numpy as np
import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    """
    Mock dataset for testing SL agents.

    Generates random trajectory data matching the structure of real datasets.
    Returns tensors in shared data space format (like wrapped environments).
    """

    def __init__(
        self,
        num_trajectories=10,
        trajectory_length=50,
        obs_dim=10,
        action_dim=4,
        include_images=False,
        image_size=84,
        obs_type="flat",
        device="cpu",
    ):
        """
        Initialize mock dataset.

        Args:
            num_trajectories: Number of trajectories in the dataset
            trajectory_length: Length of each trajectory
            obs_dim: Observation dimension (for flat observations)
            action_dim: Action dimension
            include_images: Whether to include image observations
            image_size: Size of square images (if include_images=True)
            obs_type: "flat" or "dict" observations
            device: Device to place tensors on (e.g., "cpu", "cuda")
        """
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.include_images = include_images
        self.image_size = image_size
        self.obs_type = obs_type
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-generate data
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate random trajectory data."""
        data = []

        for traj_idx in range(self.num_trajectories):
            trajectory = {}

            # Generate observations
            if self.obs_type == "flat":
                trajectory["observations"] = np.random.randn(
                    self.trajectory_length, self.obs_dim
                ).astype(np.float32)
            elif self.obs_type == "dict":
                trajectory["observations"] = {
                    "state": np.random.randn(self.trajectory_length, self.obs_dim // 2).astype(np.float32),
                    "privileged": np.random.randn(self.trajectory_length, self.obs_dim // 2).astype(np.float32),
                }
                if self.include_images:
                    trajectory["observations"]["image"] = np.random.randint(
                        0, 255,
                        size=(self.trajectory_length, 3, self.image_size, self.image_size),
                        dtype=np.uint8
                    )

            # Generate actions
            trajectory["actions"] = np.random.randn(
                self.trajectory_length, self.action_dim
            ).astype(np.float32)

            # Generate rewards
            trajectory["rewards"] = np.random.randn(self.trajectory_length).astype(np.float32)

            # Generate dones (last step is done)
            trajectory["dones"] = np.zeros(self.trajectory_length, dtype=bool)
            trajectory["dones"][-1] = True

            data.append(trajectory)

        return data

    def __len__(self):
        """Return total number of steps across all trajectories."""
        return self.num_trajectories * self.trajectory_length

    def __getitem__(self, idx):
        """
        Get a single transition as tensors in shared data space format.

        Args:
            idx: Global step index

        Returns:
            Dictionary containing observation (dict of tensors), action (tensor),
            action_is_pad (tensor), reward, done
        """
        traj_idx = idx // self.trajectory_length
        step_idx = idx % self.trajectory_length

        trajectory = self.data[traj_idx]

        item = {}

        # Convert observations to tensors in shared data space format
        if self.obs_type == "flat":
            item["observation"] = torch.from_numpy(trajectory["observations"][step_idx]).to(self.device)
        else:
            # Dict observations - convert each key to tensor
            item["observation"] = {
                k: torch.from_numpy(v[step_idx]).to(self.device)
                for k, v in trajectory["observations"].items()
            }

        # Convert action to tensor
        item["action"] = torch.from_numpy(trajectory["actions"][step_idx]).to(self.device)

        # Add action_is_pad tensor (mimics LeRobot dataset format)
        # All True by default (no padding in mock data)
        item["action_is_pad"] = torch.ones_like(item["action"], dtype=torch.bool, device=self.device)

        # Rewards and dones can remain as scalars
        item["reward"] = trajectory["rewards"][step_idx]
        item["done"] = trajectory["dones"][step_idx]

        return item


def create_mock_trajectory_data(
    num_episodes=5,
    episode_length=50,
    obs_shape=(10,),
    action_shape=(4,),
    as_dict=False
):
    """
    Create mock trajectory data for testing.

    Args:
        num_episodes: Number of episodes
        episode_length: Steps per episode
        obs_shape: Shape of observations
        action_shape: Shape of actions
        as_dict: Return observations as dict

    Returns:
        Dictionary with trajectory data
    """
    total_steps = num_episodes * episode_length

    if as_dict:
        observations = {
            "state": np.random.randn(total_steps, *obs_shape).astype(np.float32),
            "image": np.random.randint(0, 255, size=(total_steps, 3, 84, 84), dtype=np.uint8),
        }
    else:
        observations = np.random.randn(total_steps, *obs_shape).astype(np.float32)

    actions = np.random.randn(total_steps, *action_shape).astype(np.float32)
    rewards = np.random.randn(total_steps).astype(np.float32)

    # Create dones array (end of each episode)
    dones = np.zeros(total_steps, dtype=bool)
    for i in range(num_episodes):
        dones[(i + 1) * episode_length - 1] = True

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "episode_starts": dones,
    }


class SequenceMockDataset(Dataset):
    """
    Mock dataset that returns sequences for recurrent models.

    Used for testing LSTM/Transformer policies with SL.
    Returns tensors in shared data space format.
    """

    def __init__(
        self,
        num_sequences=20,
        sequence_length=16,
        obs_dim=10,
        action_dim=4,
        device="cpu",
    ):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-generate sequences
        self.observations = np.random.randn(
            num_sequences, sequence_length, obs_dim
        ).astype(np.float32)
        self.actions = np.random.randn(
            num_sequences, sequence_length, action_dim
        ).astype(np.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Return sequence as tensors on specified device (mimics LeRobot format)."""
        action_tensor = torch.from_numpy(self.actions[idx]).to(self.device)
        return {
            "observation": torch.from_numpy(self.observations[idx]).to(self.device),
            "action": action_tensor,
            # Add action_is_pad tensor (mimics LeRobot dataset format)
            "action_is_pad": torch.ones_like(action_tensor, dtype=torch.bool, device=self.device),
        }