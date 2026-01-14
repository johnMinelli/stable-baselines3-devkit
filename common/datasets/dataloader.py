from typing import Any, Dict, List

import numpy as np
import torch
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms
from torch.utils.data import DataLoader as TorchDataLoader

from common.utils import DictObj

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


class DataLoader(TorchDataLoader):
    """DataLoader that aggregates observation.* keys and handles tensors appropriately."""

    def __init__(
        self,
        dataset: "LeRobotDataset",
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 0,
        persistent_workers: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the data loader with a custom collate function.

        :param dataset: The dataset to load data from
        :param batch_size: Batch size for loading
        :param shuffle: Whether to shuffle the data
        :param num_workers: Number of workers for loading
        :param seed: Random seed for shuffling
        :param persistent_workers: Whether to maintain worker processes between iterations
        """
        # Set up random generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.device = device
        self.aux_obs_keys = [
            "left_pts_2d",
            "left_pts_3d",
            "right_pts_2d",
            "right_pts_3d",
            "action_is_pad",
            "left_pts_2d_is_pad",
            "right_pts_2d_is_pad",
            "state",
            "privileged",
            "image",
            "wrist_image",
            "expert_mask",
        ]

        # Initialize parent class with custom collate function
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            generator=generator,
            persistent_workers=persistent_workers and num_workers > 0,
        )

    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate function that aggregates all observation.* keys into a single obs dict and maintains a separate the action key
        """
        result = {"obs": {}, "actions": None, "infos": {}}

        for k in batch[0].keys():
            if k == "action" or k == "actions":
                result["actions"] = torch.stack([item[k] for item in batch], dim=0).to(self.device)
            elif k.startswith("observation.") or k in self.aux_obs_keys:
                # extract the key without the "observation." prefix
                obs_key = k[len("observation.") :] if k.startswith("observation.") else k
                if obs_key.startswith("images."):
                    obs_key = obs_key[len("images.") :]
                    result["obs"]["images"] = result["obs"].get("images", {})
                    if isinstance(batch[0][k], torch.Tensor):
                        result["obs"]["images"][obs_key] = torch.stack([item[k] for item in batch], dim=0).to(
                            self.device
                        )
                    elif isinstance(batch[0][k], (list, tuple, np.ndarray)):
                        raise Exception("What case is that?")
                        # result["obs"]["images"][obs_key] = torch.stack([torch.tensor(item[k]) for item in batch], dim=0).to(self.device)
                elif isinstance(batch[0][k], torch.Tensor):
                    result["obs"][obs_key] = torch.stack([item[k] for item in batch], dim=0).to(self.device)
                elif isinstance(batch[0][k], (list, tuple, np.ndarray)):  # noqa: R506
                    raise Exception("What case is that?")
                    # result["obs"][obs_key] = torch.stack([torch.tensor(item[k]) for item in batch], dim=0).to(self.device)
                else:
                    # for non-tensor types, just collect them
                    result["obs"][obs_key] = [item[k] for item in batch]
            # Handle existing observation keys
            elif (k == "obs" or k == "observation") and isinstance(batch[0][k], dict):
                # if observation is already a dict, merge its contents
                for item_idx, item in enumerate(batch):
                    for sub_key, value in item[k].items():
                        if sub_key not in result["obs"]:
                            if isinstance(value, torch.Tensor):
                                result["obs"][sub_key] = [None] * len(batch)
                            else:
                                result["obs"][sub_key] = []

                        if isinstance(value, torch.Tensor):
                            result["obs"][sub_key][item_idx] = value
                        else:
                            result["obs"][sub_key].append(value)

                # stack tensors after collecting all items
                for sub_key in result["obs"]:
                    if all(isinstance(x, torch.Tensor) for x in result["obs"][sub_key]):
                        result["obs"][sub_key] = torch.stack(result["obs"][sub_key], dim=0).to(self.device)
            # Handle NL strings
            elif k == "task" or k == "prompt":
                result["obs"]["prompt"] = [item[k] for item in batch]
            # Other fields are aggregated as infos
            else:
                if isinstance(batch[0][k], torch.Tensor):
                    result["infos"][k] = torch.stack([item[k] for item in batch], dim=0).to(self.device)
                elif isinstance(batch[0][k], (list, tuple, np.ndarray)):
                    result["infos"][k] = torch.stack([torch.tensor(item[k]) for item in batch], dim=0).to(self.device)
                else:
                    # for non-tensor types, just collect them
                    result["infos"][k] = [item[k] for item in batch]

        return result


def resolve_delta_timestamps(data_cfg):
    """Resolves delta_timestamps config key (in-place) by using `eval`.

    Doesn't do anything if delta_timestamps is not specified or has already been resolve (as evidenced by
    the data type of its values).
    """
    delta_timestamps = getattr(data_cfg, "delta_timestamps", {})
    for key in delta_timestamps:
        if isinstance(delta_timestamps[key], str):
            data_cfg.delta_timestamps[key] = eval(delta_timestamps[key])


# {
# additional transformation

#   # Apply data augmentation if training
#   if train:
#   # Simple augmentations example
#       if
#   "wrist" not in orig_key:  # Only augment non-wrist cameras
#   b, c, height, width = image.shape
#   # Apply crop and resize
#   crop_h, crop_w = int(height * 0.95), int(width * 0.95)
#   top = torch.randint(0, height - crop_h + 1, (b,), device=image.device)
#   left = torch.randint(0, width - crop_w + 1, (b,), device=image.device)
#
#   cropped_images = []
#   for i in range(b):
#       cropped = image[i:i + 1, :, top[i]:top[i] + crop_h, left[i]:left[i] + crop_w]
#   cropped = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)
#   cropped_images.append(cropped)
#   image = torch.cat(cropped_images, dim=0)
#
#   # Random rotation
#   angle = torch.empty(b, device=image.device).uniform_(-5, 5)
#   rotated_images = []
#   for i in range(b):
#       rotated = F.interpolate(F.rotate(image[i:i + 1], angle[i].item(), interpolation=F.InterpolationMode.BILINEAR),
#           size=(height, width), mode='bilinear', align_corners=False)
#   rotated_images.append(rotated)
#   image = torch.cat(rotated_images, dim=0)
#
#   # Color jitter for all cameras
#   b = image.shape[0]
#   brightness_factor = torch.empty(b, 1, 1, 1, device=image.device).uniform_(0.7, 1.3)
#   contrast_factor = torch.empty(b, 1, 1, 1, device=image.device).uniform_(0.6, 1.4)
#   image = brightness_factor * image
#   image = (image - 0.5) * contrast_factor + 0.5
#   # Clip back to [-1, 1] range
#   image = torch.clamp(image, -1.0, 1.0)
# }


def make_dataset(dataset_cfg: Dict[str, Any]) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    :param dataset_cfg: A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig
    :raises NotImplementedError: The MultiLeRobotDataset is currently deactivated
    :return: LeRobotDataset | MultiLeRobotDataset
    """
    dataset_cfg = DictObj(dataset_cfg)

    image_transforms = ImageTransforms(dataset_cfg.image_transforms) if dataset_cfg.image_transforms.enable else None

    if isinstance(dataset_cfg.dataset_repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            dataset_cfg.dataset_repo_id,
            revision=dataset_cfg.dataset_revision,
            root=dataset_cfg.dataset_root,
            force_cache_sync=True,
        )
        resolve_delta_timestamps(dataset_cfg)
        dataset = LeRobotDataset(
            dataset_cfg.dataset_repo_id,
            root=dataset_cfg.dataset_root,
            revision=getattr(dataset_cfg, "revision", None),
            episodes=getattr(dataset_cfg, "episodes", None),
            delta_timestamps=getattr(dataset_cfg, "delta_timestamps", {}),
            image_transforms=image_transforms,
            video_backend=getattr(dataset_cfg, "video_backend", None),
        )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset_repo_id,
            episodes=getattr(dataset_cfg, "episodes", None),
            delta_timestamps=getattr(dataset_cfg, "delta_timestamps", {}),
            image_transforms=image_transforms,
            video_backend=getattr(dataset_cfg, "video_backend", None),
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if getattr(dataset_cfg, "use_imagenet_stats", False):
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


# Example of creating normalization statistics
def compute_normalization_stats(dataset, observation_keys=None, action_keys=None):
    """
    Compute normalization statistics from a dataset.

    :param dataset: Dataset to compute stats from
    :param observation_keys: Keys for observations to compute stats for
    :param action_keys: Keys for actions to compute stats for
    :return: Dictionary of normalization statistics
    """
    if observation_keys is None:
        observation_keys = ["state"]

    if action_keys is None:
        action_keys = ["actions"]

    # Initialize statistics
    stats = {}

    # Collect all data first
    obs_data = {k: [] for k in observation_keys}
    action_data = {k: [] for k in action_keys}

    for i in range(len(dataset)):
        sample = dataset[i]

        for k in observation_keys:
            if k in sample:
                obs_data[k].append(sample[k])

        for k in action_keys:
            if k in sample:
                action_data[k].append(sample[k])

    # Compute statistics for observations
    for k in observation_keys:
        if not obs_data[k]:
            continue

        data = np.concatenate(obs_data[k], axis=0)
        stats[k] = {
            "mean": torch.tensor(np.mean(data, axis=0), dtype=torch.float32),
            "std": torch.tensor(np.std(data, axis=0), dtype=torch.float32),
            "min": torch.tensor(np.min(data, axis=0), dtype=torch.float32),
            "max": torch.tensor(np.max(data, axis=0), dtype=torch.float32),
        }

    # Compute statistics for actions
    for k in action_keys:
        if not action_data[k]:
            continue

        data = np.concatenate(action_data[k], axis=0)
        stats[k] = {
            "mean": torch.tensor(np.mean(data, axis=0), dtype=torch.float32),
            "std": torch.tensor(np.std(data, axis=0), dtype=torch.float32),
            "min": torch.tensor(np.min(data, axis=0), dtype=torch.float32),
            "max": torch.tensor(np.max(data, axis=0), dtype=torch.float32),
        }

    return stats
