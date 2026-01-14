import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import constant_fn
from torch import Tensor, nn


def pad_to_dim(x: torch.Tensor, target_dim: int, axis: int = -1, padding_side: str = "right") -> torch.Tensor:
    """
    Pad a tensor to the target dimension with zeros along the specified axis.

    Args:
        x: Input tensor
        target_dim: Target dimension size
        axis: Axis along which to pad
        padding_side: Which side to add padding ('left' or 'right')
    """
    padded_x = x

    if axis < 0:
        axis = x.dim() + axis

    current_dim = x.shape[axis]
    if current_dim < target_dim:
        # Init padding shape with zeros
        pad = [0] * (2 * x.dim())
        # Calculate padding index based on axis
        pad_idx = 2 * (x.dim() - 1 - axis)

        # Apply padding to left or right side based on padding_side parameter
        if padding_side.lower() == "left":
            pad[pad_idx] = target_dim - current_dim  # Left padding
        else:  # Default to right padding
            pad[pad_idx + 1] = target_dim - current_dim  # Right padding

        padded_x = torch.nn.functional.pad(x, pad)

    return padded_x


def pad_tensors(tensor_list):
    """
    Pad a tensors from a list to max shape adding zeros at the beginning.

    Args:
        tensor_list: List of tensors
    """
    if tensor_list is None or (not isinstance(tensor_list, list) and not isinstance(tensor_list, tuple)):
        return tensor_list

    if len(tensor_list) == 1:
        return tensor_list[0]

    # Find max shape for each dimension
    max_shapes = [max(t.shape[dim] for t in tensor_list) for dim in range(len(tensor_list[0].shape))]

    # Pad and concatenate
    padded = []
    for tensor in tensor_list:
        # Calculate padding (reversed order for torch.nn.functional.pad)
        padding = []
        for dim in reversed(range(len(tensor.shape))):
            pad_size = max_shapes[dim] - tensor.shape[dim]
            padding.extend([pad_size, 0])

        padded_tensor = torch.nn.functional.pad(tensor, padding, value=0) if any(p > 0 for p in padding) else tensor
        padded.append(padded_tensor)

    return padded


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def rope_fn(x, positions, max_wavelength=10_000):
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (..., d) where d is the embedding dimension
        positions: Position indices tensor
        max_wavelength: Maximum wavelength for frequency generation (default: 10000)

    Returns:
        Tensor with rotary position embeddings applied, same shape as input
    """
    dtype = x.dtype
    d = x.shape[-1]

    # Generate frequencies
    freq_exponents = (2.0 / d) * torch.arange(d // 2, device=x.device, dtype=torch.float32)
    timescale = max_wavelength**freq_exponents

    # Generate radians
    radians = positions.unsqueeze(-1) / timescale.unsqueeze(0).unsqueeze(0)
    radians = radians.unsqueeze(-2)  # Add head dimension

    # Apply sin/cos
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    # Apply rotation
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(dtype)


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(1.0))
    else:
        assert callable(value_schedule)
    # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
    # Some types are have odd behaviors when part of a Schedule, like numpy floats
    return get_lambda_schedule(value_schedule)


def get_lambda_schedule(value_schedule) -> Schedule:

    def func(progress_remaining, current_lr=None):
        return value_schedule(progress_remaining, current_lr)

    return func


def get_constant_schedule(val: float) -> Schedule:

    def func(progress_remaining, current_lr=None):
        return current_lr if current_lr is not None else val

    return func


def get_linear_schedule(start: float, end: float = 0.0, warmup_fraction: float = 0.05) -> Schedule:

    def func(progress_remaining, current_lr=None):
        return end if (1 - progress_remaining) > warmup_fraction else start + (1 - progress_remaining) * (end - start) / warmup_fraction

    return func


def get_cosine_schedule_with_warmup(initial_value: float, final_value: float = 0.0, warmup_fraction: float = 0.05) -> Schedule:
    """
    Create a schedule with cosine annealing that has a warmup period.

    :param initial_value: Initial learning rate value
    :param final_value: Final learning rate value after complete decay (usually close to 0)
    :param warmup_fraction: Fraction of total training steps used for warmup
    :return: Schedule function compatible with SB3
    """

    def func(progress_remaining: float, current_lr=None) -> float:
        # Convert progress_remaining (1->0) to progress (0->1)
        progress = 1.0 - progress_remaining

        # Warmup phase
        if progress < warmup_fraction:
            # Linear warmup from 0 to initial_value
            return progress * initial_value / warmup_fraction

        # Else cosine annealing phase
        # Adjusted progress after warmup (0->1)
        progress_adjusted = (progress - warmup_fraction) / (1 - warmup_fraction)
        # Cosine decay from initial_value to final_value
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_adjusted))
        return final_value + (initial_value - final_value) * cosine_decay

    return func


def update_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Pytorch optimizer
    :param learning_rate: New learning rate value
    """
    learning_rate = learning_rate if isinstance(learning_rate, list) else [learning_rate] * len(optimizer.param_groups)
    for p, lr in zip(optimizer.param_groups, learning_rate):
        p["lr"] = lr


def get_activation_fn(activation: str) -> Callable:
    """
    Return an activation function given a string identifier.

    Args:
        activation: String identifier for activation function ('relu', 'gelu', 'glu', 'tanh')

    Returns:
        Activation function from torch.nn.functional

    Raises:
        RuntimeError: If activation string is not recognized
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def get_activation_mod(activation: str) -> nn.Module:
    """
    Return an activation module given a string identifier.

    Args:
        activation: String identifier for activation function ('relu', 'gelu', 'glu', 'tanh')

    Returns:
        Activation module from torch.nn

    Raises:
        RuntimeError: If activation string is not recognized
    """
    if activation == "lrelu":
        return nn.LeakyReLU()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "elu":
        return nn.ELU()
    if activation == "glu":
        return nn.GLU()
    if activation == "tanh":
        return nn.Tanh()
    raise RuntimeError(f"activation should be relu/gelu/elu/glu, not {activation}.")


def wasserstein_gaussian(mean1, std1, mean2, std2, eps=1e-8):
    """
    Compute Wasserstein-2 distance between two multivariate Gaussians
    with diagonal covariance matrices.

    For N(μ₁, Σ₁) and N(μ₂, Σ₂) with diagonal Σ:
    W₂²(P,Q) = ||μ₁-μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁^½ Σ₂ Σ₁^½)^½)

    With diagonal covariance, this simplifies to:
    W₂²(P,Q) = ||μ₁-μ₂||² + Σᵢ(√σ₁ᵢ - √σ₂ᵢ)²

    Args:
        mean1, mean2: [batch, action_dim] means
        std1, std2: [batch, action_dim] standard deviations

    Returns:
        wasserstein_dist: [batch] Wasserstein distances
    """
    # Mean difference term
    mean_diff = (mean1 - mean2).pow(2).sum(dim=-1)

    # Covariance term (simplified for diagonal case)
    # Σᵢ(√σ₁ᵢ - √σ₂ᵢ)²
    sqrt_std1 = torch.sqrt(std1.pow(2) + eps)
    sqrt_std2 = torch.sqrt(std2.pow(2) + eps)
    cov_diff = (sqrt_std1 - sqrt_std2).pow(2).sum(dim=-1)

    # Total Wasserstein-2 distance squared
    wasserstein_sq = mean_diff + cov_diff

    # Return the distance (not squared)
    return torch.sqrt(wasserstein_sq + eps)


def wasserstein_divergence_penalty(mean1, std1, mean2, std2, target_dist=0.1):
    """
    Compute Wasserstein divergence penalty for PPO.

    Args:
        mean1: new policy mean
        std1: new policy std
        mean2: old policy mean
        std2: old policy std
        target_dist: target Wasserstein distance (similar to target_kl)

    Returns:
        penalty: scalar penalty
        wasserstein_mean: mean Wasserstein distance for logging
    """
    w_dist = wasserstein_gaussian(mean1, std1, mean2, std2)
    w_mean = w_dist.mean()

    # Penalty: quadratic beyond target
    # If W < target: no penalty
    # If W > target: quadratic penalty
    penalty = torch.clamp(w_dist - target_dist, min=0.0).pow(2).mean()

    return penalty, w_mean
