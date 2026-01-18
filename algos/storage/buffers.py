import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Union, Any

import numpy as np
import torch
from accelerate import Accelerator
from einops import repeat
from gymnasium import spaces
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise, ActionNoise
from torch.utils.data import DataLoader, Dataset

from common.utils import DictObj


@dataclass
class ReplayBufferSamples:
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor


@dataclass
class RolloutBufferSamples:
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_mask: torch.Tensor
    mask: torch.Tensor


@dataclass
class RecurrentRolloutBufferSamples:
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_mask: torch.Tensor
    memories: torch.Tensor
    mask: torch.Tensor


@dataclass
class TransformerRolloutBufferSamples:
    observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_mask: torch.Tensor
    memories: torch.Tensor
    memory_mask: torch.Tensor
    memory_indices: torch.Tensor
    mask: torch.Tensor


class TorchBuffer:
    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device,
        cpu_offload: bool = False,
        num_workers: int = 0,
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        self.obs_shapes = {key: {"shape":space.shape, "dtype":space.dtype} for key, space in observation_space.spaces.items()} if self.is_dict_obs else observation_space.shape
        self.action_dim = action_space.shape[0]
        self.pos = 0
        self.full = False
        self.device = self.out_device = device
        self.cpu_offload = cpu_offload
        self.num_workers = num_workers
        self.accelerator = None

        if self.cpu_offload:
            self.device = torch.device("cpu")

    def set_accelerator(self, accelerator: Accelerator):
        self.accelerator = accelerator

    def _flatten_helper(self, arr: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Swap and then flatten first two dimensions."""
        if isinstance(arr, dict):
            return {k: self._flatten_helper(v) for k, v in arr.items()}

        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)

        return arr.view(shape[0] * shape[1], *shape[2:])  # [env1step1obs, env1step2obs, env1step3obs]


class ReplayBuffer(TorchBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of elements in the buffer
    :param n_envs: Number of parallel environments
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param cpu_offload: Enable CPU offloading
    :param num_workers: Number of workers for data loading
    :param add_action_noise: Whether to add action noise
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    """
    def __init__(self, *args, add_action_noise=False, optimize_memory_usage=True, **kwargs):
        super().__init__(*args, **kwargs)


        self.optimize_memory_usage = optimize_memory_usage
        self.action_noise: ActionNoise = VectorizedActionNoise(NormalActionNoise(np.array([0]),np.array([0.1])), self.n_envs) if add_action_noise else None
        self.observations = {key: torch.zeros((self.n_envs, self.buffer_size, *obs_info["shape"]), device=self.device, dtype=eval("torch."+str(obs_info["dtype"]))) for key, obs_info in self.obs_shapes.items()} \
            if self.is_dict_obs else torch.zeros((self.n_envs, self.buffer_size, *self.obs_shapes), device=self.device)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = {k: torch.zeros_like(v) for k, v in self.observations.items()} \
                if self.is_dict_obs else torch.zeros_like(self.observations)

        self.actions = torch.zeros((self.n_envs, self.buffer_size, self.action_dim), device=self.device)
        self.rewards = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.dones = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.timeouts = torch.zeros((self.n_envs, self.buffer_size), device=self.device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        # Copy to avoid modification by reference
        if self.is_dict_obs:
            for key, tensor in obs.items():
                self.observations[key][:, self.pos] = tensor.clone()
                if self.optimize_memory_usage:
                    self.observations[key][:, (self.pos + 1) % self.buffer_size] = next_obs[key].clone()
                else:
                    self.next_observations[key][:, self.pos] = next_obs[key].clone()
        else:
            self.observations[:, self.pos] = obs.clone()
            if self.optimize_memory_usage:
                self.observations[:, (self.pos + 1) % self.buffer_size] = next_obs.clone()
            else:
                self.next_observations[:, self.pos] = next_obs.clone()

        self.actions[:, self.pos] = torch.clip(action+torch.from_numpy(self.action_noise()).to(action.device), -1, 1) if self.action_noise is not None else action
        self.rewards[:, self.pos] = reward
        self.dones[:, self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def restart_noise(self, dones: torch.Tensor) -> None:
        """Reset noise for corresponding environments that have been resetted."""
        if self.action_noise is not None and torch.any(dones):
            self.action_noise.reset(**dict(indices=torch.nonzero(dones).tolist()))

    def get(self, batch_size: int) -> Generator[ReplayBufferSamples, Any, Any]:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of elements to sample
        :return: Generator yielding ReplayBufferSamples
        """
        stored_samples = (self.buffer_size if self.full else self.pos) * self.n_envs
        if not self.full and batch_size > stored_samples:  # In case the buffer is not full we wait to get enough data before training
            print(f"Training step skipped: not enough data ({stored_samples} samples stored) to sample a batch of size {batch_size}.")
            return []

        while True:
            if not self.optimize_memory_usage:
                step_indices = torch.randint(0, self.buffer_size if self.full else self.pos, (batch_size,), device=self.device)
            else:  # Do not sample the element with index `self.pos` as the transition is invalid
                if self.full:
                    step_indices = (torch.randint(1, self.buffer_size, (batch_size,), device=self.device) + self.pos) % self.buffer_size
                else:
                    step_indices = torch.randint(0, self.pos, (batch_size,), device=self.device)

            env_indices = torch.randint(0, self.n_envs, (batch_size,), device=self.device)

            if self.optimize_memory_usage:
                # Get next observations from the (step + 1) % buffer_size position
                next_step_indices = (step_indices + 1) % self.buffer_size
                if self.is_dict_obs:
                    observations = {key: tensor[env_indices, step_indices].to(self.out_device) for key, tensor in self.observations.items()}
                    next_observations = {key: tensor[env_indices, next_step_indices].to(self.out_device) for key, tensor in self.observations.items()}
                else:
                    observations = self.observations[env_indices, step_indices].to(self.out_device)
                    next_observations = self.observations[env_indices, next_step_indices].to(self.out_device)
            else:
                if self.is_dict_obs:
                    observations = {key: tensor[env_indices, step_indices].to(self.out_device) for key, tensor in self.observations.items()}
                    next_observations = {key: tensor[env_indices, step_indices].to(self.out_device) for key, tensor in self.next_observations.items()}
                else:
                    observations = self.observations[env_indices, step_indices].to(self.out_device)
                    next_observations = self.next_observations[env_indices, step_indices].to(self.out_device)

            yield ReplayBufferSamples(
                observations=observations,
                actions=self.actions[env_indices, step_indices].to(self.out_device),
                next_observations=next_observations,
                dones=(self.dones[env_indices, step_indices] * (1 - self.timeouts[env_indices, step_indices])).to(self.out_device),
                rewards=self.rewards[env_indices, step_indices].to(self.out_device),
                mask=torch.ones_like(self.rewards[env_indices, step_indices]).to(self.out_device),
            )

    def reset(self) -> None:
        self.pos = 0
        self.full = False


class RolloutBuffer(TorchBuffer):
    def __init__(self, *args, gae_lambda: float = 0.95, gamma: float = 0.99, normalize_advantage: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage

        if self.num_workers:
            print("`num_workers` parameter ignored: not supported by RolloutBuffer class.")

    def initialize_buffer(self) -> None:
        # Standard rollout buffer tensors
        self.observations = {key: torch.zeros((self.n_envs, self.buffer_size, *obs_info["shape"]), device=self.device, dtype=eval("torch."+str(obs_info["dtype"]))) for key, obs_info in self.obs_shapes.items()} \
            if self.is_dict_obs else torch.zeros((self.n_envs, self.buffer_size, *self.obs_shapes), device=self.device)
        self.actions = torch.zeros((self.n_envs, self.buffer_size, self.action_dim), device=self.device)
        self.rewards = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.advantages = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.returns = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.dones = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.values = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.log_probs = torch.zeros((self.n_envs, self.buffer_size), device=self.device)
        self.mean = torch.zeros((self.n_envs, self.buffer_size, self.action_dim), device=self.device)
        self.std = torch.zeros((self.n_envs, self.buffer_size, self.action_dim), device=self.device)
        self.action_mask = torch.zeros((self.n_envs, self.buffer_size, self.action_dim), device=self.device, dtype=torch.bool)
        self.generator_ready = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        assert not self.generator_ready, "The buffer requires a reset operation!"
        if self.full:
            return self.reset()

        if self.is_dict_obs:
            for key, tensor in obs.items():
                self.observations[key][:, self.pos] = tensor
        else:
            self.observations[:, self.pos] = obs
        self.actions[:, self.pos] = action
        self.rewards[:, self.pos] = reward
        self.dones[:, self.pos] = done
        self.values[:, self.pos] = value
        self.log_probs[:, self.pos] = log_prob
        self.mean[:, self.pos] = mean
        self.std[:, self.pos] = std
        self.action_mask[:, self.pos] = action_mask

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
        return None

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_is_not_terminal = 1.0 - dones.long().to(self.device)
                next_value = last_values.to(self.device)
            else:
                next_is_not_terminal = 1.0 - self.dones[:, step + 1]
                next_value = self.values[:, step + 1]

            delta = self.rewards[:, step] + self.gamma * next_value * next_is_not_terminal - self.values[:, step]
            last_gae_lam = delta + next_is_not_terminal * self.gamma * self.gae_lambda * last_gae_lam
            self.advantages[:, step] = last_gae_lam

        self.returns = self.advantages + self.values

        if self.normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        indices = torch.randperm(self.buffer_size * self.n_envs, device=self.device)

        # Prepare the data flattening the class' tensors
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "mean", "std", "advantages", "returns", "action_mask"]:
                self.__dict__[tensor] = self._flatten_helper(self.__dict__[tensor])
            self.generator_ready = True

        batch_size = batch_size or self.buffer_size * self.n_envs
        start_idx = 0

        while start_idx < self.buffer_size * self.n_envs:
            # start yielding from [env1step1obs, env1step2obs, env1step3obs, ...]
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_idx: torch.Tensor) -> RolloutBufferSamples:
        if self.is_dict_obs:
            observations = {key: tensor[batch_idx].to(self.out_device) for key, tensor in self.observations.items()}
        else:
            observations = self.observations[batch_idx].to(self.out_device)

        return RolloutBufferSamples(
            observations=observations,
            actions=self.actions[batch_idx].to(self.out_device),
            old_values=self.values[batch_idx].flatten().to(self.out_device),
            old_log_prob=self.log_probs[batch_idx].flatten().to(self.out_device),
            old_mean=self.mean[batch_idx].to(self.out_device),
            old_std=self.std[batch_idx].to(self.out_device),
            advantages=self.advantages[batch_idx].flatten().to(self.out_device),
            returns=self.returns[batch_idx].flatten().to(self.out_device),
            action_mask=self.action_mask[batch_idx].to(self.out_device),
            mask=torch.ones_like(self.returns[batch_idx].flatten(), device=self.out_device),
        )

    def reset(self) -> None:
        self.full = False
        self.generator_ready = False
        self.initialize_buffer()
        self.pos = 0


class RecurrentRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, memory_shape, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory_shape = memory_shape

    def initialize_buffer(self) -> None:
        # Standard rollout buffer tensors
        self.memories = torch.zeros((self.n_envs, self.buffer_size+1, *self.memory_shape), device=self.device)

        super().initialize_buffer()

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        action_mask: torch.Tensor,
        memory: torch.Tensor,
    ):
        assert not self.generator_ready, "The buffer requires a reset operation!"
        if self.full:
            return self.reset()

        self.memories[:, self.pos+1] = memory.detach()

        return super().add(obs, action, reward, done, value, log_prob, mean, std, action_mask)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        indices = torch.randperm(self.buffer_size * self.n_envs, device=self.device)

        # Prepare the data flattening the class' tensors
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "memories", "log_probs", "mean", "std", "advantages", "returns", "action_mask"]:
                self.__dict__[tensor] = self._flatten_helper(self.__dict__[tensor])
            self.generator_ready = True

        batch_size = batch_size or self.buffer_size * self.n_envs
        start_idx = 0

        while start_idx < self.buffer_size * self.n_envs:
            # start yielding from [env1step1obs, env1step2obs, env1step3obs, ...]
            yield self._get_samples(indices[start_idx:start_idx+batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_idx: torch.Tensor) -> RecurrentRolloutBufferSamples:
        if self.is_dict_obs:
            observations = {key: tensor[batch_idx].to(self.out_device) for key, tensor in self.observations.items()}
        else:
            observations = self.observations[batch_idx].to(self.out_device)

        # we need to shift the idx for memories since `self.memories` is long `self.buffer_size+1`
        batch_idx_shifted = ((batch_idx // self.buffer_size) * (self.buffer_size+1)) + (batch_idx % self.buffer_size)

        return RecurrentRolloutBufferSamples(
            observations=observations,
            actions=self.actions[batch_idx].to(self.out_device),
            old_values=self.values[batch_idx].flatten().to(self.out_device),
            old_log_prob=self.log_probs[batch_idx].flatten().to(self.out_device),
            old_mean=self.mean[batch_idx].to(self.out_device),
            old_std=self.std[batch_idx].to(self.out_device),
            advantages=self.advantages[batch_idx].flatten().to(self.out_device),
            returns=self.returns[batch_idx].flatten().to(self.out_device),
            action_mask=self.action_mask[batch_idx].to(self.out_device),
            memories=self.memories[batch_idx_shifted].to(self.out_device),
            mask=torch.ones_like(self.returns[batch_idx].flatten(), device=self.out_device),
        )

    def get_last(self) -> torch.Tensor:
        """
        Get the last stored LSTM hidden states.

        :return: Memory tensor at current position with shape (n_envs, *memory_shape)
        """
        return self.memories[:, self.pos].to(self.out_device)

    def restart_memory(self, dones: torch.Tensor, init_tensor: Optional[torch.Tensor] = None) -> None:
        """
        Reset stored memories for corresponding environments that have been reset.

        :param dones: Boolean tensor indicating which environments have been reset
        :param init_tensor: Optional tensor to use for initialization instead of zeros.
                           Should have shape (n_envs, *memory_shape) or broadcastable to it.
        """
        if torch.any(dones):
            self.memories[dones, self.pos] = init_tensor[dones].to(self.device) if init_tensor is not None else 0

    def reset(self) -> None:
        # TODO we could get memory[self.buffer_size] and put it in memory[0]
        return super().reset()


class TransformerRolloutBuffer(TorchBuffer):
    def __new__(cls, *args, **kwargs):
        if cls is TransformerRolloutBuffer:
            if kwargs.get("num_workers", 0) > 0:
                return super().__new__(TransformerRolloutDataloader)
            if kwargs.get("cpu_offload", False):
                return super().__new__(OffloadTransformerRolloutBuffer)

        return super().__new__(cls)

    def __init__(self, *args, max_tokens: int = 0, max_tokens_per_step: int = 0, gae_lambda: float = 0.95, gamma: float = 0.99,
                 normalize_advantage: bool = True, seq_length: int = 1,
                 memory_length: int = 1, num_blocks: int = 4, embed_dim: int = 64, num_heads: int = 8,
                 causal_memory: bool = False, memorize_cache: bool = True, num_attn_projections: int = 3,
                 self_attn_cache: bool = False, cross_attn_cache: bool = False, **kwargs):
        """Buffer implementation for a Transformer.

        This implementation offers both possibility of storing at each step a token like a recurrent model that autoregressively works on an increasing sequence (`memorize_cache` False)
        or store at each step the attention projections (e.g. qkv) and so accumulate the tokens at the cache level (`memorize_cache` True)
        - Regarding the implementation where we store the token sequence positional embedding indexes and masks are already precomputed at initialization and memory is efficiently reconstructed
        gathering the relevant steps in the past.
        - Storing the cache is much more demanding in terms of computation and memory therefore we provide different optimized solutions (e.g. TransformerRolloutDataloader and OffloadTransformerRolloutBuffer)
        in addition to the more linear implementation of this class. We notice the major slowdown in _reconstruct_cache. The option of offloading the data for saving space is easily hidden behind
        the cost of memory reconstruction.

        :param max_tokens: if memorize_cache is True; since cache grows fast the storage is implemented as a moving window with a maximum number of tokens. Retrieving (by reconstruction) the cache will result in a tensor of `max_tokens` zero left-padded.
        :param max_tokens_per_step: if memorize_cache is True; ideally the maximum number of tokens between qkv that is being stored at each step. Smoller sequences will be zero left-padded.
        :param gae_lambda: gae discount for computing the rollout returns.
        :param gamma: gamma for computing the rollout returns.
        :param normalize_advantage: whether to normalize advantages respect the whole rollout.
        :param seq_length: if memorize_cache is False; represent the number of q tokens necessary to build the attention mask.
        :param memory_length: if memorize_cache is False; the maximum number of steps/tokens of memory we consider going back.
        :param num_blocks: number of blocks of the transformer for which we save the token at each step (e.g. a multi-layer rnn for which we save the state at each layer).
        :param embed_dim: the embedding dimension of the tokens to save.
        :param num_heads: if memorize_cache is True; the number of heads dimension of the cache tokens to save.
        :param causal_memory: if memorize_cache is False; whether to use a diagonal "causal" mask or full.
        :param memorize_cache: store cache memory instead of sequence tokens.
        :param num_attn_projections: if memorize_cache is True; number of projection stored in cache e.g. 3 if storing q, k, v.
        :param self_attn_cache: if memorize_cache is True; consider cache for self-attention layer.
        :param cross_attn_cache: if memorize_cache is True; consider cache for cross-attention layer.
        """
        super().__init__(*args, **kwargs)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.seq_length = seq_length
        self.memory_length = memory_length
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.max_step_tokens = max_tokens_per_step
        self.causal_memory = causal_memory
        self.memorize_cache = memorize_cache
        self.num_attn_projections = num_attn_projections
        self.self_attn_cache = self_attn_cache
        self.cross_attn_cache = cross_attn_cache
        self.restarted_envs = torch.zeros(self.n_envs, device=self.device, dtype=torch.bool)

        if self.memorize_cache and (self.max_tokens == 0 or self.max_step_tokens == 0):
            print(f"No cache tokens will be used since `max_tokens` parameter is set to {self.max_tokens} and `max_step_tokens` parameter is set to {self.max_step_tokens}.`")

    def initialize_buffer(self) -> None:
        def zero_init(attr, shape, dtype=None, device=None):
            if device is None:
                device = self.device
            if not hasattr(self, attr):
                setattr(self, attr, torch.zeros(shape, dtype=dtype, device=device))
            else:
                getattr(self, attr).zero_()  # in place
                setattr(self, attr, getattr(self, attr).reshape(shape))

        # Standard rollout buffer tensors
        self.observations = {
            key: torch.zeros((self.n_envs, self.buffer_size, *obs_info["shape"]), device=self.device, dtype=eval("torch." + str(obs_info["dtype"])))
            for key, obs_info in self.obs_shapes.items()
        } if self.is_dict_obs else torch.zeros((self.n_envs, self.buffer_size, *self.obs_shapes), device=self.device)
        zero_init("actions", shape=(self.n_envs, self.buffer_size, self.action_dim))
        zero_init("rewards", shape=(self.n_envs, self.buffer_size))
        zero_init("advantages", shape=(self.n_envs, self.buffer_size))
        zero_init("returns", shape=(self.n_envs, self.buffer_size))
        zero_init("dones", shape=(self.n_envs, self.buffer_size))
        zero_init("values", shape=(self.n_envs, self.buffer_size))
        zero_init("log_probs", shape=(self.n_envs, self.buffer_size))
        zero_init("mean", shape=(self.n_envs, self.buffer_size, self.action_dim))
        zero_init("std", shape=(self.n_envs, self.buffer_size, self.action_dim))
        # Additional tensors for transformer
        zero_init("action_mask", shape=(self.n_envs, self.buffer_size, self.action_dim), dtype=torch.bool)
        zero_init("restarted_envs", shape=(self.n_envs,), dtype=torch.bool)

        # set of memories
        if hasattr(self, "memory_ordered_id_batch"):
            start = (self.memory_ordered_id_batch.view(self.n_envs, self.buffer_size+1, self.memory_length)[:, self.pos, 0] +
                     self.memory_mask_batch.view(self.n_envs, self.buffer_size+1, self.seq_length, self.memory_length)[:, self.pos, 0].sum(-1)).unsqueeze(-1)  # min ordered batch + mask summed
            start[self.restarted_envs] = 0
        else:
            start = 0

        if self.memorize_cache:
            attn_stores = self.num_attn_projections*(self.self_attn_cache+self.cross_attn_cache)
            # To avoid the time overhead of add()-get_last() when offloading we double-save the last step vars to have them already in out_device
            zero_init("last_mask", shape=(self.n_envs, 2, self.max_tokens, self.max_tokens), dtype=torch.bool, device=self.out_device)
            zero_init("last_memory", shape=(self.n_envs, self.num_blocks, attn_stores, self.max_tokens, self.num_heads, self.embed_dim // self.num_heads), device=self.out_device)

            zero_init("mask_cache", shape=(self.n_envs, self.buffer_size+1, 2, self.max_tokens, self.max_tokens), dtype=torch.bool)

            # zero_init("attn_cache", shape=(self.n_envs, self.buffer_size+1, self.num_blocks, attn_stores, self.max_tokens, self.num_heads, self.embed_dim // self.num_heads))
            # Instead of storing all cache for all steps, since the info overlap between steps, we incrementally store the cache saving the token counts
            zero_init("attn_cache_incremental", shape=(self.n_envs, self.buffer_size+1, self.num_blocks, attn_stores, self.max_step_tokens, self.num_heads, self.embed_dim // self.num_heads))
            # Track the number of tokens added in cache at each step
            zero_init("cache_step_sizes", shape=(self.n_envs, self.buffer_size+1, attn_stores), dtype=torch.long)
            # Track the cumulative number of tokens per attention store
            zero_init("cache_sizes", shape=(self.n_envs, self.buffer_size+1, attn_stores), dtype=torch.long)
        else:
            zero_init("memories", shape=(self.n_envs, self.buffer_size+1, self.num_blocks, self.embed_dim))
            # attention mask to use, always diagonal except for first steps of an episode where we have less memory (0-ed seq)
            zero_init("memory_mask_batch", shape=(self.n_envs, self.buffer_size+1, self.seq_length, self.memory_length), dtype=torch.bool)
            # each row being a `memory_len`-long set of indexes, we use it to gather a slice of memories
            zero_init("memory_selector_indices_batch", shape=(self.n_envs, self.buffer_size+1, self.memory_length), dtype=torch.int)
            # ordered_id is used as positional encoding index (step execution index of the env): reset at done but not at re-initialization of buffer
            zero_init("memory_ordered_id_batch", shape=(self.n_envs, self.buffer_size+1, self.memory_length), dtype=torch.int)

            # Prepare batch memory masks and indices
            self._memory_mask = torch.tril(torch.ones((self.memory_length + 1, self.memory_length), device=self.device, dtype=torch.bool), diagonal=-1)
            repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length, device=self.device).unsqueeze(0), self.memory_length - 1, dim=0).long()
            if self.buffer_size+1 >= self.memory_length:
                self._memory_indices = torch.stack([torch.clip(torch.arange(i, i + self.memory_length, device=self.device), max=self.buffer_size-1) for i in range((self.buffer_size+1) - self.memory_length + 1)]).long()
                self._memory_indices = torch.cat((repetitions, self._memory_indices))
            else:
                self._memory_indices = repetitions[:self.buffer_size+1]

            self.memory_mask_batch[:] = self._memory_mask[torch.clamp(torch.arange(self.buffer_size+1, device=self.device).view(-1, 1) - (torch.arange(self.seq_length - 1, -1, -1, device=self.device) if self.causal_memory else torch.zeros(self.seq_length, device=self.device, dtype=torch.long)), 0, self.memory_length)]
            for i in range(self.buffer_size+1):
                self.memory_selector_indices_batch[:, i] = self._memory_indices[i].int()
                self.memory_ordered_id_batch[:, i] = start + self._memory_indices[i].int().unsqueeze(0)

        self.generator_ready = False

    def add(self, obs: torch.Tensor = None, action: torch.Tensor = None, reward: torch.Tensor = None,
            done: torch.Tensor = None, value: torch.Tensor = None, log_prob: torch.Tensor = None,
            mean: torch.Tensor = None, std: torch.Tensor = None,
            action_mask: torch.Tensor = None, memory: Union[torch.Tensor | List[torch.Tensor]] = None,
            mask: Union[torch.Tensor | List[torch.Tensor]] = None) -> None:
        assert not self.generator_ready, "The buffer requires a reset operation!"
        if self.full:
            return self.reset()

        if obs is not None:
            if self.is_dict_obs:
                for key in self.observations.keys():
                    if key in obs:
                        self.observations[key][:, self.pos] = obs[key]
            else:
                self.observations[:, self.pos] = obs
        action is not None and self.actions.__setitem__((slice(None), self.pos), action)
        action_mask is not None and self.action_mask.__setitem__((slice(None), self.pos), action_mask)
        reward is not None and self.rewards.__setitem__((slice(None), self.pos), reward)
        done is not None and self.dones.__setitem__((slice(None), self.pos), done)
        value is not None and self.values.__setitem__((slice(None), self.pos), value)
        log_prob is not None and self.log_probs.__setitem__((slice(None), self.pos), log_prob)
        mean is not None and self.mean.__setitem__((slice(None), self.pos), mean)
        std is not None and self.std.__setitem__((slice(None), self.pos), std)

        if self.memorize_cache:
            if (self.max_tokens and (self.self_attn_cache+self.cross_attn_cache)>0 and
                    memory is not None and (not isinstance(memory, list) or (isinstance(memory, list) and len(memory)>0)) and mask is not None):
                for i, attn_cache in enumerate(memory):  # ideally memory is a list of (self.num_attn_projections * (self.self_attn_cache+self.cross_attn_cache))
                    self.last_memory[:, :, i, -min(attn_cache.size(2), self.max_tokens):] = attn_cache[:, :, -min(attn_cache.size(2), self.max_tokens):]

                    # we don't store the cache as it is but last addition
                    # self.attn_cache[:, self.pos+1, :, i, -min(attn_cache.size(2), self.max_tokens):] = attn_cache[:, :, -self.max_tokens:]
                    current_cache_size = self._get_actual_cache_size(attn_cache)

                    # Get incremental update for each environment
                    prev_size = self.cache_sizes[:, max(self.pos-1, 0), i]
                    new_size = current_cache_size.to(self.device)
                    new_tokens_per_env = torch.where(new_size > prev_size, new_size - prev_size, new_size)
                    assert (new_tokens_per_env <= self.max_tokens).all(), "Single step cache size is exceeding the (global) maximum number of tokens stored in the buffer."
                    assert (new_tokens_per_env <= self.max_step_tokens).all(), "In theory this is allowed, but in practice while we store left-truncated step's tokens in attn_cache_incremental it would be expensive doing the same for last_memory. Let's try to avoid this situation."
                    max_new_tokens = new_tokens_per_env.max().item()

                    # Store the tokens: left-padded in incremental storage if less than max_step_tokens, or left-truncated if too many
                    if max_new_tokens > 0:
                        self.attn_cache_incremental[:, self.pos+1, :, i, -min(attn_cache.size(2), self.max_step_tokens):] = attn_cache[:, :, -min(attn_cache.size(2), self.max_step_tokens):]

                    # Store how many new tokens were added for each env at this step
                    self.cache_step_sizes[:, self.pos+1, i] = new_tokens_per_env.clip(min=0, max=self.max_step_tokens)
                    self.cache_sizes[:, self.pos+1, i] = torch.where(new_size > prev_size, prev_size+self.cache_step_sizes[:, self.pos, i], self.cache_step_sizes[:, self.pos, i]).clip(min=0, max=self.max_tokens)

                for i, mask_cache in enumerate(mask):
                    self.last_mask[:, i, -min(mask_cache.size(1), self.max_tokens):, -min(mask_cache.size(2), self.max_tokens):] = mask_cache[:, -min(mask_cache.size(1), self.max_tokens):, -min(mask_cache.size(2), self.max_tokens):]
                    self.mask_cache[:, self.pos+1, i, -min(mask_cache.size(1), self.max_tokens):, -min(mask_cache.size(2), self.max_tokens):] = mask_cache[:, -min(mask_cache.size(1), self.max_tokens):, -min(mask_cache.size(2), self.max_tokens):]
        else:
            self.memories[:, self.pos+1] = memory

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

        return None

    def restart_memory(self, dones: torch.Tensor, steps=None) -> None:
        """Reset stored memories for corresponding environments that have been reset.

        :param steps: if memorize_cache is False; number of steps to which propagate back the reset
        """
        self.restarted_envs = dones  # when pos==buffer_size the below cycle has no effect therefore we apply the restart at re-initialization of buffer
        if torch.any(dones):
            if self.memorize_cache:
                self.last_memory[dones] = 0
                self.last_mask[dones] = 0
                # Reset cache sizes for done environments
                if not self.full:
                    self.cache_sizes[dones, self.pos] = 0
            else:
                self.memories[dones, self.pos] = 0
                span_back = (self.pos + min(steps + 1, (self.buffer_size+1) - self.pos)) if steps is not None else self.buffer_size+1
                for i in range(self.pos, span_back):
                    zero_i = i - self.pos
                    for j in reversed(range(self.seq_length)):
                        self.memory_mask_batch[dones, i, j, :] = self._memory_mask[
                            torch.clip(torch.tensor(zero_i, device=self.device), 0, self.memory_length)  # single element sequence
                            if not self.causal_memory else torch.clip(
                                torch.clip(torch.tensor(zero_i, device=self.device), 0, self.memory_length) - (self.seq_length - 1 - j), 0, self.memory_length)  # multi element sequence and triangular mask
                        ]
                    # Note: accessing a flattened (self.n_envs*self.buffer_size) list of memories we try to gather [dones, i:i+self.memory_length] using the `memory_selector_indices_batch`. We have to make sure i+self.memory_length < self.buffer_size.
                    self.memory_selector_indices_batch[dones, i] = torch.clip(self.pos + self._memory_indices[zero_i].int(), max=self.buffer_size - 1)
                    # reset pos emb indices
                    self.memory_ordered_id_batch[dones, i] = self._memory_indices[zero_i].int()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor) -> None:
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_is_not_terminal = 1.0 - dones.long().to(self.device)
                next_value = last_values.to(self.device)
            else:
                next_is_not_terminal = 1.0 - self.dones[:, step + 1]
                next_value = self.values[:, step + 1]

            delta = self.rewards[:, step] + self.gamma * next_value * next_is_not_terminal - self.values[:, step]
            last_gae_lam = delta + next_is_not_terminal * self.gamma * self.gae_lambda * last_gae_lam
            self.advantages[:, step] = last_gae_lam

        self.returns = self.advantages + self.values

        if self.normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def _prepare_data(self):
        """Flatten tensors in the buffer for easy indexing"""
        if not self.generator_ready:
            tensors_to_flatten = ["observations", "actions", "values", "log_probs", "mean", "std", "advantages", "action_mask", "returns"] + (
                ["mask_cache"] if self.memorize_cache else ["memories", "memory_mask_batch", "memory_selector_indices_batch", "memory_ordered_id_batch"]
            )

            for tensor in tensors_to_flatten:
                self.__dict__[tensor] = self._flatten_helper(self.__dict__[tensor])

            self.generator_ready = True

    def _get_actual_cache_size(self, cache: torch.Tensor) -> torch.Tensor:
        """
        Detect actual number of non-zero tokens in cache for each environment
        """
        # cache shape: [n_envs, num_layers, seq_len, num_heads, head_dim]
        # Sum across heads and head_dim to detect non-zero tokens
        non_zero = cache.abs().sum(dim=[1, 3, 4]) > 0  # [n_envs, seq_len]

        # Find first non-zero position for each env using argmax
        # argmax returns 0 if all False, so we need to check if any non-zero exists
        has_nonzero = non_zero.any(dim=1)  # [n_envs]
        first_nonzero_pos = non_zero.long().argmax(dim=1)  # [n_envs]
        sizes = torch.where(has_nonzero, cache.shape[2] - first_nonzero_pos, torch.zeros_like(first_nonzero_pos))

        return sizes

    def _reconstruct_cache(self, env_idx: int, step: int) -> torch.Tensor:
        """
        Reconstruct full cache at a given step by combining incremental updates
        """
        n_stores, iters_per_group, attn_groups = self.num_attn_projections, 2, int(self.self_attn_cache+self.cross_attn_cache)

        # Start with empty cache
        cache = torch.zeros(self.num_blocks, attn_groups*n_stores, self.max_tokens, self.num_heads, self.embed_dim // self.num_heads, device=self.device)

        # Process each attention store
        for store in range(iters_per_group*attn_groups*attn_groups):
            # Find how many tokens we need to accumulate
            store_i = ((store//iters_per_group)*n_stores)+(store%iters_per_group)  # 0,1 , 4,5 , 8,9 , ...
            total_tokens_needed = self.cache_sizes[env_idx, step, store_i]

            if total_tokens_needed == 0:
                continue

            # Walk backwards to find where to start
            tokens_accumulated, s, current_pos = 0, step, self.max_tokens
            while tokens_accumulated < total_tokens_needed and s >= 0:
                included_stores = 1 if store%iters_per_group == 0 else 3  # hard coded bit to address q and kvs
                step_tokens = self.cache_step_sizes[env_idx, s, store_i]
                tokens_accumulated += step_tokens
                # Fill cache for this step
                new_tokens = min(step_tokens.item(), current_pos)
                if new_tokens > 0:
                    cache[:, store:store+included_stores, current_pos-new_tokens:current_pos] = self.attn_cache_incremental[env_idx, s, :, store:store+included_stores, -new_tokens:]
                    current_pos -= new_tokens
                s -= 1

        return cache

    def get(self, batch_size: Optional[int] = None) -> Generator[TransformerRolloutBufferSamples, None, None]:
        indices = torch.randperm(self.buffer_size * self.n_envs, device=self.device)

        # Prepare the data flattening the class' tensors
        self._prepare_data()

        batch_size = batch_size or self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            # start yielding from [env1step1obs, env1step2obs, env1step3obs, ...]
            samples = self._get_samples(indices[start_idx : start_idx + batch_size])
            yield samples
            start_idx = start_idx + batch_size
            del samples
            gc.collect()

    def _get_samples(self, batch_idx: torch.Tensor) -> TransformerRolloutBufferSamples:
        # we drop `n_envs` first dimension in the flattening therefore we sum the buffer_size*env;
        if self.is_dict_obs:
            observations = {key: tensor[batch_idx].to(self.out_device) for key, tensor in self.observations.items()}
        else:
            observations = self.observations[batch_idx].to(self.out_device)

        if self.memorize_cache:
            # memories = self.attn_cache[batch_idx].to(self.out_device)
            # masks = self.mask_cache[batch_idx].to(self.out_device)

            # Reconstruct caches for sampled indices
            attn_stores = self.num_attn_projections*(self.self_attn_cache+self.cross_attn_cache)
            memories = torch.zeros(len(batch_idx), self.num_blocks, attn_stores, self.max_tokens, self.num_heads, self.embed_dim // self.num_heads, device=self.out_device)

            if attn_stores > 0:
                for i, idx in enumerate(batch_idx):
                    env_idx = idx // self.buffer_size
                    step_idx = idx % self.buffer_size

                    # Reconstruct cache at this step
                    cache = self._reconstruct_cache(env_idx.item(), step_idx.item())
                    memories[i] = cache.to(self.out_device)

            # we need to shift the idx for masks since `self.mask_cache` is flattened as other tensors, but differently it's long `self.buffer_size+1`
            batch_idx_shifted = ((batch_idx // self.buffer_size) * (self.buffer_size+1)) + (batch_idx % self.buffer_size)
            masks = self.mask_cache[batch_idx_shifted].to(self.out_device)
            indices = torch.empty(0)
        else:
            # we need to shift the idx for flattened tensors long `self.buffer_size+1`
            batch_idx_shifted = ((batch_idx // self.buffer_size) * (self.buffer_size+1)) + (batch_idx % self.buffer_size)
            selector_indices_batch_flattened = (((batch_idx // self.buffer_size) * (self.buffer_size+1)).unsqueeze(-1) + self.memory_selector_indices_batch[batch_idx_shifted]).flatten()
            # then we gather `batch_size`x`memory_length` memories (in `get_last()` we gather only `1`x`memory_length`)
            memories = torch.gather(self.memories, 0,
                                    repeat(selector_indices_batch_flattened, "... -> ... b h", b=self.num_blocks, h=self.embed_dim).long(),
                                    ).reshape(len(batch_idx), self.memory_length, self.num_blocks, self.embed_dim).to(self.out_device)
            masks = self.memory_mask_batch[batch_idx_shifted].to(self.out_device)
            indices = self.memory_ordered_id_batch[batch_idx_shifted].to(self.out_device)
        return TransformerRolloutBufferSamples(
            **{
                "observations": observations,
                "actions": self.actions[batch_idx].to(self.out_device),
                "old_values": self.values[batch_idx].flatten().to(self.out_device),
                "old_log_prob": self.log_probs[batch_idx].flatten().to(self.out_device),
                "old_mean": self.mean[batch_idx].to(self.out_device),
                "old_std": self.std[batch_idx].to(self.out_device),
                "advantages": self.advantages[batch_idx].flatten().to(self.out_device),
                "returns": self.returns[batch_idx].flatten().to(self.out_device),
                "action_mask": self.action_mask[batch_idx].to(self.out_device),
                "memories": memories,
                "memory_mask": masks,
                "memory_indices": indices,
                "mask": torch.ones_like(self.returns[batch_idx].flatten()).to(self.out_device),
            }
        )

    def get_last(self):
        if self.memorize_cache:  # noqa: R505
            memories = self.last_memory.to(self.out_device)  # not offloaded for speedup
            masks = self.last_mask.to(self.out_device)  # not offloaded for speedup

            return memories, masks

        else:
            memories = torch.gather(self.memories, 1,
                                    repeat(self.memory_selector_indices_batch[:, self.pos], "... -> ... b h", b=self.num_blocks,
                                           h=self.embed_dim).long(), ).contiguous().to(self.out_device)
            masks = self.memory_mask_batch[:, self.pos].to(self.out_device)
            indices = self.memory_ordered_id_batch[:, self.pos].to(self.out_device)

            return memories, masks, indices

    def reset(self) -> None:
        # TODO we could get memory[self.buffer_size] and put it in memory[0]
        self.full = False
        self.generator_ready = False
        self.initialize_buffer()
        self.pos = 0


class OffloadTransformerRolloutBuffer(TransformerRolloutBuffer):
    """Offload memory data on cpu and hide memory reconstruction cost for one batch while moving another one to gpu and executing training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.cpu_offload, "cpu_offload is required for OptimizedTransformerRolloutBuffer"
        assert self.out_device == torch.device("cuda"), "out_device must be 'cuda' for OptimizedTransformerRolloutBuffer"

        # 1. PRE-ALLOCATION FOR DOUBLE BUFFERING
        # We use two sets of buffers in pinned memory: one for the CPU to fill (rollout storage to pinned location), one for the GPU to read (pinned location to GPU)
        self.pinned_buffers = []
        self.num_pinned_buffers = 2

        # Defer memory allocation until the first 'get' call when we know the batch_size (assumed constant)
        self.is_pinned_initialized = False

        # 2. THREADING & STREAMS
        self.prefetch_executor = ThreadPoolExecutor(max_workers=1)  # The single thread moving in CPU
        self.transfer_stream = torch.cuda.Stream()
        self.transfer_event = torch.cuda.Event()

    def _init_pinned_buffers(self, batch_size):
        """Allocates page-locked memory (pinned) for fast CPU->GPU transfer."""
        if self.is_pinned_initialized:
            return

        # Determine shapes
        # We use a dummy sample to get shapes/dtypes
        dummy_idx = torch.tensor([0], device=self.device)
        # We use the original _get_samples but keep it on CPU to check shapes
        original_out_device = self.out_device
        self.out_device = torch.device("cpu")  # Temporarily force CPU
        sample_meta = super()._get_samples(dummy_idx)
        self.out_device = original_out_device  # Restore

        for _ in range(self.num_pinned_buffers):
            # Helper to recursively create pinned tensors
            def create_pinned(structure):
                if isinstance(structure, torch.Tensor):
                    # Create tensor with batch_size dimension expanded
                    shape = (batch_size, *structure.shape[1:])
                    return torch.zeros(shape, dtype=structure.dtype).pin_memory()
                if isinstance(structure, dict):
                    return {k: create_pinned(v) for k, v in structure.items()}
                if hasattr(structure, "__dataclass_fields__"):  # Handle Dataclass
                    return {f: create_pinned(getattr(structure, f)) for f in structure.__dataclass_fields__}
                return structure

            pinned_sample = create_pinned(sample_meta)
            self.pinned_buffers.append(pinned_sample)

        self.is_pinned_initialized = True

    def _fill_pinned_buffer(self, batch_idx, buffer_idx):
        """
        CPU Worker Function:
        Reconstructs cache and writes directly into the pinned buffer.
        CRITICAL: No GPU ops allowed here.
        """
        dest = self.pinned_buffers[buffer_idx]

        # We access the raw storage directly.
        # Optimized inner loop: avoid .to(device) calls inside loop

        # 1. Observations
        if self.is_dict_obs:
            for key in self.observations.keys():
                # Direct copy
                dest["observations"][key].copy_(self.observations[key][batch_idx])
        else:
            dest["observations"].copy_(self.observations[batch_idx])

        # 2. Cache Reconstruction (The Heavy CPU Part)
        if self.memorize_cache:
            # This loop is CPU bound. Since we are in a thread, it runs in parallel with GPU training of previous batch.
            for i, idx in enumerate(batch_idx):
                env_idx = (idx // self.buffer_size).item()
                step_idx = (idx % self.buffer_size).item()

                # Reconstruct (returns CPU tensor)
                cache = self._reconstruct_cache(env_idx, step_idx)

                # Direct write to pinned memory (Zero Copy overhead to intermediate tensors)
                dest["memories"][i].copy_(cache)

            # we need to shift the idx for masks since `self.mask_cache` is flattened as other tensors, but differently it's long `self.buffer_size+1`
            batch_idx_shifted = ((batch_idx // self.buffer_size) * (self.buffer_size+1)) + (batch_idx % self.buffer_size)
            dest["memory_mask"].copy_(self.mask_cache[batch_idx_shifted])
        else:
            # Vectorized path for non-memory cache (Standard Transformer)
            # we need to shift the idx for flattened tensors long `self.buffer_size+1`
            batch_idx_shifted = ((batch_idx // self.buffer_size) * (self.buffer_size+1)) + (batch_idx % self.buffer_size)
            selector_indices_batch_flattened = (((batch_idx // self.buffer_size) * (self.buffer_size+1)).unsqueeze(-1) + self.memory_selector_indices_batch[batch_idx_shifted]).flatten()

            # We must perform the gather on the device where storage is (likely CPU or GPU) then copy to pinned
            memories = torch.gather(self.memories, 0, repeat(selector_indices_batch_flattened, "... -> ... b h", b=self.num_blocks, h=self.embed_dim).long())
            dest["memories"].copy_(memories.reshape(len(batch_idx), self.memory_length, self.num_blocks, self.embed_dim))
            dest["memory_mask"].copy_(self.memory_mask_batch[batch_idx_shifted])
            dest["memory_indices"].copy_(self.memory_ordered_id_batch[batch_idx_shifted])

        # 3. Copy other tensors
        dest["actions"].copy_(self.actions[batch_idx])
        dest["old_values"].copy_(self.values[batch_idx].flatten())
        dest["old_log_prob"].copy_(self.log_probs[batch_idx].flatten())
        dest["old_mean"].copy_(self.mean[batch_idx])
        dest["old_std"].copy_(self.std[batch_idx])
        dest["advantages"].copy_(self.advantages[batch_idx].flatten())
        dest["returns"].copy_(self.returns[batch_idx].flatten())
        dest["action_mask"].copy_(self.action_mask[batch_idx])
        dest["mask"].fill_(1.0)  # Reset mask

        return buffer_idx

    def get(self, batch_size: Optional[int] = None):
        """
        Optimized Generator with Threaded Prefetching.
        """
        # Prepare indices
        indices = torch.randperm(self.buffer_size * self.n_envs, device=self.device)
        self._prepare_data()  # Flattens tensors

        batch_size = batch_size or self.buffer_size * self.n_envs
        self._init_pinned_buffers(batch_size)

        # Pipelining Variables
        total_samples = self.buffer_size * self.n_envs
        num_batches = (total_samples + batch_size - 1) // batch_size

        # Prime the pump: Submit first batch to thread
        current_idx = 0
        batch_indices = indices[current_idx : current_idx + batch_size]

        # We submit the fetch task to the executor: reconstruct cache and get data to pinned memory location
        future = self.prefetch_executor.submit(self._fill_pinned_buffer, batch_indices.cpu(), 0)

        for i in range(num_batches):
            # 1. Wait for CPU to finish preparing Batch i
            filled_buf_idx = future.result()  # This blocks until CPU work is done

            # 2. Submit in a thread the fetch task for a second batch (async)
            next_idx = current_idx + batch_size
            if next_idx < total_samples:
                next_batch_indices = indices[next_idx : min(next_idx + batch_size, total_samples)]
                # Handle last batch size mismatch if necessary, or drop last
                if len(next_batch_indices) == batch_size:
                    next_buf_idx = (filled_buf_idx + 1) % self.num_pinned_buffers
                    future = self.prefetch_executor.submit(self._fill_pinned_buffer, next_batch_indices.cpu(), next_buf_idx)
                else:
                    # Edge case: last batch smaller. For simplicity in this snippets, we might skip or handle dynamically
                    pass

            # 3. Move pinned data to GPU (non-blocking)
            pinned_data = self.pinned_buffers[filled_buf_idx]
            gpu_batch = {}
            with torch.cuda.stream(self.transfer_stream):

                def transfer(struct):
                    if isinstance(struct, torch.Tensor):
                        return struct.to(self.out_device, non_blocking=True)
                    if isinstance(struct, dict):
                        return {k: transfer(v) for k, v in struct.items()}

                # Transfer dict of pinned tensors to GPU
                for k, v in pinned_data.items():
                    gpu_batch[k] = transfer(v)

                # Post-pend the event of transfer completed
                self.transfer_stream.record_event(self.transfer_event)

            # 4. Synchronize Compute Stream
            # This tells the GPU: "Don't run the model on this batch until transfer is done"
            # It does NOT block the CPU. CPU loops back to step 1/2.
            torch.cuda.current_stream().wait_event(self.transfer_event)

            # 5. Reconstruct Dataclass and Yield
            yield TransformerRolloutBufferSamples(**gpu_batch)

            current_idx = next_idx

        # Cleanup
        del future


class TransformerRolloutDataset(Dataset):
    """Offload memory data on cpu and hide memory reconstruction cost of each sample in batch through cpu parallelizing: multi worker dataloader"""

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle dictionary observations and other structures"""
        result = {}
        keys = batch[0].keys()

        for key in keys:
            if key == "observations" and isinstance(batch[0][key], dict):  # Handle dictionary observations
                result[key] = {obs_key: torch.cat([b[key][obs_key] for b in batch], dim=0) for obs_key in batch[0][key].keys()}
            else:  # Handle other tensors
                result[key] = torch.cat([b[key] for b in batch], dim=0)
        return DictObj(result)  # policy access elements as attributes but accelerator device placement works on dicts

    def __init__(self, buffer):
        """
        Dataset wrapper for TransformerRolloutBuffer

        :param buffer: An initialized TransformerRolloutBuffer instance
        """
        self.buffer = buffer
        self.buffer._prepare_data()
        self.length = self.buffer.buffer_size * self.buffer.n_envs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get a single item from the buffer"""
        if self.buffer.is_dict_obs:
            observations = {key: tensor[idx : idx + 1] for key, tensor in self.buffer.observations.items()}
        else:
            observations = self.buffer.observations[idx : idx + 1]

        if self.buffer.memorize_cache:
            # Need to reconstruct cache for this specific index
            env_idx = idx // self.buffer.buffer_size
            step_idx = idx % self.buffer.buffer_size
            # we need to shift the idx for masks since `self.mask_cache` is flattened as other tensors, but differently it's long `self.buffer_size+1`
            idx_shifted = ((idx // self.buffer.buffer_size) * (self.buffer.buffer_size+1)) + (idx % self.buffer.buffer_size)

            mem = {
                "memories": self.buffer._reconstruct_cache(env_idx, step_idx).unsqueeze(0),  # self.buffer.attn_cache[idx:idx + 1],
                "memory_mask": self.buffer.mask_cache[idx_shifted:idx_shifted + 1],
                "memory_indices": torch.empty(0)
            }
        else:
            # we need to shift the idx for flattened tensors long `self.buffer_size+1`
            idx_shifted = ((idx // self.buffer.buffer_size) * (self.buffer.buffer_size+1)) + (idx % self.buffer.buffer_size)
            selector_indices_batch_flattened = (((idx // self.buffer.buffer_size) * (self.buffer.buffer_size+1)).unsqueeze(-1) +
                                                self.buffer.memory_selector_indices_batch[idx_shifted:idx_shifted+1]).flatten()
            # for memories we gather `batch_size`x`memory_length` (in `get_last()` we gather only `1`x`memory_length`)
            mem = {
                "memories": torch.gather(self.buffer.memories, 0, repeat(selector_indices_batch_flattened, "... -> ... b h", b=self.buffer.num_blocks, h=self.buffer.embed_dim).long()).reshape(1, self.buffer.memory_length, self.buffer.num_blocks, self.buffer.embed_dim),
                "memory_mask": self.buffer.memory_mask_batch[idx_shifted:idx_shifted + 1],
                "memory_indices": self.buffer.memory_ordered_id_batch[idx_shifted:idx_shifted + 1]
            }
        return {"observations": observations,
                "actions": self.buffer.actions[idx:idx + 1],
                "old_values": self.buffer.values[idx:idx + 1].flatten(),
                "old_log_prob": self.buffer.log_probs[idx:idx + 1].flatten(),
                "old_mean": self.buffer.mean[idx:idx + 1],
                "old_std": self.buffer.std[idx:idx + 1],
                "advantages": self.buffer.advantages[idx:idx + 1].flatten(),
                "returns": self.buffer.returns[idx:idx + 1].flatten(),
                "action_mask": self.buffer.action_mask[idx:idx + 1],
                **mem,
                "mask": torch.ones_like(self.buffer.returns[idx:idx + 1].flatten()), }


class TransformerRolloutDataloader(TransformerRolloutBuffer):
    """Extension of TransformerRolloutBuffer with DataLoader support"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = None

    def get(self, batch_size: Optional[int] = None, shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
        """
        Returns a DataLoader for iterating over the buffer data

        :param batch_size: Size of each batch (defaults to buffer_size * n_envs if None)
        :param shuffle: Whether to shuffle the data
        :param pin_memory: Whether to pin memory in CPU (helpful for GPU training)
        :return: A PyTorch DataLoader instance
        """
        if not self.full and self.pos == 0:
            raise ValueError("Buffer is empty, cannot create DataLoader")

        if not self.generator_ready:
            # Create a dataset from this buffer
            dataset = TransformerRolloutDataset(self)  # makes the generator_ready
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size or dataset.length,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=pin_memory if self.cpu_offload else False,
                collate_fn=TransformerRolloutDataset.collate_fn,
            )
            if self.accelerator:
                self.dataloader = self.accelerator.prepare_data_loader(self.dataloader)

        # Use the custom collate function to handle our data structures
        return self.dataloader

    def reset(self) -> None:
        self.dataloader = None
        super().reset()
