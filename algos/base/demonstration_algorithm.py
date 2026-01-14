"""Module of base classes and helper methods for imitation learning algorithms. Modified from Imitation library"""

import sys
import time
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import torch as th
import torch.utils.data as th_data
from gymnasium import spaces
from imitation.data import rollout, types
from imitation.util import util
from stable_baselines3.common import policies
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean

from algos.base.base_algorithm import BaseAlgorithm


class FixedHorizonMixin:
    """Mixin class to handle fixed/variable horizon checking functionality."""

    def __init__(self, allow_variable_horizon: bool = False):
        """
        Initialize the FixedHorizonMixin.

        :param  allow_variable_horizon: If True, allows episodes of different lengths.
                                  If False (default), raises error for variable horizons.
        """
        self.allow_variable_horizon = allow_variable_horizon
        self._horizon = None

        if allow_variable_horizon:
            self.logger.warn(
                "Running with `allow_variable_horizon` set to True. "
                "Some algorithms are biased towards shorter or longer "
                "episodes, which may significantly confound results. "
                "Additionally, even unbiased algorithms can exploit "
                "the information leak from the termination condition, "
                "producing spuriously high performance. See "
                "https://imitation.readthedocs.io/en/latest/getting-started/"
                "variable-horizon.html for more information.",
            )

    def _check_fixed_horizon(self, horizons: Iterable[int]) -> None:
        """
        Checks that episode lengths in `horizons` are fixed and equal to prior calls.

        If algorithm is safe to use with variable horizon episodes (e.g. behavioral
        cloning), then just don't call this method.

        :param horizons: An iterable sequence of episode lengths.
        :raises ValueError: The length of trajectories in trajs differs from one
                another, or from trajectory lengths in previous calls to this method.
        """
        if self.allow_variable_horizon:  # skip check
            return

        # horizons = all horizons seen so far (including trajs)
        horizons = set(horizons)
        if self._horizon is not None:
            horizons.add(self._horizon)

        if len(horizons) == 1:
            self._horizon = horizons.pop()
        elif len(horizons) > 1:
            raise ValueError(
                f"Episodes of different length detected: {horizons}. "
                "Variable horizon environments are discouraged -- "
                "termination conditions leak information about reward. See "
                "https://imitation.readthedocs.io/en/latest/getting-started/"
                "variable-horizon.html for more information. "
                "If you are SURE you want to run imitation on a "
                "variable horizon task, then please pass in the flag: "
                "`allow_variable_horizon=True`.",
            )


class DemonstrationAlgorithm(BaseAlgorithm, FixedHorizonMixin):
    """An algorithm that learns from demonstration: BC, IRL, etc."""

    def __init__(
        self,
        policy: Union[str, Type[policies.BasePolicy]],
        env: Union[GymEnv, str, None],
        lr_value: Union[float, Schedule],
        n_epochs: int,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        monitor_wrapper: bool = True,
        lr_scheduler: Optional[str] = None,
        lr_warmup_fraction: Optional[float] = None,
        lr_linear_end_value: Optional[float] = 0,
        lr_linear_end_fraction: Optional[float] = 0.05,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        demonstrations: Optional[th.utils.data.Dataset] = None,
        val_demonstrations: Optional[th.utils.data.Dataset] = None,
        allow_variable_horizon: bool = False,
        _init_setup_model: bool = True,
    ):
        """
        Creates an algorithm that learns from demonstrations.

        :param n_epochs: Number of epochs to execute learning on the given dataset.
        :param demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
        :param val_demonstrations: Set of demonstrations for validation (optional).
        :param allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check.
        :param _init_setup_model: Whether to build the network at the creation of the instance
        """
        BaseAlgorithm.__init__(
            self,
            policy=policy,
            env=env,
            lr_value=lr_value,
            lr_scheduler=lr_scheduler,
            lr_warmup_fraction=lr_warmup_fraction,
            lr_linear_end_value=lr_linear_end_value,
            lr_linear_end_fraction=lr_linear_end_fraction,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        FixedHorizonMixin.__init__(self, allow_variable_horizon)

        self.demonstrations = demonstrations
        self.val_demonstrations = val_demonstrations
        self.n_epochs = n_epochs

        if demonstrations is not None:
            self.set_demonstrations(demonstrations, val_demonstrations)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """
        Create the policy network and setup learning rate schedule.

        This method initializes the policy using the specified policy class,
        sets up the learning rate schedule, and moves the policy to the device.
        """
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if not hasattr(self, "demonstrations_data_loader") and self.demonstrations:
            self.set_demonstrations(self.demonstrations, self.val_demonstrations)

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )

        self.policy = self.policy.to(self.device)

    def __getstate__(self) -> Dict:
        """
        Prepare object state for pickling by removing unpicklable logger.

        :returns: Object state dictionary without the logger.
        """
        state = self.__dict__.copy()
        # logger can't be pickled as it depends on open files
        del state["_logger"]
        return state

    def __setstate__(self, state):
        """
        Restore object state from pickle.

        :param state: The state dictionary to restore
        """
        self.__dict__.update(state)

    def set_demonstrations(
        self, demonstrations: th.utils.data.Dataset, val_demonstrations: th.utils.data.Dataset = None
    ) -> None:
        """
        Set the demonstration datasets for training and validation.

        :param demonstrations: Training demonstration dataset
        :param val_demonstrations: Optional validation demonstration dataset
        """
        self.demonstrations_data_loader = make_data_loader(demonstrations, self.batch_size, self.seed)
        if val_demonstrations is not None:
            self.val_data_loader = make_data_loader(val_demonstrations, self.batch_size, self.seed)

    def train(self, batch) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def test(self, batch, init_logs: bool = False) -> None:
        """
        Consume current rollout data.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            [
                self.logger.record(k, safe_mean([ep_info[k] for ep_info in self.ep_info_buffer]))
                for k in self.ep_info_buffer[0].keys()
                if k not in ["r", "l"]
            ]
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int = None,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        val_interval: int = 100,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Train the model using demonstration data for the specified number of timesteps.

        :param total_timesteps: Total number of timesteps to train for
        :param callback: Callback(s) called at every step with state of the algorithm
        :param log_interval: Number of iterations between logging training metrics
        :param val_interval: Number of iterations between validation runs
        :param tb_log_name: Name for tensorboard logging
        :param reset_num_timesteps: Whether to reset the timestep counter
        :param progress_bar: Whether to display a progress bar
        :return: The trained model instance
        """
        iteration = 0
        n_samples = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
        self._update_learning_rate(self.policy.optimizer)

        callback.on_training_start(locals(), globals())

        assert self.demonstrations_data_loader is not None, "`demonstrations` must not be None."

        self.policy.train()
        for epoch in range(self.n_epochs):
            for batch in self.demonstrations_data_loader:
                batch_samples = len(batch["obs"]["state"])
                batch_description = f"Epoch {epoch}/{self.n_epochs}"

                if self.num_timesteps >= total_timesteps:  # break
                    return self

                # Train on this batch
                metrics = self.train(batch)
                self.num_timesteps += batch_samples
                n_samples += batch_samples

                callback.update_locals(locals())
                if not callback.on_step():
                    break

                # Display training infos
                if iteration % log_interval == 0:
                    self._dump_logs(iteration)

                # Validation step
                if val_interval > 0 and iteration != 0 and iteration % val_interval == 0:
                    for i, batch in enumerate(self.val_data_loader):
                        self.test(batch, init_logs=i == 0)
                    self._dump_logs(iteration)

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                self._update_learning_rate(self.policy.optimizer)

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


class _WrappedDataLoader:
    """Wraps a data loader (batch iterable) and checks for specified batch size."""

    def __init__(
        self,
        data_loader: Iterable[th.utils.data.DataLoader],
        expected_batch_size: int,
    ):
        """
        Builds _WrappedDataLoader.

        :param data_loader: The data loader (batch iterable) to wrap.
        :param expected_batch_size: The batch size to check for.
        """
        self.data_loader = data_loader
        self.expected_batch_size = expected_batch_size

    def __iter__(self) -> Iterator[types.TransitionMapping]:
        """
        Yields data from `self.data_loader`, checking `self.expected_batch_size`.

        :yields: Identity -- yields same batches as from `self.data_loader`.
        :raise ValueError: `self.data_loader` returns a batch of size not equal to
                `self.expected_batch_size`.
        """
        for batch in self.data_loader:
            if len(batch["obs"]["state"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} " f"!= {len(batch['obs'])} = len(batch['obs'])",
                )
            if len(batch["actions"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} " f"!= {len(batch['acts'])} = len(batch['acts'])",
                )
            yield batch


def make_data_loader(
    transitions: th.utils.data.Dataset,
    batch_size: int,
    seed: Optional[int] = None,
    data_loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[types.TransitionMapping]:
    """Converts demonstration data to Torch data loader.

    :param transitions: Transitions expressed directly as a `types.TransitionsMinimal`
            object, a sequence of trajectories, or an iterable of transition
            batches (mappings from keywords to arrays containing observations, etc).
    :param batch_size: The size of the batch to create. Does not change the batch size
            if `transitions` is already an iterable of transition batches.
    :param seed: Seed for the pseudo random generators.
    :param data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.
    :returns: An iterable of transition batches.
    :raises ValueError: if `transitions` is an iterable over transition batches with batch
            size not equal to `batch_size`; or if `transitions` is transitions or a
            sequence of trajectories with total timesteps less than `batch_size`.
    :raises TypeError: if `transitions` is an unsupported type.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive.")

    if isinstance(transitions, Iterable):
        # In case of episodes rollouts
        first_item, transitions = util.get_first_iter_element(transitions)
        if isinstance(first_item, types.Trajectory):
            transitions = cast(Iterable[types.Trajectory], transitions)
            transitions = rollout.flatten_trajectories(list(transitions))

    if isinstance(transitions, th.utils.data.Dataset):
        if len(transitions) < batch_size:
            raise ValueError(
                f"Number of transitions in `demonstrations` {len(transitions)} "
                f"is smaller than batch size {batch_size}.",
            )

        generator = th.Generator()
        generator.manual_seed(seed)

        kwargs: Mapping[str, Any] = {
            "shuffle": True,
            "drop_last": True,
            "generator": generator,
            **(data_loader_kwargs or {}),
        }
        return th_data.DataLoader(transitions, batch_size=batch_size, **kwargs)
    if isinstance(transitions, Iterable):
        # Safe to ignore this error since we've already converted Iterable[Trajectory]
        # `transitions` into Iterable[TransitionMapping]
        return _WrappedDataLoader(transitions, batch_size)  # type: ignore[arg-type]
    raise TypeError(f"`demonstrations` unexpected type {type(transitions)}")
