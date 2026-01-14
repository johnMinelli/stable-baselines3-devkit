from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from algos.base.demonstration_algorithm import DemonstrationAlgorithm
from algos.policies.lstm_policy import LSTMPolicy
from algos.policies.mlp_policy import MlpPolicy
from algos.policies.tcn_policy import TCNPolicy
from common.callbacks import BaseCallback
from common.datasets import *  # noqa: F403
from common.envs import *  # noqa: F403
from common.inference_interface import InferenceInterface
from common.utils import create_spaces_from_cfg

SelfSL = TypeVar("SelfSL", bound="SL")


class SL(DemonstrationAlgorithm, InferenceInterface):
    """Generic implementation of supervised learning (SL) with policy-defined loss implementation"""
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {"MlpPolicy": MlpPolicy, "LSTMPolicy": LSTMPolicy, "TCNPolicy": TCNPolicy}  # **customize at necessity with other compatible policies**

    def __init__(
        self,
        policy: Optional[BasePolicy],
        env: Optional[Union[GymEnv, str]] = None,
        ds_input_shapes: Optional[Dict[str, Any]] = None,
        ds_output_shapes: Optional[Dict[str, Any]] = None,
        lr_value: Union[float, Schedule] = 3e-4,
        lr_scheduler: Optional[str] = None,
        lr_warmup_fraction: Optional[float] = None,
        lr_linear_end_value: Optional[float] = 0,
        lr_linear_end_fraction: Optional[float] = 0.05,
        batch_size: int = 32,
        n_epochs: int = 32,
        demonstrations_data_loader: Optional[torch.utils.data.DataLoader] = None,
        val_data_loader: Optional[torch.utils.data.DataLoader] = None,
        demonstrations: Optional[torch.utils.data.Dataset] = None,
        val_demonstrations: Optional[torch.utils.data.Dataset] = None,
        mse_coef: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = None,
        max_grad_norm: Optional[float] = 1.0,
        preprocessor_class: Union[Type[Preprocessor]] = None,  # noqa: F405
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        Builds SL.

        :param policy: A Stable Baselines3 policy.
        :param env: The environment to learn from (if registered in Gym, can be str).
            Defaults to None.
        :param ds_input_shapes: Dataset input space definition: mapping between input names to their expected
            shapes. Defaults to None.
        :param ds_output_shapes: Dataset output space definition: mapping between output names to their expected
            shapes. Defaults to None.
        :param lr_value: The learning rate, it can be a constant or a function
            of the current progress remaining (from 1 to 0). Defaults to 3e-4.
        :param lr_scheduler: Type of learning rate scheduler to use (e.g., 'constant', 'linear', 'cosine', ...). Defaults to None.
        :param lr_warmup_fraction: Fraction of total training steps to use for learning
            rate warmup. Defaults to None.
        :param lr_linear_end_value: The final value of the learning rate if using a
            linear decay. Defaults to 0.
        :param lr_linear_end_fraction: The fraction of total steps at which the linear
            decay should reach its end value. Defaults to 0.05.
        :param batch_size: The number of samples in each batch of expert data.
            Defaults to 32.
        :param n_epochs: The number of epochs to train for. Defaults to 32.
        :param demonstrations_data_loader: PyTorch DataLoader for training
            demonstrations. Defaults to None.
        :param val_data_loader: PyTorch DataLoader for validation
            demonstrations. Defaults to None.
        :param demonstrations: Demonstrations from an expert (optional). Transitions
            expressed directly as a `types.TransitionsMinimal` object, a sequence
            of trajectories, or a PyTorch Dataset. Defaults to None.
        :param val_demonstrations: Validation demonstrations (optional). Defaults to None.
        :param mse_coef: Scaling applied to the policy's MSE loss.
            Defaults to 1.0.
        :param gradient_accumulation_steps: Number of steps to accumulate gradients
            before performing an optimizer update. Defaults to 1.
        :param mixed_precision: Precision mode for training (e.g., 'fp16', 'bf16').
            Defaults to None.
        :param max_grad_norm: The maximum value for the gradient clipping.
            Defaults to 1.0.
        :param preprocessor_class: Class used to preprocess observations/actions before
            passing them to the policy. Defaults to None.
        :param preprocessor_kwargs: Keyword arguments for the preprocessor
            initialization. Defaults to None.
        :param tensorboard_log: The log location for TensorBoard (if None, no logging).
            Defaults to None.
        :param policy_kwargs: Arguments to be passed to the policy on creation. Defaults to None.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages. Defaults to 0.
        :param seed: Seed for the pseudo random generators. Defaults to None.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether to build the network at the creation of the instance. Defaults to True.
        """
        self.demonstrations_data_loader: Optional[Iterable] = demonstrations_data_loader
        self.val_data_loader: Optional[Iterable] = val_data_loader
        self.batch_size = batch_size
        self.num_envs = batch_size
        self.n_epochs = n_epochs
        self.mse_coef = mse_coef
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.preprocessor_class = preprocessor_class
        self.preprocessor_kwargs = preprocessor_kwargs
        self.ds_input_shapes = ds_input_shapes
        self.ds_output_shapes = ds_output_shapes

        super().__init__(
            policy=policy,
            env=env,
            lr_value=lr_value,
            lr_scheduler=lr_scheduler,
            lr_warmup_fraction=lr_warmup_fraction,
            lr_linear_end_value=lr_linear_end_value,
            lr_linear_end_fraction=lr_linear_end_fraction,
            demonstrations=demonstrations,
            val_demonstrations=val_demonstrations,
            n_epochs=n_epochs,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        if self.device.type == torch.device("cuda").type:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.demonstrations_data_loader is None and self.demonstrations is not None:
            self.set_demonstrations(self.demonstrations, self.val_demonstrations)

        ds_observation_space, ds_action_space = create_spaces_from_cfg(self.ds_input_shapes, self.ds_output_shapes)

        if self.preprocessor_class is not None:
            self.preprocessor = (
                eval(self.preprocessor_class) if isinstance(self.preprocessor_class, str) else self.preprocessor_class)(
                observation_space=self.env.observation_space if self.env is not None else ds_observation_space,
                action_space=self.env.action_space if self.env is not None else ds_action_space, device=self.device, **self.preprocessor_kwargs)
            self.observation_space = self.preprocessor.proc_observation_space
            self.action_space = self.preprocessor.proc_action_space
        else:
            self.observation_space = ds_observation_space
            self.action_space = ds_action_space
        self.set_random_seed(self.seed)  # after action_space definition

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)

        # Setup Accelerate support
        assert self.batch_size%self.gradient_accumulation_steps == 0 or self.batch_size/self.gradient_accumulation_steps<1, "The batch size should be divisible by the number of gradient accumulation steps."
        self.accelerator = Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(sync_with_dataloader=self.batch_size/self.gradient_accumulation_steps>=1, num_steps=self.gradient_accumulation_steps),
            mixed_precision=self.mixed_precision,
            cpu=torch.device("cpu") == self.device,
        )
        self.policy, self.demonstrations_data_loader, self.policy.optimizer = self.accelerator.prepare(self.policy, self.demonstrations_data_loader, self.policy.optimizer)

    def process_obs(self, *args, **kwargs):
        return self.preprocessor(*args, **kwargs) if self.preprocessor else (*args, *kwargs)

    def process_out_actions(self, *args, **kwargs):
        return self.preprocessor.forward_post(*args, **kwargs) if self.preprocessor else (*args, *kwargs)

    def predict_rollout(
        self, env: VecEnv, callback: Optional[BaseCallback], n_episodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect a single rollout for prediction/evaluation.

        :param env: The evaluation environment
        :param callback: Callback that will be called at each step
        :param n_episodes: number of episodes to execute
        :return: Tuple of (episode_reward, episode_length)
        """
        last_obs = self.process_obs(observations=env.reset())
        if callback is not None:
            callback.on_rollout_start()

        # Initialize tracking variables for each environment
        num_envs = env.num_envs
        episodes_completed = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        sum_rewards = torch.zeros(num_envs, device=self.device)
        sum_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        successes = torch.zeros(num_envs, device=self.device)
        current_rewards = torch.zeros(num_envs, device=self.device)
        current_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        last_episode_starts = torch.zeros((num_envs, 1), device=self.device)

        while not (episodes_completed >= n_episodes).all():

            with torch.no_grad():
                actions = self.policy.predict_seq(last_obs) if hasattr(self.policy, "predict_seq") else self.policy.predict(last_obs)
                actions = self.process_out_actions(actions)

            obs, rewards, dones, infos = env.step(actions)
            obs = self.process_obs(observations=obs)

            # Update episode tracking for each environment
            current_rewards += rewards
            current_lengths += 1

            # Update cumulative statistics for completed episodes
            completed_mask = dones & (episodes_completed < n_episodes)
            sum_rewards += completed_mask * current_rewards
            sum_lengths += completed_mask * current_lengths
            successes += dones * infos["success"].to(dones.device)
            episodes_completed += completed_mask

            if torch.any(dones) and hasattr(self.policy, "reset"):
                self.policy.reset()

            # Reset current episode trackers where episodes are done
            current_rewards *= ~dones
            current_lengths *= ~dones

            if callback is not None:
                callback.update_locals(locals())
                if not callback.on_step():
                    break

            last_obs = obs
            last_episode_starts = dones

        mean_rewards = (sum_rewards / n_episodes).cpu().mean()
        mean_lengths = (sum_lengths.float() / n_episodes).cpu().mean()
        mean_success = (successes / n_episodes).cpu().mean()

        return mean_success, mean_rewards, mean_lengths

    def train(self, batch) -> None:
        """Train on a single batch of demonstrations.

        :param batch: Dictionary containing 'obs' and 'acts' keys.
        """
        self.logger.clear()
        self.logger.start_timer("s/batch")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        with self.accelerator.accumulate(self.policy):
            obs, actions = self.process_obs(observations=batch["obs"], actions=batch["actions"])

            # Run the batch and compute loss
            if hasattr(self.policy, "compute_loss"):  # delegate it to the policy
                loss_dict = self.policy.compute_loss(obs, actions)
            else:
                out = self.policy(obs)
                assert isinstance(out, torch.Tensor), "The policy is expected to return a torch.Tensor."
                loss_dict = {"mse_loss": self.mse_coef * F.mse_loss(out, actions)}

            self.accelerator.backward(sum(loss_dict.values()))

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad()
                self._n_updates += 1

        # Logging
        self.logger.record("train/n_updates", self._n_updates)
        for k, v in loss_dict.items():
            self.logger.record(f"train/{k}", float(v) if v is not None else None)

        self.logger.stop_timer("s/batch", div=len(batch["obs"]))

    def test(self, batch, init_logs=True) -> None:
        """Train on a single batch of demonstrations.

        :param batch: Dictionary containing 'obs' and 'acts' keys.
        :param init_logs: Reset logger stats, defaults to True.
        """
        if init_logs:
            self.logger.clear()

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        obs, actions = self.process_obs(observations=batch["obs"], actions=batch["actions"])
        # Run the batch and compute loss (by default we compute here the loss, but by necessity we delegated it to the policy)
        with torch.no_grad():
            loss_dict = self.policy.compute_loss(obs, actions)
        # Logging
        for k, v in loss_dict.items():
            self.logger.record(f"test/{k}", float(v) if v is not None else None)

    def learn(
        self: SelfSL,
        total_timesteps: int = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        val_interval: int = 100,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            val_interval=val_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer", "preprocessor"]

        return state_dicts, []

    def _excluded_save_params(self) -> List[str]:
        return ["demonstrations_data_loader", "val_data_loader", "demonstrations", "observation_space", "action_space"]+super()._excluded_save_params()
