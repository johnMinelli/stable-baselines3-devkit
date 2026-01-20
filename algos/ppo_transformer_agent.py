from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from algos.base.onpolicy_algorithm import OnPolicyAlgorithm
from algos.policies.ac_transformer_policy import ACTransformerPolicy
from algos.storage.buffers import RolloutBuffer, TransformerRolloutBuffer
from common.callbacks import BaseCallback
from common.envs import *  # noqa: F403, F405
from common.inference_interface import InferenceInterface

SelfTransformerPPO = TypeVar("SelfTransformerPPO", bound="TransformerPPO")


class TransformerPPO(OnPolicyAlgorithm, InferenceInterface):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for Transformer policies.

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "TransformerPolicy": ACTransformerPolicy,
    }  # **customize at necessity with other compatible policies**

    def __init__(
        self,
        policy: Union[str, Type[ACTransformerPolicy]],
        env: Union[GymEnv, str],
        lr_value: Union[float, Schedule] = 1e-4,
        lr_scheduler: Optional[str] = None,
        lr_warmup_fraction: Optional[float] = None,
        lr_linear_end_value: Optional[float] = 0,
        lr_linear_end_fraction: Optional[float] = 0.05,
        n_steps: int = 128,
        batch_size: int = None,
        num_mini_batch: int = 1,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage_per_mini_batch: bool = False,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: Optional[str] = None,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        preprocessor_class: Type[Preprocessor] = None,  # noqa: F405
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """
        Build PPO with Transformer Support.

        :param policy: The Transformer policy model to use (e.g. ACTransformerPolicy).
        :param env: The environment to learn from (if registered in Gym, can be str).
        :param lr_value: The learning rate, it can be a constant or a function
            of the current progress remaining (from 1 to 0). Defaults to 1e-4.
        :param lr_scheduler: Type of learning rate scheduler to use (e.g., 'constant', 'linear', 'cosine', ...). Defaults to None.
        :param lr_warmup_fraction: Fraction of total training steps to use for learning
            rate warmup. Defaults to None.
        :param lr_linear_end_value: The final value of the learning rate if using a
            linear decay. Defaults to 0.
        :param lr_linear_end_fraction: The fraction of total steps at which the linear
            decay should reach its end value. Defaults to 0.05.
        :param n_steps: The number of steps to run for each environment per update.
            (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel). Note: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
            See https://github.com/pytorch/pytorch/issues/29372. Defaults to 128.
        :param batch_size: Minibatch size. Defaults to None (will be calculated
            based on n_steps and num_mini_batch).
        :param num_mini_batch: Number of minibatches to split the rollout data into.
            Defaults to 1.
        :param n_epochs: Number of epochs when optimizing the surrogate loss.
            Defaults to 10.
        :param gamma: Discount factor. Defaults to 0.99.
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized
            Advantage Estimator. Defaults to 0.95.
        :param clip_range: Clipping parameter, it can be a function of the current
            progress remaining. Defaults to 0.2.
        :param clip_range_vf: Clipping parameter for the value function,
            it can be a function of the current progress remaining (from 1 to 0).
            This is a parameter specific to the OpenAI implementation. If None is passed (default),
            no clipping will be done on the value function.
            IMPORTANT: this clipping depends on the reward scaling.
        :param normalize_advantage_per_mini_batch: Whether to normalize the advantage
            per mini-batch or per rollout. Defaults to False.
        :param ent_coef: Entropy coefficient for the loss calculation. Defaults to 0.01.
        :param vf_coef: Value function coefficient for the loss calculation.
            Defaults to 0.5.
        :param max_grad_norm: The maximum value for the gradient clipping.
            Defaults to 0.5.
        :param use_sde: Whether to use State Dependent Exploration (gSDE)
            instead of action noise. Defaults to False.
        :param sde_sample_freq: When `use_sde` is True, sample a new noise matrix every n steps.
            Defaults to -1 (only sample at the beginning of the rollout).
        :param target_kl: Limit the KL divergence between updates. By default,
            there is no limit on the KL divergence, if specified can it can be used to stop the updates on current rollout
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            (commented out as less effective) or adapt the learning rate. Defaults to None.
        :param gradient_accumulation_steps: Number of steps to accumulate gradients
            before an optimizer update. Defaults to 1.
        :param mixed_precision: Precision mode for training (e.g., 'fp16', 'bf16').
            Defaults to None.
        :param rollout_buffer_class: Transformer-specific rollout buffer class.
            If None, uses TransformerRolloutBuffer. Defaults to None.
        :param rollout_buffer_kwargs: Arguments to be passed to the rollout buffer
            on creation. Defaults to None.
        :param preprocessor_class: Class used to preprocess observations/actions.
            Defaults to None.
        :param preprocessor_kwargs: Keyword arguments for the preprocessor.
            Defaults to None.
        :param stats_window_size: Window size for rollout logging (average over n
            episodes). Defaults to 100.
        :param tensorboard_log: The log location for TensorBoard. Defaults to None.
        :param policy_kwargs: Arguments to be passed to the policy on creation. Defaults to None.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages. Defaults to 0.
        :param seed: Seed for the pseudo random generators. Defaults to None.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether to build the network at the creation of the instance. Defaults to False.
        """
        super().__init__(
            policy,
            env,
            lr_value=lr_value,
            lr_scheduler=lr_scheduler,
            lr_warmup_fraction=lr_warmup_fraction,
            lr_linear_end_value=lr_linear_end_value,
            lr_linear_end_fraction=lr_linear_end_fraction,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.num_mini_batch = num_mini_batch
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.target_kl = target_kl
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.preprocessor_class = preprocessor_class
        self.preprocessor_kwargs = preprocessor_kwargs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        if self.device.type == torch.device("cuda").type:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.preprocessor_class is not None:
            self.preprocessor = (
                eval(self.preprocessor_class) if isinstance(self.preprocessor_class, str) else self.preprocessor_class)(
                observation_space=self.observation_space, action_space=self.action_space, device=self.device, **self.preprocessor_kwargs)
            self.observation_space = self.preprocessor.proc_observation_space
            self.action_space = self.preprocessor.proc_action_space

        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        ).to(self.device)

        self.rollout_buffer = (
            eval(self.rollout_buffer_class) if isinstance(self.rollout_buffer_class, str) else self.rollout_buffer_class)(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        batch_size = self.batch_size if self.batch_size is not None else self.rollout_buffer.buffer_size * self.n_envs
        # Setup Accelerate support
        assert batch_size%self.gradient_accumulation_steps == 0 or batch_size/self.gradient_accumulation_steps<1, "The batch size should be divisible by the number of gradient accumulation steps."
        self.accelerator = Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(sync_with_dataloader=batch_size/self.gradient_accumulation_steps>=1, num_steps=self.gradient_accumulation_steps),
            mixed_precision=self.mixed_precision,
            cpu=torch.device("cpu") == self.device,
        )
        self.policy, self.policy.optimizer = self.accelerator.prepare(self.policy, self.policy.optimizer)
        # self.accelerator.dataloader_config.non_blocking = True
        self.rollout_buffer.set_accelerator(self.accelerator)
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

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
        :returns: Tuple of (episode_reward, episode_length)
        """
        rollout_buffer = eval(self.rollout_buffer_class)(  # type: ignore[assignment]
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        rollout_buffer.reset()
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

        with torch.no_grad():
            while not (episodes_completed >= n_episodes).all():
                last_memory = rollout_buffer.get_last()
                actions, memory, mask = self.policy.predict(last_obs, *last_memory)

                obs, rewards, dones, infos = env.step(self.process_out_actions(actions))
                obs = self.process_obs(observations=obs)

                # Update episode tracking for each environment
                current_rewards += rewards
                current_lengths += 1

                # Update cumulative statistics for completed episodes
                completed_mask = dones & (episodes_completed < n_episodes)
                sum_rewards += completed_mask * current_rewards
                sum_lengths += completed_mask * current_lengths
                episodes_completed += completed_mask
                successes += dones & ~infos.get("truncated", torch.zeros_like(dones))

                # Reset current episode trackers where episodes are done
                current_rewards *= ~dones
                current_lengths *= ~dones

                if callback is not None:
                    callback.update_locals(locals())
                    if not callback.on_step():
                        break

                rollout_buffer.add(
                    action_mask=torch.ones_like(actions),
                    memory=memory,
                    mask=mask,
                )

                if torch.any(dones):
                    rollout_buffer.reset()

                last_obs = obs
                last_episode_starts = dones

        mean_success = (successes / n_episodes).cpu().mean()
        mean_rewards = (sum_rewards / n_episodes).cpu().mean()
        mean_lengths = (sum_lengths.float() / n_episodes).cpu().mean()

        return mean_success, mean_rewards, mean_lengths

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, TransformerRolloutBuffer
        ), f"{rollout_buffer} doesn't support transformer policy"
        assert self._last_obs is not None, "No previous observation was provided"
        if not isinstance(self._last_episode_starts, torch.Tensor):  # first rollout
            self._last_obs = self.process_obs(observations=self._last_obs)
            self._last_episode_starts = torch.tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
        else:
            self._last_episode_starts = self._last_episode_starts.clone().to(dtype=torch.float32)

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        self.logger.start_timer("s/rollout")

        while n_steps < n_rollout_steps:

            with torch.no_grad():
                last_memory = rollout_buffer.get_last()
                actions, values, distribution_output, memory, mask = self.policy(self._last_obs, *last_memory)
                log_probs, mean, std = distribution_output

            obs, rewards, dones, infos = env.step(self.process_out_actions(actions))
            obs = self.process_obs(observations=obs)
            self.logger.record_mean("rollout/mean_rew", rewards.mean().item())

            # Post-step operations
            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # Handle timeout by bootstrapping with value function see GitHub issue #633
            terminal_mask = dones & infos.get("truncated", torch.zeros_like(dones))

            if terminal_mask.any():
                # For timeouts, use the current observation (obs) which is the same as the terminal observation
                terminal_obs = {k: v if isinstance(v, str) else v[terminal_mask] if isinstance(v, torch.Tensor) else {k2: v2[terminal_mask] for k2, v2 in v.items()} for k, v in obs.items()} \
                    if isinstance(obs, dict) else obs[terminal_mask]
                with torch.no_grad():  # Note: close an eye on the fact that memories (t) and observations (t+1) are not temporally aligned.
                    terminal_values = self.policy.predict_values(
                        terminal_obs,
                        *[mem[terminal_mask] if mem is not None else None for mem in last_memory]
                    )  # type: ignore[arg-type]
                # Add discounted terminal values to rewards
                rewards[terminal_mask] += self.gamma * terminal_values

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                mean,
                std,
                torch.ones_like(actions, dtype=torch.bool),
                memory,
                mask,
            )
            rollout_buffer.restart_memory(dones)

            # Go on with next step
            self._last_obs = obs
            self._last_episode_starts = dones
            n_steps += 1

        self.logger.stop_timer("s/rollout", n_rollout_steps)

        # Compute value for the last timestep
        with torch.no_grad():
            values = self.policy.predict_values(obs, *rollout_buffer.get_last())

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.logger.clear()
        self.logger.start_timer("s/epoch")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        continue_training = True
        # train for n_epochs epochs
        batch_size = self.batch_size if self.batch_size is not None else (self.rollout_buffer.buffer_size * self.n_envs)
        for epoch in range(self.n_epochs):
            if not continue_training: break
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(batch_size//self.num_mini_batch):
                with self.accelerator.accumulate(self.policy):
                    actions = rollout_data.actions
                    advantages = rollout_data.advantages
                    rollout_memories = ([rollout_data.memories, rollout_data.memory_mask]+[rollout_data.memory_indices] if hasattr(rollout_data, "memory_indices") else [])

                    _, values, dist_out, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                        *rollout_memories
                    )
                    log_prob, new_mean, new_std = dist_out

                    # Normalize advantage; ignore if `mini_batch_size` == 1, see GH issue #325
                    if self.normalize_advantage_per_mini_batch and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # clipped surrogate loss by ratio between old and new policy (should be one at the first iteration)
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = -advantages * ratio
                    policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = torch.mean(torch.max(policy_loss_1, policy_loss_2))

                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values, reduction="none")
                    if self.clip_range_vf is not None:  # No clipping
                        value_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                        value_loss_clipped = F.mse_loss(rollout_data.returns, value_pred, reduction="none")
                        value_loss = torch.max(value_loss, value_loss_clipped)
                    value_loss = torch.mean(value_loss)

                    entropy_loss = torch.mean(entropy) if entropy is not None else torch.mean(log_prob)

                    # Full loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                    # Compute TRUE KL for adaptive learning rate
                    with torch.no_grad():
                        kl = torch.sum(torch.log(new_std / (rollout_data.old_std + 1e-5) + 1e-5) +
                                       (rollout_data.old_std.pow(2) + (rollout_data.old_mean - new_mean).pow(2)) /
                                       (2 * new_std.pow(2) + 1e-8) - 0.5, dim=-1).mean()
                    if self.target_kl is not None:  # noqa: SIM102
                        if self.lr_scheduler == "adaptive":
                            if kl > self.target_kl * 2.0:
                                self.lr_value = max(5e-6, self.lr_value / 1.5)
                            elif (kl < self.target_kl / 2.0) and (kl > 0.0):
                                self.lr_value = min(1e-4, self.lr_value * 1.5)
                            self._update_learning_rate(self.policy.optimizer)

                        # if kl > 1.5 * self.target_kl:
                        #     continue_training = False
                        #     if self.verbose >= 1:
                        #         print(f"Early stopping at step {epoch} due to reaching max kl: {kl:.2f}")
                        #     break

                    # Optimization step
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:  # note: consider that if 'sync on datalaoder end' is disabled we risk to leave grad not updated
                        self.accelerator.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                        self.policy.optimizer.step()
                        self.policy.optimizer.zero_grad()
                        # Clamp log_std after optimizer step
                        if hasattr(self.policy, "log_std"):
                            self.policy.log_std.data = self.policy.log_std.data.clamp(min=-20.0, max=2.0)
                        self._n_updates += 1

                        # Logging
                        clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                        self.logger.record_mean("train/clip_fraction", clip_fraction)
                        self.logger.record_mean("train/kl", kl.item())  # should be the mean of last epoch
                        self.logger.record_mean("train/policy_gradient_loss", policy_loss.item())
                        self.logger.record_mean("train/entropy_loss", entropy_loss.item())
                        self.logger.record_mean("train/value_loss", value_loss.item())
                        self.logger.record_mean("train/loss", loss.item())  # should be the value of last epoch

                        if not continue_training:
                            break

        # Logging
        self.logger.stop_timer("s/epoch", div=self.n_epochs)
        self.logger.record("train/n_updates", self._n_updates)
        explained_var = float(
            1 - torch.var(self.rollout_buffer.values.flatten() - self.rollout_buffer.returns.flatten()) / (torch.var(self.rollout_buffer.returns.flatten()) + 1e-8)
        )
        self.logger.record_mean("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record_mean("train/std", torch.exp(self.policy.log_std).mean().item())
        if hasattr(self.policy, "std"):
            self.logger.record_mean("train/std", self.policy.std.mean().item())

    def learn(
        self: SelfTransformerPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TransformerPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTransformerPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return ["observation_space", "action_space"] + super()._excluded_save_params()  # noqa: RUF005
