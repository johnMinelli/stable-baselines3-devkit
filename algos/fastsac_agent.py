from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from torch.nn import functional as F

from algos.base.offpolicy_algorithm import OffPolicyAlgorithm
from algos.policies.fastsac_policy import FastSACActor, FastSACCritic, FastSACPolicy
from algos.storage.buffers import ReplayBuffer
from common.callbacks import BaseCallback
from common.envs import *  # noqa: F403, F405
from common.inference_interface import InferenceInterface
from common.utils import get_parameters_by_name

SelfFastSAC = TypeVar("SelfFastSAC", bound="FastSAC")


class FastSAC(OffPolicyAlgorithm, InferenceInterface):
    """
    Soft Actor-Critic with Distributional Critics (FastSAC)

    Paper: https://arxiv.org/abs/2512.01996v1; https://arxiv.org/abs/2505.22642
    Webpage with Code: https://younggyo.me/fastsac-humanoid/

    Key differences from standard SAC:
    - Critic outputs num_atoms logits (distribution over returns) instead of single Q-value
    - Uses categorical cross-entropy loss instead of MSE
    - Network architecture: SiLU activation, LayerNorm, decreasing dimensions
    - Policy frequency: don't update actor every gradient step
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": FastSACPolicy,
    }

    policy: FastSACPolicy
    actor: FastSACActor
    critic: FastSACCritic
    critic_target: FastSACCritic

    def __init__(
        self,
        policy: Union[str, type[FastSACPolicy]],
        env: Union[GymEnv, str],
        lr_value: Union[float, Schedule] = 3e-4,
        lr_scheduler: Optional[str] = None,
        lr_warmup_fraction: Optional[float] = None,
        lr_linear_end_value: Optional[float] = 1e-8,
        lr_linear_end_fraction: Optional[float] = 0.005,
        batch_size: int = 8192,
        gamma: float = 0.97,
        tau: float = 0.125,
        learning_starts: int = 10,
        train_freq: int = (1, "step"),
        gradient_steps: int = 8,
        action_noise=None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        target_entropy_ratio: float = 0.0,
        policy_frequency: int = 4,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        mixed_precision: Optional[str] = None,
        max_grad_norm: float = 0.0,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        replay_buffer_checkpoint: Optional[Union[str, Path]] = None,
        preprocessor_class: Type[GymPreprocessor] = None,
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
        Build FastSAC.

        :param policy: The policy model to use (MlpPolicy).
        :param env: The environment to learn from (if registered in Gym, can be str).
        :param lr_value: The learning rate, it can be a constant or a function
            of the current progress remaining (from 1 to 0). Defaults to 3e-4.
        :param lr_scheduler: Type of learning rate scheduler to use (e.g., 'constant', 'linear', 'cosine', ...). Defaults to None.
        :param lr_warmup_fraction: Fraction of total training steps to use for learning
            rate warmup. Defaults to None.
        :param lr_linear_end_value: The final value of the learning rate if using a
            linear decay. Defaults to 1e-8.
        :param lr_linear_end_fraction: The fraction of total steps at which the linear
            decay should reach its end value. Defaults to 0.005.
        :param batch_size: Minibatch size for each gradient update. Defaults to 8192.
        :param gamma: The discount factor. Defaults to 0.97.
        :param tau: The soft update coefficient ("Polyak update", between 0 and 1). Defaults to 0.125.
        :param learning_starts: How many steps of the model to collect transitions for before learning starts. Defaults to 10000.
        :param train_freq: Update the model every ``train_freq`` steps. Defaults to (1, "step").
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``).
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout. Defaults to 8.
        :param action_noise: The action noise type (None by default), this can help
            for hard exploration problem. Cf common.noise for the different action noise types. Defaults to None.
        :param ent_coef: Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.) Controlling exploration/exploitation trade-off.
            Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value). Defaults to "auto".
        :param target_update_interval: Update the target network every ``target_update_interval`` gradient steps. Defaults to 1.
        :param target_entropy: Target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``).
        :param target_entropy_ratio: Ratio to scale target entropy (0.0 = -action_dim). Defaults to 0.
        :param policy_frequency: Update actor every policy_frequency gradient steps. Defaults to 4.
        :param use_sde: Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration. Defaults to False.
        :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE.
            Default: -1 (only sample at the beginning of the rollout). Defaults to -1.
        :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling during the warm up phase (before learning starts). Defaults to False.
        :param mixed_precision: Precision mode for training (e.g., 'fp16', 'bf16'). Defaults to None.
        :param max_grad_norm: Maximum gradient norm (0 = no clipping). Defaults to 0.
        :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected. Defaults to None.
        :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation. Defaults to None.
        :param replay_buffer_checkpoint: Path locating the checkpoint file of the replay buffer to load. Defaults to None.
        :param preprocessor_class: Class used to preprocess observations/actions. Defaults to None.
        :param preprocessor_kwargs: Keyword arguments for the preprocessor. Defaults to None.
        :param stats_window_size: Window size for rollout logging (average over n episodes). Defaults to 100.
        :param tensorboard_log: The log location for TensorBoard. Defaults to None.
        :param policy_kwargs: Arguments to be passed to the policy on creation. Defaults to None.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages. Defaults to 0.
        :param seed: Seed for the pseudo random generators. Defaults to None.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible. Defaults to "auto".
        :param _init_setup_model: Whether to build the network at the creation of the instance. Defaults to True.
        """
        super().__init__(
            policy,
            env,
            lr_value=lr_value,
            lr_scheduler=lr_scheduler,
            lr_warmup_fraction=lr_warmup_fraction,
            lr_linear_end_value=lr_linear_end_value,
            lr_linear_end_fraction=lr_linear_end_fraction,
            n_steps=1,  # NStepReplayBuffer not implemented
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            replay_buffer_checkpoint=replay_buffer_checkpoint,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,),
        )

        self.target_entropy = target_entropy
        self.target_entropy_ratio = target_entropy_ratio
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.policy_frequency = policy_frequency
        self.ent_coef_optimizer: Optional[torch.optim.Adam] = None
        self.mixed_precision = mixed_precision
        self.preprocessor_class = preprocessor_class
        self.preprocessor_kwargs = preprocessor_kwargs
        self.max_grad_norm = max_grad_norm

        if _init_setup_model:
            self._setup_model()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        if self.device.type == torch.device("cuda").type:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        if self.preprocessor_class is not None:
            self.preprocessor = (
                eval(self.preprocessor_class) if isinstance(self.preprocessor_class, str) else self.preprocessor_class)(
                observation_space=self.observation_space, action_space=self.action_space, device=self.device, **self.preprocessor_kwargs)
            self.observation_space = self.preprocessor.proc_observation_space
            self.action_space = self.preprocessor.proc_action_space

        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs,
        ).to(self.device)

        if self.replay_buffer_checkpoint is None:
            self.replay_buffer = (
                eval(self.replay_buffer_class) if isinstance(self.replay_buffer_class, str) else self.replay_buffer_class)(
                observation_space=self.observation_space,
                action_space=self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                **self.replay_buffer_kwargs,
            )
        else:  # Load replay buffer from checkpoint if provided
            if self.verbose >= 1:
                print(f"Loading replay buffer from {self.replay_buffer_checkpoint}")
            self.load_replay_buffer(
                self.replay_buffer_checkpoint, truncate_last_traj=self.replay_buffer_kwargs.get("optimize_memory_usage")
            )

        # Setup entropy coefficient learning
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 0.001  # FastSAC default
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.AdamW([self.log_ent_coef], lr=self.lr_schedule(1), betas=(0.9, 0.95))
        else:
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)

        # Setup Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(sync_with_dataloader=True, num_steps=1),
            mixed_precision=self.mixed_precision,
            cpu=torch.device("cpu") == self.device,
        )
        self.policy, self.policy.actor.optimizer, self.policy.critic.optimizer = self.accelerator.prepare(self.policy, self.policy.actor.optimizer, self.policy.critic.optimizer)
        if self.ent_coef_optimizer: self.ent_coef_optimizer = self.accelerator.prepare(self.ent_coef_optimizer)
        self.policy.optimizer = [self.policy.actor.optimizer, self.policy.critic.optimizer] + ([self.ent_coef_optimizer] if self.ent_coef_optimizer is not None else [])
        self._create_aliases()
        self.replay_buffer.set_accelerator(self.accelerator)

        # Running mean and running var for batch norm
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            self.target_entropy = -float(-np.prod(self.env.action_space.shape).astype(np.float32)) * (1.0 - self.target_entropy_ratio)
        else:
            self.target_entropy = float(self.target_entropy)
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

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

        with torch.no_grad():
            while not (episodes_completed >= n_episodes).all():
                actions = self.policy.predict(last_obs)

                obs, rewards, dones, infos = env.step(self.process_out_actions(actions)[1])
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

                last_obs = obs

        mean_success = (successes / n_episodes).cpu().mean()
        mean_rewards = (sum_rewards / n_episodes).cpu().mean()
        mean_lengths = (sum_lengths.float() / n_episodes).cpu().mean()

        return mean_success, mean_rewards, mean_lengths

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        learning_starts: int = 0,
        log_interval: int | None = None,
    ) -> RolloutReturn:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.\
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        assert isinstance(replay_buffer, ReplayBuffer), f"{replay_buffer} doesn't support off-policy algorithms."
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        assert self._last_obs is not None, "No previous observation was provided"
        if not isinstance(self._last_episode_starts, torch.Tensor):  # first rollout
            self._last_obs = self.process_obs(observations=self._last_obs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps, n_episodes = 0, 0
        callback.on_rollout_start()

        self.logger.start_timer("s/rollout")

        while should_collect_more_steps(train_freq, n_steps, n_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Select action randomly or according to policy
                if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):  # Warmup phase
                    actions = torch.tensor([self.action_space.sample() for _ in range(env.num_envs)], device=self.device)
                else:
                    actions = self.policy(self._last_obs, deterministic=False)
            buffer_actions, actions = self.process_out_actions(actions)
            obs, rewards, dones, infos = env.step(actions)
            obs = self.process_obs(observations=obs)
            self.logger.record_mean("rollout/mean_rew", rewards.mean().item())

            # Post-step operations
            self.num_timesteps += env.num_envs
            n_steps += 1
            n_episodes += dones.sum()

            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(n_steps * env.num_envs, n_episodes, continue_training=False)

            # Recover terminal obs for reset envs since info
            obs_ = deepcopy(obs)
            dones_ = dones
            if dones.any():
                terminal_obs_processed = self.process_obs({k: v if isinstance(v, str) else v[dones] if isinstance(v, torch.Tensor) else {k2: v2[dones] for k2, v2 in v.items()} for k, v in infos["terminal_obs"].items()} if isinstance(infos["terminal_obs"], dict) \
                    else infos["terminal_obs"][dones])
                if isinstance(obs_, dict):
                    for key in obs_:
                        obs_[key][dones] = terminal_obs_processed[key]
                else:
                    obs_[dones] = terminal_obs_processed
                dones_ = dones & ~infos.get("truncated", torch.zeros_like(dones))

            # Store data in replay buffer
            replay_buffer.add(self._last_obs, obs_, buffer_actions, rewards, dones_)
            replay_buffer.restart_noise(dones)

            self._on_step()
            self._last_obs = obs

        self.logger.stop_timer("s/rollout", train_freq.frequency)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return RolloutReturn(n_steps * env.num_envs, n_episodes, continue_training=True)

    def train(self, gradient_steps: int) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.logger.clear()
        self.logger.start_timer("s/epoch")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer] + ([self.ent_coef_optimizer] if self.ent_coef_optimizer is not None else [])
        self._update_learning_rate(optimizers)

        for rollout_data in self.replay_buffer.get(self.batch_size):
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Get entropy coefficient
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # === Critic Update (Distributional) ===
            with torch.no_grad():
                # For n-step replay, discount factor is gamma**n_steps (when no early termination)
                discount = rollout_data.discounts if hasattr(rollout_data, "discounts") else torch.full_like(rollout_data.rewards, self.gamma)
                # Get next actions and log probs from actor
                next_actions, next_log_prob = self.actor.action_log_prob(rollout_data.next_observations)
                # Compute bootstrap mask (1 for non-terminal, handle truncation)
                bootstrap = (1 - rollout_data.dones).float()
                # Compute adjusted rewards (subtract entropy term from rewards)
                adjusted_rewards = rollout_data.rewards - discount * bootstrap * ent_coef * next_log_prob

                # Get target distributions via projection
                target_distributions = self.critic_target.projection(
                    rollout_data.next_observations,
                    next_actions,
                    adjusted_rewards,
                    bootstrap,
                    discount,
                )  # [n_critics, batch, num_atoms]

            # Get current Q-network outputs (logits)
            current_q_values = self.critic(rollout_data.observations, rollout_data.actions)
            # Compute distributional critic loss (categorical cross-entropy)
            q_log_probs = F.log_softmax(current_q_values, dim=-1)  # [n_critics, batch, num_atoms]

            critic_loss = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=1).sum(dim=0)

            self.critic.optimizer.zero_grad()
            self.accelerator.backward(critic_loss)

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            self.critic.optimizer.step()

            self.logger.record_mean("train/critic_loss", critic_loss.item())

            # Log Q-value statistics
            with torch.no_grad():
                q_values = self.critic.get_q_values(rollout_data.observations, rollout_data.actions)
                self.logger.record_mean("train/qf_max", q_values.max().item())
                self.logger.record_mean("train/qf_min", q_values.min().item())
                self.logger.record_mean("train/qf_mean", q_values.mean().item())

            # === Entropy Coefficient Update ===
            ent_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Use next_log_prob (already computed) to match holosoma implementation
                # _, log_prob = self.actor.action_log_prob(rollout_data.observations)
                ent_loss = (-self.log_ent_coef.exp() * (next_log_prob.detach() + self.target_entropy)).mean()

                self.ent_coef_optimizer.zero_grad()
                self.accelerator.backward(ent_loss)
                self.ent_coef_optimizer.step()

                self.logger.record_mean("train/ent_loss", ent_loss.item())

            self.logger.record_mean("train/ent_coef", ent_coef.item())

            # === Actor Update (every policy_frequency steps) ===
            if self._n_updates % self.policy_frequency == 0:
                # Get actions and log probs from actor
                actions_pi, log_prob = self.actor.action_log_prob(rollout_data.observations)

                # Compute Q-values for policy actions
                q_values_pi = self.critic.get_q_values(rollout_data.observations, actions_pi)
                # Use mean Q-value across critics (holosoma uses mean, not min like standard SAC)
                mean_q_pi = q_values_pi.mean(dim=0)

                # Actor loss: maximize Q - alpha * log_prob
                actor_loss = (ent_coef * log_prob - mean_q_pi).mean()

                self.actor.optimizer.zero_grad()
                self.accelerator.backward(actor_loss)

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                self.actor.optimizer.step()

                self.logger.record_mean("train/actor_loss", actor_loss.item())

                # Log action statistics
                with torch.no_grad():
                    _, _, kwargs = self.actor.get_action_dist_params(rollout_data.observations)
                    self.logger.record_mean("train/policy_entropy", -log_prob.mean().item())

            self._n_updates += 1

            # Update target networks
            if self._n_updates % self.target_update_interval == 0:
                # Soft update using Polyak update (torch._foreach for efficiency)
                with torch.no_grad():
                    src_params = list(self.critic.parameters())
                    tgt_params = list(self.critic_target.parameters())
                    torch._foreach_mul_(tgt_params, 1.0 - self.tau)
                    torch._foreach_add_(tgt_params, src_params, alpha=self.tau)

            if self._n_updates % gradient_steps == 0:
                break

        self.logger.stop_timer("s/epoch")
        self.logger.record("train/n_updates", self._n_updates)

    def learn(
        self: SelfFastSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "FastSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFastSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def _excluded_save_params(self) -> List[str]:
        return ["actor", "critic", "critic_target", "observation_space", "action_space"] + super()._excluded_save_params()
