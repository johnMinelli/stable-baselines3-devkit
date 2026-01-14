"""
RecurrentActorCriticPolicy as part of the library policies of SB3, strictly modified only to allow compatibility with provided agents. Differently from original implementation:
- feature extraction i.e. preprocessing happen outside of the policy. This way we deal both pre and post in same class dependently to the source and algorithm.
- update of the hidden happens outside (as part of the buffer management) to not make the policy deal with restarts and make the number of arguments more general between policies.
- distribution output includes also mode and std to compute real kl.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.recurrent.type_aliases import RNNStates


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the LSTM
        constructor.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Union[float, Schedule],
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            lstm_hidden_size: int = 256,
            n_lstm_layers: int = 1,
            shared_lstm: bool = False,
            enable_critic_lstm: bool = True,
            lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.hidden_size = lstm_hidden_size
        self.n_layers = n_lstm_layers
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.hidden_state_shape = (4 if (self.enable_critic_lstm or self.shared_lstm) else 2, n_lstm_layers, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
                self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
                self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                batch_first=True,
                **self.lstm_kwargs,
            )
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _unpack_lstm_states(self, lstm_states: th.Tensor) -> Tuple[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor]]:
        """
        Unpack the batch_first stacked LSTM states tensor into tuples for PyTorch LSTM.

        :param lstm_states: Stacked LSTM states tensor
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs or
                   (batch_size, 2, n_lstm_layers, lstm_hidden_size) for actor-only
        :return: Tuple of (pi_states, vf_states) where each is a tuple of (h, c)
            in format (n_layers, batch, hidden) for PyTorch LSTM
        """
        if self.enable_critic_lstm or self.shared_lstm:
            # Split into 4 chunks: [h_pi, c_pi, h_vf, c_vf] along dim=1
            h_pi, c_pi, h_vf, c_vf = th.unbind(lstm_states, dim=1)
            # Transpose from (batch, n_layers, hidden) to (n_layers, batch, hidden) for PyTorch LSTM
            h_pi = h_pi.transpose(0, 1).contiguous()
            c_pi = c_pi.transpose(0, 1).contiguous()
            h_vf = h_vf.transpose(0, 1).contiguous()
            c_vf = c_vf.transpose(0, 1).contiguous()
            pi_states = (h_pi, c_pi)
            vf_states = (h_vf, c_vf)
        else:
            # No separate critic LSTM, split into (h, c) for actor only
            h_pi, c_pi = th.unbind(lstm_states, dim=1)
            # Transpose from (batch, n_layers, hidden) to (n_layers, batch, hidden)
            h_pi = h_pi.transpose(0, 1).contiguous()
            c_pi = c_pi.transpose(0, 1).contiguous()
            pi_states = vf_states = (h_pi, c_pi)

        return pi_states, vf_states

    def _pack_lstm_states(
        self,
        lstm_states_pi: Tuple[th.Tensor, th.Tensor],
        lstm_states_vf: Optional[Tuple[th.Tensor, th.Tensor]] = None
    ) -> th.Tensor:
        """
        Pack LSTM state tuples back into batch_first stacked tensor format.

        :param lstm_states_pi: Actor LSTM states tuple (h, c) in format (n_layers, batch, hidden)
        :param lstm_states_vf: Critic LSTM states tuple (h, c) in format (n_layers, batch, hidden).
            If None, uses lstm_states_pi for both.
        :return: Stacked LSTM states tensor
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs or
                   (batch_size, 2, n_lstm_layers, lstm_hidden_size) for actor-only
        """
        if self.enable_critic_lstm or self.shared_lstm:
            if lstm_states_vf is None:
                lstm_states_vf = lstm_states_pi
            # Transpose each state back to batch_first format
            h_pi_out = lstm_states_pi[0].transpose(0, 1)
            c_pi_out = lstm_states_pi[1].transpose(0, 1)
            h_vf_out = lstm_states_vf[0].transpose(0, 1)
            c_vf_out = lstm_states_vf[1].transpose(0, 1)
            # Stack: [h_pi, c_pi, h_vf, c_vf] along dim=1
            # Output shape: (batch, 4, n_layers, hidden)
            return th.stack([h_pi_out, c_pi_out, h_vf_out, c_vf_out], dim=1)
        else:
            # Transpose each state back to batch_first format
            h_pi_out = lstm_states_pi[0].transpose(0, 1)
            c_pi_out = lstm_states_pi[1].transpose(0, 1)
            # Stack: [h_pi, c_pi] along dim=1
            # Output shape: (batch, 2, n_layers, hidden)
            return th.stack([h_pi_out, c_pi_out], dim=1)

    @staticmethod
    def _process_sequence(
            features: th.Tensor,
            lstm_states: Tuple[th.Tensor, th.Tensor],
            episode_starts: th.Tensor,
            lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous hidden and cell states of the LSTM, respectively
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, lstm.input_size))
        episode_starts = episode_starts.reshape((n_seq, -1))

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(
            self,
            obs: th.Tensor,
            lstm_states: th.Tensor,
            episode_starts: Optional[th.Tensor] = None,
            deterministic: bool = False,
            get_distribution: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, Union[Tuple[th.Tensor, th.Tensor, th.Tensor], Any], th.Tensor]:
        """
        Forward pass in all the networks (actor and critic).

        :param obs: Observation
        :param lstm_states: The last hidden and memory states for the LSTM as a single concatenated tensor.
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs
            Contains: [h_pi, c_pi, h_vf, c_vf] stacked in dimension 1
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case). If None, no reset is performed.
        :param deterministic: Whether to sample or use deterministic actions
        :param get_distribution: If True, return the full distribution object. Otherwise return (log_prob, mode, stddev) tuple
        :return: action, value, dist_out (distribution or tuple of log_prob/mode/stddev), and updated lstm_states
            with shape (batch_size, 4, n_lstm_layers, lstm_hidden_size)
        """
        # Unpack LSTM states from batch_first format to PyTorch LSTM format
        pi_states, vf_states = self._unpack_lstm_states(lstm_states)

        # Preprocess the observation if needed
        features = obs["state"]

        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features

        # If episode_starts is not provided, create a tensor of zeros (no resets)
        if episode_starts is None:
            batch_size = obs.shape[0] if not isinstance(obs, dict) else obs[next(iter(obs.keys()))].shape[0]
            episode_starts = th.zeros(batch_size, dtype=th.float32, device=self.device)

        # Process through LSTM
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, pi_states, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, vf_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf).squeeze(-1)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)

        # Create distribution output matching ac_mlp_policy format
        dist_out = distribution if get_distribution else (distribution.log_prob(actions), distribution.mode(), distribution.distribution.stddev)

        # Pack LSTM states back to batch_first format
        output_states = self._pack_lstm_states(lstm_states_pi, lstm_states_vf)

        return actions, values, dist_out, output_states

    def get_distribution(
            self,
            obs: th.Tensor,
            lstm_states: th.Tensor,
            episode_starts: Optional[th.Tensor] = None,
    ) -> Tuple[Distribution, th.Tensor]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param lstm_states: The last hidden and memory states for the LSTM as a single stacked tensor.
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs or
                   (batch_size, 2, n_lstm_layers, lstm_hidden_size) for actor-only
            Contains: [h_pi, c_pi, h_vf, c_vf] stacked in dimension 1 (or [h_pi, c_pi] for actor-only)
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case). If None, no reset is performed.
        :return: the action distribution and updated lstm_states in the same stacked format
        """
        # Unpack LSTM states from batch_first format to PyTorch LSTM format
        pi_states, vf_states = self._unpack_lstm_states(lstm_states)

        # If episode_starts is not provided, create a tensor of zeros (no resets)
        if episode_starts is None:
            batch_size = obs.shape[0] if not isinstance(obs, dict) else obs[next(iter(obs.keys()))].shape[0]
            episode_starts = th.zeros(batch_size, dtype=th.float32, device=self.device)

        # Process features
        features = obs["state"]
        latent_pi, lstm_states_pi = self._process_sequence(features, pi_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)

        # Pack LSTM states back, keeping vf states unchanged
        output_states = self._pack_lstm_states(lstm_states_pi, vf_states)

        return self._get_action_dist_from_latent(latent_pi), output_states

    def predict_values(
            self,
            obs: th.Tensor,
            lstm_states: th.Tensor,
            episode_starts: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :param lstm_states: The last hidden and memory states for the LSTM as a single stacked tensor.
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs
            Contains: [h_pi, c_pi, h_vf, c_vf] stacked in dimension 1
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case). If None, no reset is performed.
        :return: the estimated values
        """
        # Unpack LSTM states from batch_first format to PyTorch LSTM format
        pi_states, vf_states = self._unpack_lstm_states(lstm_states)

        # If episode_starts is not provided, create a tensor of zeros (no resets)
        if episode_starts is None:
            batch_size = obs.shape[0] if not isinstance(obs, dict) else obs[next(iter(obs.keys()))].shape[0]
            episode_starts = th.zeros(batch_size, dtype=th.float32, device=self.device)

        # Preprocess the observation if needed
        features = obs["state"]

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, vf_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, pi_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf).squeeze(-1)

    def evaluate_actions(
            self, obs: th.Tensor, actions: th.Tensor, lstm_states: th.Tensor, episode_starts: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor, Tuple[th.Tensor, th.Tensor, th.Tensor], th.Tensor]:
        """
        Evaluate actions according to the current policy, given the observations.

        :param obs: Observation
        :param actions: Actions to evaluate
        :param lstm_states: The last hidden and memory states for the LSTM as a single stacked tensor.
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs
            Contains: [h_pi, c_pi, h_vf, c_vf] stacked in dimension 1
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case). If None, no reset is performed.
        :return: predicted actions (sampled from distribution), estimated values,
            dist_out tuple (log_prob, mode, stddev), and entropy of the action distribution
        """
        # Get predicted actions and distribution from forward pass
        predicted_actions, values, distribution, _ = self.forward(obs, lstm_states, episode_starts, get_distribution=True)

        # Create dist_out using the provided actions (not the predicted ones)
        dist_out = distribution.log_prob(actions), distribution.mode(), distribution.distribution.stddev
        entropy = distribution.entropy()

        return predicted_actions, values, dist_out, entropy

    def _predict(
            self,
            observation: th.Tensor,
            lstm_states: th.Tensor,
            episode_starts: th.Tensor,
            deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM as a single stacked tensor.
            Shape: (batch_size, 4, n_lstm_layers, lstm_hidden_size) for separate LSTMs
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and updated lstm_states tensor
        """
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        mode = self.training
        self.set_training_mode(False)
        actions, states = self._predict(observation, lstm_states=state, episode_starts=episode_start, deterministic=deterministic)
        self.set_training_mode(mode)

        return actions, states


MlpLstmPolicy = RecurrentActorCriticPolicy
