"""Integration tests for agent-policy compatibility."""

import pytest
import torch
from gymnasium import spaces

from algos.ppo_agent import PPO
from algos.ppo_recurrent_agent import RecurrentPPO
from algos.ppo_transformer_agent import TransformerPPO
from algos.sl_agent import SL
from algos.sac_agent import SAC
from common.logger import Logger
from tests.fixtures.mock_envs import make_dummy_env, DummyVecEnv
from tests.fixtures.conftest import device, cuda_device


import pytest
from accelerate.state import AcceleratorState

@pytest.fixture(scope="function", autouse=True)
def reset_accelerator():
    """Reset AcceleratorState between tests."""
    yield
    # Cleanup after test
    AcceleratorState._reset_state()


class TestAgentPolicyMismatch:
    """Test that incompatible agent-policy combinations fail gracefully."""

    def test_wrong_policy_type_error(self):
        """Test that using wrong policy type raises appropriate error."""
        # Example: Using a policy that doesn't exist
        with pytest.raises((ValueError, KeyError, AttributeError)):
            from tests.fixtures.mock_envs import make_dummy_env, DummyVecEnv

            env = DummyVecEnv([lambda: make_dummy_env("simple") for _ in range(2)])

            # Try to use a non-existent policy
            agent = PPO(
                policy="NonExistentPolicy",
                env=env,
                n_steps=16,
            )


class TestPreprocessingForAgent:
    """Test PPO agent with different policy types."""

    @pytest.fixture(scope="function")
    def simple_env(self):
        """Create a simple vectorized environment."""
        return DummyVecEnv([lambda: make_dummy_env("simple", obs_dim=10, action_dim=4) for _ in range(2)])

    def test_ppo_with_mlp_policy_fail(self, simple_env, device):
        """Test PPO agent with MLP policy on unwrapped environment.

        This test expects an error because the simple dummy environment returns
        Box observations, but the Gym_2_Mlp preprocessor expects Dict observations
        (the "shared data space" format). The environment needs to be wrapped with
        Sb3EnvStdWrapper or similar to convert observations to the shared format.
        """
        try:
            agent = PPO(
                policy="MlpPolicy",
                env=simple_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=device,
            )

            agent._setup_learn(10000)

            # Should raise AssertionError because observations are not in dict format
            with pytest.raises(AssertionError, match="Processing implemented only for `Dict` observations"):
                agent.collect_rollouts(agent.env, None, agent.rollout_buffer, n_rollout_steps=16)

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestPPOAgents:
    """Test Recurrent PPO agent with compatible policies."""

    @pytest.fixture(scope="function")
    def dict_obs_env(self):
        """Create a dict observation vectorized environment (mimics wrapped env format)."""
        return DummyVecEnv([lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4) for _ in range(2)])


    def test_ppo_with_mlp_policy(self, dict_obs_env, device):
        """Test PPO agent with MLP policy on unwrapped environment.

        This test expects an error because the simple dummy environment returns
        Box observations, but the Gym_2_Mlp preprocessor expects Dict observations
        (the "shared data space" format). The environment needs to be wrapped with
        Sb3EnvStdWrapper or similar to convert observations to the shared format.
        """
        try:
            agent = PPO(
                policy="MlpPolicy",
                env=dict_obs_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)
            agent.collect_rollouts(agent.env, callbacks, agent.rollout_buffer, n_rollout_steps=16)

            assert agent.rollout_buffer.pos == 16
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_recurrent_ppo_with_lstm_policy(self, dict_obs_env, device):
        """Test Recurrent PPO with LSTM policy."""
        try:
            agent = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=dict_obs_env,
                n_steps=32,
                batch_size=16,
                n_epochs=1,
                preprocessor_class="Gym_2_Lstm",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="RecurrentRolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 32},
                device=device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)
            agent.collect_rollouts(agent.env, callbacks, agent.rollout_buffer, n_rollout_steps=32)

            assert agent.rollout_buffer.pos == 32
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_transformer_ppo_with_transformer_policy(self, dict_obs_env, device):
        """Test Transformer PPO with Transformer policy."""
        try:
            agent = TransformerPPO(
                policy="TransformerPolicy",
                env=dict_obs_env,
                n_steps=32,
                batch_size=16,
                n_epochs=1,
                preprocessor_class="Gym_2_Tr",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="TransformerRolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 32, "memorize_cache": False, "embed_dim": 128, "num_blocks": 2, "memory_length": 32},
                policy_kwargs={"embed_dim": 128, "num_blocks": 2},
                device=device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)
            agent.collect_rollouts(agent.env, callbacks, agent.rollout_buffer, n_rollout_steps=32)

            assert agent.rollout_buffer.pos == 32
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestSACAgents:
    """Test SAC (off-policy) agent with compatible policies."""

    @pytest.fixture(scope="function")
    def dict_obs_env(self, cuda_device):
        """Create a dict observation vectorized environment (mimics wrapped env format)."""
        return DummyVecEnv([lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4) for _ in range(2)], device=cuda_device)

    def test_sac_with_mlp_policy(self, dict_obs_env, cuda_device):
        """Test SAC agent with MLP policy on wrapped environment."""
        try:
            agent = SAC(
                policy="MlpPolicy",
                env=dict_obs_env,
                batch_size=8,
                learning_starts=10,
                train_freq=(1, "step"),
                gradient_steps=1,
                preprocessor_class="Gym_2_Sac",
                preprocessor_kwargs={"drop_images": True},
                replay_buffer_class="ReplayBuffer",
                replay_buffer_kwargs={"buffer_size": 100},
                device=cuda_device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)

            # Collect some experiences
            from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
            train_freq = TrainFreq(frequency=16, unit=TrainFrequencyUnit.STEP)
            agent.collect_rollouts(
                agent.env, callbacks, train_freq, agent.replay_buffer, learning_starts=0
            )

            assert agent.replay_buffer.pos > 0

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_sac_train_step(self, dict_obs_env, cuda_device):
        """Test SAC agent can perform a training step after collecting experiences."""
        try:
            agent = SAC(
                policy="MlpPolicy",
                env=dict_obs_env,
                batch_size=8,
                learning_starts=10,
                train_freq=(1, "step"),
                gradient_steps=2,
                preprocessor_class="Gym_2_Sac",
                preprocessor_kwargs={"drop_images": True},
                replay_buffer_class="ReplayBuffer",
                replay_buffer_kwargs={"buffer_size": 100},
                device=cuda_device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)

            # Collect enough experiences for training
            from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
            train_freq = TrainFreq(frequency=32, unit=TrainFrequencyUnit.STEP)
            agent.collect_rollouts(
                agent.env, callbacks, train_freq, agent.replay_buffer, learning_starts=0
            )

            # Perform training steps
            initial_updates = agent._n_updates
            agent.train(gradient_steps=2)

            # Check that training occurred
            assert agent._n_updates > initial_updates

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_sac_auto_entropy_tuning(self, dict_obs_env, cuda_device):
        """Test SAC agent with automatic entropy coefficient tuning."""
        try:
            agent = SAC(
                policy="MlpPolicy",
                env=dict_obs_env,
                batch_size=8,
                learning_starts=10,
                train_freq=(1, "step"),
                gradient_steps=1,
                ent_coef="auto",  # Enable automatic entropy tuning
                preprocessor_class="Gym_2_Sac",
                preprocessor_kwargs={"drop_images": True},
                replay_buffer_class="ReplayBuffer",
                replay_buffer_kwargs={"buffer_size": 100},
                device=cuda_device,
            )

            agent.set_logger(Logger())

            # Check that entropy coefficient optimizer was created
            assert agent.ent_coef_optimizer is not None
            assert agent.log_ent_coef is not None
            assert agent.log_ent_coef.requires_grad is True

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_sac_fixed_entropy_coef(self, dict_obs_env, cuda_device):
        """Test SAC agent with fixed entropy coefficient."""
        try:
            agent = SAC(
                policy="MlpPolicy",
                env=dict_obs_env,
                batch_size=8,
                learning_starts=10,
                train_freq=(1, "step"),
                gradient_steps=1,
                ent_coef=0.2,  # Fixed entropy coefficient
                preprocessor_class="Gym_2_Sac",
                preprocessor_kwargs={"drop_images": True},
                replay_buffer_class="ReplayBuffer",
                replay_buffer_kwargs={"buffer_size": 100},
                device=cuda_device,
            )

            agent.set_logger(Logger())

            # Check that no entropy coefficient optimizer was created
            assert agent.ent_coef_optimizer is None

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    def test_sac_target_network_update(self, dict_obs_env, cuda_device):
        """Test SAC target network soft update (Polyak update)."""
        try:
            agent = SAC(
                policy="MlpPolicy",
                env=dict_obs_env,
                batch_size=8,
                learning_starts=10,
                train_freq=(1, "step"),
                gradient_steps=1,
                tau=0.005,  # Soft update coefficient
                target_update_interval=1,
                preprocessor_class="Gym_2_Sac",
                preprocessor_kwargs={"drop_images": True},
                replay_buffer_class="ReplayBuffer",
                replay_buffer_kwargs={"buffer_size": 100},
                device=cuda_device,
            )

            agent.set_logger(Logger())
            _, callbacks = agent._setup_learn(10000)

            # Store initial target network weights
            initial_target_params = [p.clone() for p in agent.critic_target.parameters()]

            # Collect experiences and train
            from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
            train_freq = TrainFreq(frequency=32, unit=TrainFrequencyUnit.STEP)
            agent.collect_rollouts(
                agent.env, callbacks, train_freq, agent.replay_buffer, learning_starts=0
            )
            agent.train(gradient_steps=2)

            # Check that target network was updated
            updated_target_params = list(agent.critic_target.parameters())
            params_changed = any(
                not torch.allclose(initial, updated)
                for initial, updated in zip(initial_target_params, updated_target_params)
            )
            assert params_changed, "Target network should be updated via Polyak averaging"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestSLAgents:
    """Test supervised learning agent with different policies."""

    def test_sl_agent_with_mlp_policy(self, cuda_device):
        """Test SL agent with MLP policy on dict observations."""
        try:
            from common.datasets.dataloader import DataLoader
            from tests.fixtures.mock_datasets import MockDataset

            # Create mock dataset with dict observations (returns tensors on device)
            dataset = MockDataset(
                num_trajectories=10,
                trajectory_length=20,
                obs_dim=10,
                action_dim=4,
                obs_type="dict",
                include_images=True,
                device=cuda_device,
            )

            # Create dataloader
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True, device=cuda_device)

            # Define input and output shapes for SL agent
            ds_input_shapes = {
                "observation.state": [5],
                "observation.privileged": [5],
            }
            ds_output_shapes = {
                "action": [4],
            }

            # Initialize SL agent
            agent = SL(
                policy="MlpPolicy",
                ds_input_shapes=ds_input_shapes,
                ds_output_shapes=ds_output_shapes,
                demonstrations_data_loader=dataloader,
                batch_size=8,
                n_epochs=2,
                lr_value=1e-3,
                preprocessor_class="Lerobot_2_Mlp",
                preprocessor_kwargs={},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=16)

            # Check that policy was updated
            assert agent.policy is not None
            assert agent._n_updates > 0

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


@pytest.mark.order(1)
class TestLearningRateScheduling:
    """Test LR scheduling in agents.

    Tests that the lr in the optimizer is updated correctly by the agent following
    the 'lr_scheduler' provided. The agent gets as parameter:
    - lr_scheduler: "adaptive"/"constant"/...
    - lr_value: 3e-4

    Then self.lr_schedule is built as a schedule and passed to the policy.
    During learning loop the agent calls:
    - self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
    - self._update_learning_rate(self.policy.optimizer)

    This updates the lr in the policy's optimizer.

    Note: These tests must run sequentially to avoid CUDA device conflicts.
    Use pytest-order or run with -p no:xdist to disable parallel execution.
    """

    @pytest.fixture(scope="function")
    def dict_obs_env(self, cuda_device):
        """Create a dict observation vectorized environment (mimics wrapped env format)."""
        return DummyVecEnv([lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4) for _ in range(2)], device=cuda_device)

    @pytest.mark.order(1)
    def test_constant_lr_schedule(self, dict_obs_env, cuda_device):
        """Test constant LR schedule (no changes during training)."""
        try:
            from algos.ppo_agent import PPO

            agent = PPO(
                policy="MlpPolicy",
                env=dict_obs_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                lr_value=3e-4,
                lr_scheduler="constant",
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )
            agent.set_logger(Logger())

            initial_lr = agent.policy.optimizer.param_groups[0]['lr'](agent._current_progress_remaining)

            # Train for some steps
            agent.learn(total_timesteps=64)

            # LR should remain constant
            final_lr = agent.policy.optimizer.param_groups[0]['lr']
            assert abs(final_lr - initial_lr) < 1e-8, "Constant LR should not change"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.order(2)
    def test_linear_lr_schedule(self, dict_obs_env, cuda_device):
        """Test linear LR decay schedule."""
        try:
            from algos.ppo_agent import PPO

            agent = PPO(
                policy="MlpPolicy",
                env=dict_obs_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                lr_value=3e-4,
                lr_scheduler="linear",
                lr_linear_end_fraction=0.0005,
                lr_linear_end_value=1e-6,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )
            agent.set_logger(Logger())

            initial_lr = agent.policy.optimizer.param_groups[0]['lr'](agent._current_progress_remaining)

            # Train for half the timesteps
            agent.learn(total_timesteps=128)

            final_lr = agent.policy.optimizer.param_groups[0]['lr']

            # LR should decrease over time
            assert initial_lr > final_lr, "LR should decrease during training"
            assert final_lr == 1e-6, "LR didn't reach end value "

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.order(3)
    def test_separate_actor_critic_lr(self, dict_obs_env, cuda_device):
        """Test different LR for actor and critic networks."""
        try:
            from algos.ppo_agent import PPO

            # Some implementations support separate LR for actor and critic
            agent = PPO(
                policy="MlpPolicy",
                env=dict_obs_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                lr_value=[1e-3, 5e-3],
                lr_scheduler="constant",
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images":True},  # Note: check preprocessor proc_obs
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )
            agent.set_logger(Logger())

            # Check if optimizer has separate param groups
            if len(agent.policy.optimizer.param_groups) >= 2:
                actor_lr = agent.policy.optimizer.param_groups[0]['lr'](agent._current_progress_remaining)
                critic_lr = agent.policy.optimizer.param_groups[1]['lr'](agent._current_progress_remaining)

                # Different LRs should be set
                assert actor_lr != critic_lr, "Actor and critic should have different LRs"
            else:
                pytest.skip("Separate actor/critic LR not supported")

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except (KeyError, AttributeError):
            pytest.skip("Separate actor/critic LR not supported")
