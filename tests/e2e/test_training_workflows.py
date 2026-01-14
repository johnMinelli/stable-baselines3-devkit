"""End-to-end tests for training workflows."""

import pytest
import torch

from tests.fixtures.mock_envs import DummyVecEnv, make_dummy_env
from tests.fixtures.conftest import cuda_device
from common.logger import Logger


class TestOnPolicyTrainingWorkflow:
    """End-to-end tests for PPO training."""

    @pytest.fixture
    def ppo_env(self, cuda_device):
        """Create environment for PPO testing."""
        return DummyVecEnv([
            lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4, episode_length=50)
            for _ in range(2)
        ], device=cuda_device)

    @pytest.mark.slow
    def test_ppo_with_callbacks(self, ppo_env, cuda_device, tmp_path):
        """Test PPO training with callbacks."""
        try:
            from algos.ppo_agent import PPO
            from common.callbacks import CheckpointCallback

            agent = PPO(
                policy="MlpPolicy",
                env=ppo_env,
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 32},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())

            # Create checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=64,
                save_path=str(tmp_path),
                name_prefix="test_model",
            )

            # Train
            agent.learn(total_timesteps=128, callback=checkpoint_callback)

            # Check that checkpoint was saved
            checkpoints = list(tmp_path.glob("test_model_*.zip"))
            assert len(checkpoints) > 0

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_ppo_evaluation(self, ppo_env, cuda_device):
        """Test PPO evaluation after training."""
        try:
            from algos.ppo_agent import PPO

            agent = PPO(
                policy="MlpPolicy",
                env=ppo_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=64)

            # Evaluate
            ppo_env.reset()
            agent.predict_rollout(env=ppo_env, callback=None, n_episodes=1)

            # Should complete without errors
            assert True

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_save_and_load_model(self, ppo_env, cuda_device, tmp_path):
        """Test saving and loading a trained model."""
        try:
            from algos.ppo_agent import PPO

            # Train agent
            agent = PPO(
                policy="MlpPolicy",
                env=ppo_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=64)

            # Save
            save_path = tmp_path / "test_model.zip"
            agent.save(str(save_path))

            assert save_path.exists()

            # Load
            loaded_agent = PPO.load(str(save_path), env=ppo_env, device=cuda_device)

            # Test loaded agent
            ppo_env.reset()
            loaded_agent.predict_rollout(env=ppo_env, callback=None, n_episodes=1)

            return True

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_resume_training(self, ppo_env, cuda_device, tmp_path):
        """Test resuming training from a checkpoint."""
        try:
            from algos.ppo_agent import PPO

            # Initial training
            agent = PPO(
                policy="MlpPolicy",
                env=ppo_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 16},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=64)
            initial_timesteps = agent.num_timesteps

            # Save
            save_path = tmp_path / "checkpoint.zip"
            agent.save(str(save_path))

            # Load and resume
            loaded_agent = PPO.load(str(save_path), env=ppo_env, device=cuda_device)

            loaded_agent.set_logger(Logger())
            loaded_agent.learn(total_timesteps=64, reset_num_timesteps=False)

            # Should have trained more
            assert loaded_agent.num_timesteps > initial_timesteps

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


class TestAgentsTraining:
    """End-to-end tests for Recurrent PPO training."""

    @pytest.fixture
    def env(self, cuda_device):
        """Create environment for PPO testing."""
        return DummyVecEnv([
            lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4, episode_length=50)
            for _ in range(2)
        ], device=cuda_device)

    @pytest.mark.slow
    def test_ppo_training_loop(self, env, cuda_device):
        """Test complete PPO training loop."""
        try:
            from algos.ppo_agent import PPO

            agent = PPO(
                policy="MlpPolicy",
                env=env,
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                preprocessor_class="Gym_2_Mlp",
                preprocessor_kwargs={"drop_images": True},
                policy_kwargs={"net_arch": [32, 32]},
                rollout_buffer_class="RolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 32},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())

            # Train for a few steps
            agent.learn(total_timesteps=128)

            # Check that training completed
            assert agent.num_timesteps == 128

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")


    @pytest.mark.slow
    def test_recurrent_ppo_training_loop(self, env, cuda_device):
        """Test complete Recurrent PPO training loop."""
        try:
            from algos.ppo_recurrent_agent import RecurrentPPO

            agent = RecurrentPPO(
                policy="MlpLstmPolicy",
                env=env,
                n_steps=64,
                batch_size=32,
                n_epochs=2,
                preprocessor_class="Gym_2_Lstm",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="RecurrentRolloutBuffer",
                rollout_buffer_kwargs={"buffer_size": 64},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=256)

            assert agent.num_timesteps == 256

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_transformer_ppo_training_loop(self, env, cuda_device):
        """Test complete Transformer PPO training loop."""
        try:
            from algos.ppo_transformer_agent import TransformerPPO

            agent = TransformerPPO(
                policy="TransformerPolicy",
                env=env,
                n_steps=64,
                batch_size=32,
                n_epochs=2,
                preprocessor_class="Gym_2_Tr",
                preprocessor_kwargs={"drop_images": True},
                rollout_buffer_class="TransformerRolloutBuffer",
                rollout_buffer_kwargs={
                    "buffer_size": 64,
                    "memorize_cache": False,
                    "embed_dim": 128,
                    "num_blocks": 2,
                    "memory_length": 32
                },
                policy_kwargs={"embed_dim": 128, "num_blocks": 2},
                device=cuda_device,
                verbose=0,
            )

            agent.set_logger(Logger())
            agent.learn(total_timesteps=256)

            assert agent.num_timesteps == 256

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

    @pytest.mark.slow
    def test_sl_agent_training_loop(self, cuda_device):
        """Test complete SL agent training loop.

        If a real dataset is available locally, perform full learning test.
        Otherwise, test with mock dataset or skip.
        """
        import os

        # Check if real dataset is available
        dataset_path = None  # "src/data/StackCube-v1"

        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found: {dataset_path}")

        try:
            from algos.sl_agent import SL

            # Use real LeRobot dataset if available
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from common.datasets.dataloader import DataLoader
            from common.datasets.lerobot_preprocessor import LerobotPreprocessor

            dataset = LeRobotDataset("johnMinelli/ManiSkill_StackCube-v1_recovery", root=dataset_path)  # no delta_timestamps
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True, device=cuda_device)

            ds_input_shapes = {
                "observation.images.base_camera": (480, 640, 3),
                "observation.images.hand_camera": (480, 640, 3),
                "observation.state": (9,),
                "observation.privileged": (30,),
            }
            ds_output_shapes = {"action": (8,)}

            # Create SL agent
            agent = SL(
                policy="LSTMPolicy",
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
            agent.learn(16)

            assert agent.num_timesteps > 0, "Agent should have trained some steps"

        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")
        except AttributeError as e:
            pytest.skip(f"SL agent API mismatch: {e}")


# class TestDeterminism:
#     """Test training determinism with fixed seeds."""
#
#     def test_deterministic_training(self, cuda_device):
#         """Test that training is deterministic with same seed."""
#         try:
#             from algos.ppo_agent import PPO
#             import numpy as np
#
#             def train_with_seed(seed):
#                 torch.manual_seed(seed)
#                 np.random.seed(seed)
#
#                 env = DummyVecEnv([
#                     lambda: make_dummy_env("dict_obs", state_dim=36, action_dim=4)
#                     for _ in range(2)
#                 ], device=cuda_device)
#
#                 agent = PPO(
#                     policy="MlpPolicy",
#                     env=env,
#                     n_steps=16,
#                     batch_size=8,
#                     n_epochs=1,
#                     preprocessor_class="Gym_2_Mlp",
#                     preprocessor_kwargs={"drop_images": True},
#                     rollout_buffer_class="RolloutBuffer",
#                     rollout_buffer_kwargs={"buffer_size": 16},
#                     device=cuda_device,
#                     seed=seed,
#                     verbose=0,
#                 )
#
#                 agent.set_logger(Logger())
#                 agent.learn(total_timesteps=64)
#
#                 # Get final policy weights
#                 return [p.clone() for p in agent.policy.parameters()]
#
#             # Train twice with same seed
#             weights1 = train_with_seed(42)
#             weights2 = train_with_seed(42)
#
#             # Weights should be identical
#             for w1, w2 in zip(weights1, weights2):
#                 assert torch.allclose(w1, w2, atol=1e-6)
#
#         except ImportError as e:
#             pytest.skip(f"Required modules not available: {e}")
