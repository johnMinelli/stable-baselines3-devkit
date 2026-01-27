"""
Integration tests for actual training scripts.

These tests validate that the entry points (train.py, train_off.py, predict.py)
work with real configurations and environments.
"""

import os
import subprocess
import sys

import pytest


@pytest.fixture(scope="module", autouse=True)
def change_to_src_directory():
    """
    Change working directory to src/ if it exists in the current directory.

    The IDE sets cwd to one level below root, so we need to check if src/ exists
    and move into it for the training scripts to work properly.
    """
    original_cwd = os.getcwd()

    # Check if src directory exists in current working directory
    src_dir = os.path.join(original_cwd, "src")
    if os.path.isdir(src_dir):
        os.chdir(src_dir)
        print(f"Changed working directory from {original_cwd} to {src_dir}")

    yield

    # Restore original directory
    os.chdir(original_cwd)


class TestTrainScriptIntegration:
    """Test train.py with real environments."""

    @pytest.mark.slow
    def test_train_script_imports(self):
        """Test that train.py can be imported without errors."""
        try:
            # Try importing the main module
            import train  # noqa F401

            assert True
        except Exception as e:
            pytest.fail(f"Failed to import train.py: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ppo_mlp_isaac_quick_training(self):
        """Test PPO MLP training with Isaac Lab for a few steps."""
        try:
            # Check if Isaac Lab is available
            import isaaclab  # noqa F401
        except ImportError:
            pytest.skip("Isaac Lab not available")

        # Run training for just a few steps
        cmd = [
            sys.executable, "train.py",
            "--task", "Isaac-Lift-Cube-Franka-v0",
            "--envsim", "isaaclab",
            "--headless",
            "--num_envs", "2",
            "--agent", "custom_ppo_mlp",
            "--device", "cuda",
            "--sim_device", "cuda",
            "--max_iterations", "2",  # Just 2 iterations
            "--save_interval", "1000000",  # Don't save
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Check that it didn't crash
        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ppo_lstm_isaac_quick_training(self):
        """Test PPO LSTM training with Isaac Lab for a few steps."""
        try:
            import isaaclab  # noqa F401
        except ImportError:
            pytest.skip("Isaac Lab not available")

        cmd = [
            sys.executable, "train.py",
            "--task", "Isaac-Velocity-Flat-Anymal-C-v0",
            "--envsim", "isaaclab",
            "--headless",
            "--num_envs", "2",
            "--agent", "custom_ppo_lstm",
            "--device", "cpu",
            "--sim_device", "cpu",
            "--max_iterations", "2",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ppo_mlp_maniskill_quick_training(self):
        """Test PPO with ManiSkill environment."""
        try:
            import mani_skill  # noqa F401
        except ImportError:
            pytest.skip("ManiSkill not available")

        cmd = [
            sys.executable, "train.py",
            "--task", "StackCube-v1",
            "--envsim", "maniskill",
            "--num_envs", "2",
            "--agent", "Maniskill/stack_ppo_mlp_cfg",
            "--device", "cuda",
            "--sim_device", "cuda",
            "--max_iterations", "2",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ppo_mlp_mjx_quick_training(self):
        """Test PPO with MuJoCo Playground environment."""
        try:
            import mujoco_playground  # noqa F401
        except ImportError:
            pytest.skip("MuJoCo Playground not available")

        cmd = [
            sys.executable, "train.py",
            "--task", "CartpoleBalance",
            "--envsim", "mujoco_playground",
            "--num_envs", "2",
            "--agent", "MujocoPlayground/cartpole_ppo_mlp_cfg",
            "--device", "cuda",
            "--sim_device", "cuda",
            "--max_iterations", "2",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ppo_mlp_aloha_quick_training(self):
        """Test PPO with Aloha environment."""
        try:
            import gym_aloha  # noqa F401
        except ImportError:
            pytest.skip("gym_aloha not available")

        cmd = [
            sys.executable, "train.py",
            "--task", "gym_aloha/AlohaInsertion-v0",
            "--envsim", "aloha",
            "--num_envs", "1",
            "--agent", "Aloha/insertion_ppo_mlp_cfg",
            "--device", "cuda",
            "--max_iterations", "2",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_sac_mlp_isaac_quick_training(self):
        """Test SAC MLP (off-policy) training with Isaac Lab for a few steps."""
        try:
            import isaaclab  # noqa F401
        except ImportError:
            pytest.skip("Isaac Lab not available")

        cmd = [
            sys.executable, "train.py",
            "--task", "Isaac-Velocity-Flat-Anymal-C-v0",
            "--envsim", "isaaclab",
            "--headless",
            "--num_envs", "2",
            "--agent", "Isaac/velocity_sac_mlp_cfg",
            "--device", "cuda",
            "--sim_device", "cuda",
            "--max_iterations", "2",  # Just 2 iterations
            "--save_interval", "1000000",  # Don't save
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        assert result.returncode == 0, f"SAC Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


class TestTrainOffScriptIntegration:
    """Test train_off.py with real datasets."""

    @pytest.mark.slow
    def test_train_off_script_imports(self):
        """Test that train_off.py can be imported."""
        try:
            import train_off  # noqa F401

            assert True
        except Exception as e:
            pytest.fail(f"Failed to import train_off.py: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.dataset
    def test_sl_lstm_quick_training(self):
        """Test SL LSTM training with dataset."""
        # Check if dataset exists
        # dataset_path = "data/StackCube-v1"
        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found at {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa F401
        except ImportError:
            pytest.skip("LeRobot not available")

        cmd = [
            sys.executable, "train_off.py",
            "--task", "SL",
            "--agent", "Lerobot/StackCube/lerobot_sl_lstm_cfg",
            "--device", "cuda",
            "--val_interval", "10000",
            "--n_epochs", "1",  # Just 1 epochs
            "--batch_size", "8",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout (dataset loading can be slow)
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.dataset
    def test_sl_tcn_quick_training(self):
        """Test SL TCN training with dataset."""
        # dataset_path = "data/StackCube-v1"
        # if not os.path.exists(dataset_path):
        #     pytest.skip(f"Dataset not found at {dataset_path}")

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa F401
        except ImportError:
            pytest.skip("LeRobot not available")

        cmd = [
            sys.executable, "train_off.py",
            "--task", "SL",
            "--agent", "Lerobot/StackCube/lerobot_sl_tcn_cfg",
            "--device", "cuda",
            "--val_interval", "10000",
            "--n_epochs", "1",
            "--batch_size", "8",
            "--save_interval", "1000000",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


class TestPredictScriptIntegration:
    """Test predict.py with trained models."""

    @pytest.mark.slow
    def test_predict_script_imports(self):
        """Test that predict.py can be imported."""
        try:
            import predict  # noqa F401

            assert True
        except Exception as e:
            pytest.fail(f"Failed to import predict.py: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_predict_with_checkpoint(self, tmp_path):
        """Test prediction with a saved checkpoint."""
        # This requires a trained checkpoint
        # For now, we'll skip if no checkpoint exists
        checkpoint_dir = "save"
        if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
            pytest.skip("No saved checkpoints found")

        try:
            import isaaclab  # noqa F401
        except ImportError:
            pytest.skip("Isaac Lab not available")

        # Find a checkpoint
        checkpoints = []
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith(".zip"):
                    checkpoints.append(os.path.join(root, file))

        if not checkpoints:
            pytest.skip("No checkpoint files found")

        checkpoint = checkpoints[0]

        cmd = [
            sys.executable, "predict.py",
            "--task", "Isaac-Velocity-Flat-Anymal-C-v0",
            "--envsim", "isaaclab",
            "--num_envs", "1",
            "--val_episodes", "2",  # Just 2 episodes
            "--agent", "custom_ppo_mlp",
            "--device", "cpu",
            "--sim_device", "cpu",
            "--resume",
            "--checkpoint", checkpoint,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"Prediction failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


class TestConfigCompatibility:
    """Test that all config files are compatible with training scripts."""

    def test_all_agent_configs_exist(self):
        """Test that agent config files referenced in README exist."""
        config_paths = [
            "configs/agents/Isaac/lift_ppo_mlp_cfg.yaml",
            "configs/agents/Isaac/velocity_ppo_lstm_cfg.yaml",
            "configs/agents/Isaac/velocity_ppo_tr_cfg.yaml",
            "configs/agents/Isaac/velocity_sac_mlp_cfg.yaml",
            "configs/agents/Maniskill/stack_ppo_mlp_cfg.yaml",
            "configs/agents/Aloha/insertion_ppo_mlp_cfg.yaml",
            "configs/agents/Lerobot/StackCube/lerobot_sl_lstm_cfg.yaml",
        ]

        missing = []
        for path in config_paths:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            pytest.fail(f"Missing config files: {missing}")

    @pytest.mark.parametrize("config_file", [
        "configs/agents/Isaac/lift_ppo_mlp_cfg.yaml",
        "configs/agents/Isaac/velocity_ppo_lstm_cfg.yaml",
        "configs/agents/Maniskill/stack_ppo_mlp_cfg.yaml",
        "configs/agents/Lerobot/StackCube/lerobot_sl_lstm_cfg.yaml",
    ])
    def test_config_file_loadable(self, config_file):
        """Test that config files can be loaded."""
        if not os.path.exists(config_file):
            pytest.skip(f"Config file not found: {config_file}")

        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert isinstance(config, dict)


class TestEndToEndWorkflow:
    """Test complete workflows from config to inference."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_train_and_predict_workflow(self, tmp_path):
        """Test training a model and then using it for prediction."""
        try:
            import isaaclab  # noqa F401
        except ImportError:
            pytest.skip("Isaac Lab not available")

        # Step 1: Train for a few iterations
        train_cmd = [
            sys.executable, "train.py",
            "--task", "Isaac-Lift-Cube-Franka-v0",
            "--envsim", "isaaclab",
            "--headless",
            "--num_envs", "2",
            "--agent", "custom_ppo_mlp",
            "--device", "cpu",
            "--sim_device", "cpu",
            "--max_iterations", "3",
            "--save_interval", "2",  # Save after 2 iterations
            "--experiment_name", "test_e2e_workflow",
        ]

        train_result = subprocess.run(
            train_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert train_result.returncode == 0, f"Training failed:\n{train_result.stderr}"

        # Step 2: Find the saved checkpoint
        save_dir = "save/Isaac-Lift-Cube-Franka-v0_test_e2e_workflow"
        if not os.path.exists(save_dir):
            pytest.fail("Training didn't create save directory")

        # Find checkpoint
        checkpoints = []
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if file.endswith(".zip"):
                    checkpoints.append(os.path.join(root, file))

        if not checkpoints:
            pytest.fail("Training didn't save checkpoint")

        checkpoint = checkpoints[0]

        # Step 3: Run prediction
        predict_cmd = [
            sys.executable, "predict.py",
            "--task", "Isaac-Lift-Cube-Franka-v0",
            "--envsim", "isaaclab",
            "--headless",
            "--num_envs", "1",
            "--val_episodes", "1",
            "--agent", "custom_ppo_mlp",
            "--device", "cpu",
            "--sim_device", "cpu",
            "--checkpoint", checkpoint,
            "--resume",
        ]

        predict_result = subprocess.run(
            predict_cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        assert predict_result.returncode == 0, f"Prediction failed:\n{predict_result.stderr}"

        # Cleanup
        import shutil

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
