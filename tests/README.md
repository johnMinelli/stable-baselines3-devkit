# Test Suite Documentation

This directory contains comprehensive tests for the RL/IL devkit. The test suite is organized into unit tests, integration tests, and end-to-end tests.

## Directory Structure

```
tests/
├── test_smoke.py                         # Quick smoke tests (run first!)
├── unit/                                 # Unit tests for individual components
│   ├── test_preprocessors.py             # Preprocessor normalization, transformations
│   ├── test_policies.py                  # Policy forward passes, shapes, initialization
│   ├── test_buffers.py                   # Buffer operations (add, sample, GAE)
│   ├── test_utils.py                     # Utility functions, space helpers
│   ├── test_wrappers.py                  # Environment wrapper tests
│   ├── test_datasets.py                  # Dataset loader and preprocessor tests
│   └── test_config.py                    # Configuration loading and validation
├── integration/                          # Integration tests for component interactions
│   ├── test_agent_policy_flow.py         # Agent-policy compatibility and flow
│   └── test_preprocessor_policy_flow.py  # Data flow through pipeline
├── e2e/                                  # End-to-end workflow tests
│   ├── test_training_workflows.py        # Complete training loops with mocks
│   └── test_readme_training_scripts.py   # Real training script execution
├── fixtures/                             # Shared test fixtures
│   ├── conftest.py                       # Shared pytest fixtures (device, cuda_device)
│   ├── mock_envs.py                      # Mock gym environments (DummyVecEnv)
│   └── mock_datasets.py                  # Mock datasets for SL (MockDataset, SequenceMockDataset)
├── README.md                             # This file
└── TESTING_STRATEGY.md                   # Testing strategy and coverage

```

## Current Test Status

| Component | Unit Tests | Integration (Mock) | Integration (Real) |
|-----------|------------|-------------------|-------------------|
| Preprocessors | ✅ | ✅ | ✅ |
| Policies | ✅ | ✅ | ✅ |
| SAC Policy | ✅ | ✅ | ✅ |
| Buffers | ✅ | ✅ | ✅ |
| PPO Agent | ✅ | ✅ | 🔴 Needs env |
| RecurrentPPO | ✅ | ✅ | 🔴 Needs env |
| TransformerPPO | ✅ | ✅ | 🔴 Needs env |
| SAC Agent | ✅ | ✅ | 🔴 Needs env |
| SL Agent | ✅ | ✅ | 🔴 Needs data |
| train.py | ❌ | ❌ | ✅ (if env installed) |
| train_off.py | ❌ | ❌ | ✅ (if data exists) |
| predict.py | ❌ | ❌ | ✅ (if checkpoint exists) |


## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test Categories

**Unit tests only:**
```bash
pytest tests/unit/
```

**Integration tests only:**
```bash
pytest tests/integration/
```

**End-to-end tests only:**
```bash
pytest tests/e2e/
```

### Run Tests by Marker

**Quick tests (exclude slow tests):**
```bash
pytest -m "not slow"
```

**GPU tests only:**
```bash
pytest -m gpu
```

**Smoke tests (fast validation):**
```bash
pytest -m smoke
```

#### Test Execution Notes
The tests in `test_agent_policy_flow.py` must run sequentially to avoid CUDA device conflicts. Install `pytest-order` for proper execution:
```bash
pip install pytest-order
```

Alternatively, run them individually.

## Setting Up Full Integration Testing

To run **all tests** including script integration:

1. **Install environments:**
   ```bash
   pip install isaaclab  # For Isaac Lab tests
   pip install mani_skill  # For ManiSkill tests
   pip install playground  # For MuJoCo Playground tests
   pip install gym_aloha  # For Aloha tests
   ```

2. **Prepare datasets:**
   ```bash
   # Download/prepare datasets to data/
   # Ensure data/StackCube-v1, data/PegInsertionSide-v1, etc. exist
   ```

3. **Run end-to-end script tests:**
   ```bash
   # Run with Isaac Lab (requires Isaac Lab installed)
   pytest tests/e2e/test_readme_training_scripts.py::TestTrainScriptIntegration::test_ppo_mlp_isaac_quick_training -v

   # Run with ManiSkill (requires ManiSkill installed)
   pytest tests/e2e/test_readme_training_scripts.py::TestTrainScriptIntegration::test_ppo_mlp_maniskill_quick_training -v

   # Run with MuJoCo Playground (requires mujoco_playground installed)
   pytest tests/e2e/test_readme_training_scripts.py::TestTrainScriptIntegration::test_ppo_mlp_mjx_quick_training -v

   # Run with datasets (requires data/ populated)
   pytest tests/e2e/test_readme_training_scripts.py::TestTrainOffScriptIntegration::test_sl_lstm_quick_training -v

   # Run full Isaac Lab train→predict workflow
   pytest tests/e2e/test_readme_training_scripts.py::TestEndToEndWorkflow::test_train_and_predict_workflow -v

   # Run all script integration tests
   pytest tests/e2e/test_readme_training_scripts.py -v
   ```