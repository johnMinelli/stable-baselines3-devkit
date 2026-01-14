import argparse
import os
import random
import sys
from dataclasses import is_dataclass
from typing import Any, Dict, TypeVar

import hydra
from isaaclab.app import AppLauncher
from omegaconf import DictConfig, OmegaConf

from common.utils import lists_to_tuples

T = TypeVar("T")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an agent with Stable-Baselines3 XL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--trajectory", action="store_true", default=False, help="Record trajectories during training.")
    parser.add_argument("--log_interval", type=int, default=10, help="Every many steps log training stats.")
    parser.add_argument("--val_interval", type=int, default=100, help="Every many steps record run validation.")
    parser.add_argument("--val_episodes", type=int, default=20, help="How many simulation episodes to run in validation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of subprocesses to sample from a dataloader.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--rollout_steps", type=int, default=None, help="Number of env steps to collect per rollout.")
    parser.add_argument("--n_epochs", type=int, default=None, help="Number of epoch/iterations to execute on a batch/rollout.")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of samples per batch/rollout.")
    parser.add_argument("--num_mini_batch", type=int, default=None, help="Number mini-batches/updates during a batch/rollout.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Number of steps of accumulation per mini-batch.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--agent", type=str, default=None, help="Name of the agent.")
    parser.add_argument("--envsim", type=str, default=None, help="Simulation environment.")
    parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging for wandb.")
    parser.add_argument("--tensorboard", action="store_true", default=None, help="Enable logging for tensorboard.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument("--save_interval", type=int, default=2000, help="Every many steps you want to save.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment.")
    parser.add_argument("--sim_device", type=str, default=None, help="Device to run the simulation on. If not specified it defaults to `device`")
    parser.add_argument("--sweep_id", type=str, default=None, help="W&B sweep ID to run an agent for.")
    parser.add_argument("--wandb_run",type=str,default=None,help="Id of the wandb run to load when `resume`=True. If -1: will use the last run id.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Saved model checkpoint path. If `None` will load the last checkpoint in given `experiment_name` folder.")
    AppLauncher.add_app_launcher_args(parser)

    # parse user-provided arguments
    args_cli, hydra_args = parser.parse_known_args()
    if args_cli.video:  # enable cameras to record video
        args_cli.enable_cameras = True
    if args_cli.seed == -1:  # randomization seed
        args_cli.seed = random.randint(0, 10000)
        print(f"SEED {args_cli.seed}")

    # clear out sys.argv for Hydra
    sys.argv = (
        [sys.argv[0]]
        + hydra_args
        + ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"]
    )  # disable hydra logging

    return args_cli


def get_isaac_cfg(args_cli: argparse.Namespace):
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
    from isaaclab_tasks.utils.hydra import hydra_task_config

    env_cfg, agent_cfg = None, None

    # Hydra configs management
    @hydra_task_config(args_cli.task, f"{args_cli.agent}_cfg_entry_point")
    def get_hydra_cfg(_env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg: dict):
        nonlocal env_cfg, agent_cfg
        # override configurations with non-hydra CLI arguments
        _env_cfg.seed = args_cli.seed if args_cli.seed is not None else _agent_cfg["seed"]
        _agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else _agent_cfg["seed"]
        args_cli.seed = _agent_cfg["seed"] if args_cli.seed is None else args_cli.seed

        _env_cfg.sim.device = (
            args_cli.sim_device if args_cli.sim_device is not None
            else args_cli.device if args_cli.device is not None
            else _env_cfg.sim.device
        )
        _agent_cfg["device"] = args_cli.device if args_cli.device is not None else _agent_cfg["device"]
        args_cli.device = _agent_cfg["device"] if args_cli.device is None else args_cli.device
        _env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else _env_cfg.scene.num_envs
        if args_cli.max_iterations is not None:
            _agent_cfg["n_timesteps"] = args_cli.max_iterations * _agent_cfg["n_steps"] * _env_cfg.scene.num_envs
        env_cfg, agent_cfg = _env_cfg, _agent_cfg
        _agent_cfg["preprocessor_kwargs"]["task"] = _env_cfg.task if hasattr(_env_cfg, "task") else args_cli.task

        # for ease of hp search (sweep) we override some configs with non-hydra CLI arguments:
        cfgs = ["n_epochs", "batch_size", "num_mini_batch", "gradient_accumulation_steps"]
        for cfg in cfgs:
            if getattr(args_cli, cfg) is not None:
                _agent_cfg[cfg] = getattr(args_cli, cfg)

    get_hydra_cfg()

    return env_cfg, agent_cfg


def get_cfg(args_cli: argparse.Namespace) -> Dict[str, Any] | T:
    agent_cfg = {}

    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_config_path = os.path.normpath(os.path.join(current_dir, "../configs/agents"))

    @hydra.main(config_path=abs_config_path, config_name=args_cli.agent, version_base="1.3")
    def hydra_load_config(_cfg: DictConfig, _agent_cfg=None) -> None:
        nonlocal agent_cfg

        if len(args_cli.agent.split("/")) > 1:  # clean dict
            for i in range(len(args_cli.agent.split("/")) - 1):
                _cfg = OmegaConf.select(_cfg, args_cli.agent.split("/")[i])
        _agent_cfg = OmegaConf.to_container(_cfg, resolve=True)
        _agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else _agent_cfg["seed"]
        args_cli.seed = _agent_cfg["seed"] if args_cli.seed is None else args_cli.seed
        _agent_cfg["device"] = args_cli.device if args_cli.device is not None else _agent_cfg["device"]
        args_cli.device = _agent_cfg["device"] if args_cli.device is None else args_cli.device
        if args_cli.max_iterations is not None:
            _agent_cfg["n_timesteps"] = args_cli.max_iterations * _agent_cfg["n_steps"] * args_cli.num_envs
        _agent_cfg["preprocessor_kwargs"]["task"] = _agent_cfg["env_cfg"]["task"] if ("env_cfg" in _agent_cfg and "task" in _agent_cfg["env_cfg"]) else args_cli.task

        # for ease of hp search (sweep) we override some configs with non-hydra CLI arguments:
        cfgs = ["n_epochs", "batch_size", "num_mini_batch", "gradient_accumulation_steps"]
        for cfg in cfgs:
            if getattr(args_cli, cfg) is not None:
                _agent_cfg[cfg] = getattr(args_cli, cfg)

        agent_cfg = lists_to_tuples(_agent_cfg)

    hydra_load_config()

    # If a config class is provided, instantiate it
    if getattr(args_cli, "config_class", None) is not None:
        config_class = args_cli.config_class
        if is_dataclass(config_class):
            expected_params = set(config_class.__dataclass_fields__.keys())
            filtered_config = {k: v for k, v in agent_cfg.items() if k in expected_params}
            agent_cfg = filtered_config
        agent_cfg = config_class(**agent_cfg)

    return agent_cfg
