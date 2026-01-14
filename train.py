"""Script to train RL agent using Stable Baselines3."""

import os
from copy import deepcopy

import gymnasium as gym

from algos import *  # noqa: F401,F403
from common.callbacks import CheckpointCallback
from common.cfg_helpers import get_args, get_cfg, get_isaac_cfg
from common.logger import Logger
from common.utils import get_checkpoint_path, print_dict

# Resume policies making sure to update/freeze the loaded policy args
OVERWRITE_POLICY_ARGS = True


def main():
    global _args_cli, _env_cfg, _agent_cfg
    args_cli, env_cfg, agent_cfg = deepcopy(_args_cli), deepcopy(_env_cfg), deepcopy(_agent_cfg)

    if args_cli.envsim == "isaaclab":
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    if args_cli.envsim == "aloha":
        import gym_aloha  # noqa: F401
    if args_cli.envsim == "maniskill":
        import mani_skill.envs  # noqa: F401
    # **customize at necessity with required import**

    from common.envs.sb3_env_wrapper import Sb3EnvStdWrapper, process_sb3_cfg

    def create_env():
        env = gym.make(args_cli.task, **env_cfg)
        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(logger.log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Env specific's wrappers
        if args_cli.envsim == "aloha":
            from common.envs.aloha_wrapper import AlohaStdWrapper

            env = AlohaStdWrapper(env)
        if args_cli.envsim == "maniskill":
            from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

            from common.envs.maniskill_wrapper import (
                FlattenActionSpaceWrapper,
                ManiSkillEnvStdWrapper,
            )

            env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=True, record_metrics=False)
            env = FlattenActionSpaceWrapper(env)
            env = ManiSkillEnvStdWrapper(env)
        elif args_cli.envsim == "isaaclab" and isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        # **customize at necessity with required wrapper other envs**

        # SB3 wrapper env specific to data space (common between different data sources)
        env = Sb3EnvStdWrapper(env, backand_device=args_cli.device)
        return env

    # resume training
    root_dir = os.path.join(
        "save",
        (
            f"{args_cli.task}_{args_cli.experiment_name}"
            if args_cli.task and args_cli.experiment_name
            else args_cli.task or args_cli.experiment_name or ""
        ),
    )
    log_folder = None
    if args_cli.resume:
        checkpoint_path = (
            args_cli.checkpoint if args_cli.checkpoint else get_checkpoint_path(root_dir, ".*", "model_.*.zip")
        )
        root_dir, log_folder = os.path.split(os.path.dirname(checkpoint_path))

    logger = Logger(args_cli, log_root=root_dir, log_folder=log_folder)

    if args_cli.sweep_id and args_cli.wandb:
        import wandb

        for k, v in wandb.run.config.items():
            k = k.split(".")
            cfg = agent_cfg
            for subk in k[:-1]:
                cfg = cfg[subk]
            cfg[k[-1]] = v

    if (args_cli.resume and OVERWRITE_POLICY_ARGS) or not args_cli.resume:
        logger.log_hp(env_cfg, os.path.join(logger.log_dir, "params", "env.yaml"))
        logger.log_hp(agent_cfg, os.path.join(logger.log_dir, "params", "agent.yaml"))
    checkpoint_callback = CheckpointCallback(
        save_freq=args_cli.save_interval, save_path=logger.log_dir, name_prefix="model", verbose=2
    )
    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    agent_class = eval(agent_cfg.pop("agent_class"))
    tot_timesteps = int(agent_cfg.pop("n_timesteps"))

    # Create and wrap the environment
    env = create_env()

    # run agent
    if args_cli.resume:
        logger.log(f"Resuming from checkpoint {checkpoint_path}")
        agent = agent_class.load(checkpoint_path, env, overwrite_policy_arguments=OVERWRITE_POLICY_ARGS, **agent_cfg)
    else:
        agent = agent_class(env=env, verbose=1, **agent_cfg)
    agent.set_logger(logger)

    agent.learn(
        total_timesteps=tot_timesteps - agent.num_timesteps,
        callback=checkpoint_callback,  # [checkpoint_callback, agent_class.EvalPolicy(5, env)],
        reset_num_timesteps=False,
        progress_bar=True,
    )

    # save&close
    agent.save(os.path.join(logger.log_dir, "model"))
    env.close()


if __name__ == "__main__":
    # Get arguments
    global _args_cli, _env_cfg, _agent_cfg
    _args_cli = get_args()

    if _args_cli.envsim == "isaaclab":
        from isaaclab.app import AppLauncher

        # launch omniverse simulator app
        app_launcher = AppLauncher(_args_cli)
        simulation_app = app_launcher.app
        # ONLY AFTER APP LAUNCH
        from configs import *  # noqa: F401,F403 # Note: we import custom envs configs here

        _env_cfg, _agent_cfg = get_isaac_cfg(_args_cli)  # We don't need agent_cfg for inference
        _env_cfg = {"cfg": _env_cfg}

    elif _args_cli.envsim == "aloha":
        assert _args_cli.num_envs == 1, "Multiple environment are not supported by Aloha."
        _agent_cfg = get_cfg(_args_cli)
        _env_cfg = {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array" if _args_cli.video else None,
        }

    elif _args_cli.envsim == "maniskill":
        from sapien import Pose

        _agent_cfg = get_cfg(_args_cli)

        _env_cfg = {
            "obs_mode": _agent_cfg["env_cfg"].get("obs_mode", "state+rgb"),
            "control_mode": _agent_cfg["env_cfg"].get("control_mode", "pd_joint_delta_pos"),
            "sim_backend": _args_cli.sim_device,
            "num_envs": _args_cli.num_envs,
            "render_mode": (
                _agent_cfg["env_cfg"].get("render_mode", "rgb_array") if not _args_cli.video else "rgb_array"
            ),
        }
        # Get task-specific sensor configs from config file
        config_sensor_configs = _agent_cfg["env_cfg"].get("sensor_configs")
        if config_sensor_configs is not None:
            _env_cfg["sensor_configs"] = (
                config_sensor_configs[_args_cli.task] if _args_cli.task in config_sensor_configs else {}
            )
            # Process sensor configs and convert pose arrays back to Pose objects
            for camera_name, camera_cfg in _env_cfg["sensor_configs"].items():
                _env_cfg["sensor_configs"][camera_name] = camera_cfg.copy()
                if (
                    "pose" in camera_cfg
                    and isinstance(camera_cfg["pose"], (list, tuple))
                    and len(camera_cfg["pose"]) == 7
                ):
                    p = camera_cfg["pose"][:3]
                    q = camera_cfg["pose"][3:]
                    _env_cfg["sensor_configs"][camera_name]["pose"] = Pose(p=p, q=q)
        del _agent_cfg["env_cfg"]

    # launch script
    if _args_cli.sweep_id is not None:
        import wandb

        if not _args_cli.wandb:
            print("Since `sweep_id` has been specified WandB logging is enabled in automatic!")
            _args_cli.wandb = True
        wandb.agent(_args_cli.sweep_id, main)
    else:
        main()

    if _args_cli.envsim == "isaaclab":
        simulation_app.close()
    # **customize at necessity with required functions call for termination**
