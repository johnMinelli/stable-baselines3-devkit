"""Script to run inference/testing with a trained RL agent using Stable Baselines3."""

import os

import gymnasium as gym

from algos import *  # noqa: F401,F403
from common.cfg_helpers import get_args, get_cfg, get_isaac_cfg
from common.utils import get_checkpoint_path


def main(args_cli, env_cfg, agent_cfg):
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

            env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=False, record_metrics=True)
            env = FlattenActionSpaceWrapper(env)
            env = ManiSkillEnvStdWrapper(env)
        elif args_cli.envsim == "isaaclab" and isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        # **customize at necessity with required wrapper other envs**
        env.reset()

        # SB3 wrapper env specific to data space (common between different data sources)
        env = Sb3EnvStdWrapper(env, backand_device=args_cli.device)
        return env

    # Load the trained model
    checkpoint_path = (
        args_cli.checkpoint
        if args_cli.checkpoint
        else get_checkpoint_path(
            os.path.join(
                "save",
                (
                    f"{args_cli.task}_{args_cli.experiment_name}"
                    if args_cli.task and args_cli.experiment_name
                    else args_cli.task or args_cli.experiment_name or ""
                ),
            ),
            ".*",
            "model_.*.zip",
        )
    )
    if not checkpoint_path:
        raise ValueError("No checkpoint found! Please specify a valid checkpoint path.")
    args_cli.checkpoint = checkpoint_path

    # Create and wrap the environment
    env = create_env()

    # Load the trained agent
    print(f"Loading model from {checkpoint_path}")
    agent_cfg = process_sb3_cfg(agent_cfg)
    agent = eval(agent_cfg.pop("agent_class")).load(checkpoint_path, env, overwrite_policy_arguments=False, **agent_cfg)

    success, mean_reward, mean_length = agent.predict_episodes(
        env=env, n_eval_episodes=args_cli.val_episodes, deterministic=True
    )
    print(f"success_rate: {success:.2f}")
    print(f"mean_reward: {mean_reward:.2f}")
    print(f"mean_episode_length: {mean_length:.2f}")

    env.close()


if __name__ == "__main__":
    args_cli = get_args()

    if args_cli.envsim == "isaaclab":
        from isaaclab.app import AppLauncher

        # launch omniverse simulator app
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        # ONLY AFTER APP LAUNCH
        from configs import *  # noqa: F401,F403 # Note: we import custom envs configs here

        env_cfg, agent_cfg = get_isaac_cfg(args_cli)  # We don't need agent_cfg for inference
        env_cfg = {"cfg": env_cfg}

    elif args_cli.envsim == "aloha":
        assert args_cli.num_envs == 1, "Multiple environment are not supported by Aloha."
        agent_cfg = get_cfg(args_cli)
        env_cfg = {
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array" if args_cli.video else None,
        }

    elif args_cli.envsim == "maniskill":
        from sapien import Pose

        agent_cfg = get_cfg(args_cli)

        env_cfg = {
            "obs_mode": agent_cfg["env_cfg"].get("obs_mode", "state+rgb"),
            "control_mode": agent_cfg["env_cfg"].get("control_mode", "pd_joint_delta_pos"),
            "sim_backend": args_cli.sim_device,
            "num_envs": args_cli.num_envs,
            "render_mode": agent_cfg["env_cfg"].get("render_mode", "rgb_array") if not args_cli.video else "rgb_array",
        }
        # Get task-specific sensor configs from config file
        config_sensor_configs = agent_cfg["env_cfg"].get("sensor_configs", {})
        env_cfg["sensor_configs"] = (
            config_sensor_configs[args_cli.task] if args_cli.task in config_sensor_configs else {}
        )
        # Process sensor configs and convert pose arrays back to Pose objects
        for camera_name, camera_cfg in env_cfg["sensor_configs"].items():
            env_cfg["sensor_configs"][camera_name] = camera_cfg.copy()
            if "pose" in camera_cfg and isinstance(camera_cfg["pose"], (list, tuple)) and len(camera_cfg["pose"]) == 7:
                p = camera_cfg["pose"][:3]
                q = camera_cfg["pose"][3:]
                env_cfg["sensor_configs"][camera_name]["pose"] = Pose(p=p, q=q)

    # launch script
    main(args_cli, env_cfg, agent_cfg)

    if args_cli.envsim == "isaaclab":
        simulation_app.close()
    # **customize at necessity with required functions call for termination**
