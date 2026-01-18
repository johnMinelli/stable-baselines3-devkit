"""
Register here custom Gym environments.
Specify the env class (`entry_point`), and the agent configuration (`env_cfg_entry_point`) to allow the utilization of custom yaml configs (but also the Isaac' default one scan be specified)

e.g.
```
gym.register(
    id="Isaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "ppo_cfg_entry_point": f"{agents_classic.__name__}:sb3_ppo_cfg.yaml", <-- library default agent config
        "custom_ppo_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml", <-- custom default agent config
    },
)
```

> python train.py --task Isaac-Cartpole-v0 --envsim isaaclab --agent ppo
> python train.py --task Isaac-Cartpole-v0 --envsim isaaclab --agent custom_ppo

"""

import gymnasium as gym
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import (
    AnymalDFlatEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

from . import agents
from .tasks.lift_env_cfg import CustomFrankaCubeLiftEnvCfg

gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
        "custom_ppo_mlp_cfg_entry_point": f"{agents.__name__}.Isaac:lift_ppo_mlp_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Custom-Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": CustomFrankaCubeLiftEnvCfg,
        "custom_ppo_mlp_cfg_entry_point": f"{agents.__name__}.Isaac:lift_ppo_mlp_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Velocity-Flat-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AnymalDFlatEnvCfg,
        "custom_sac_mlp_cfg_entry_point": f"{agents.__name__}.Isaac:velocity_sac_mlp_cfg.yaml",
        "custom_ppo_mlp_cfg_entry_point": f"{agents.__name__}.Isaac:velocity_ppo_mlp_cfg.yaml",
        "custom_ppo_lstm_cfg_entry_point": f"{agents.__name__}.Isaac:velocity_ppo_lstm_cfg.yaml",
        "custom_ppo_tr_cfg_entry_point": f"{agents.__name__}.Isaac:velocity_ppo_tr_cfg.yaml",
    },
)
