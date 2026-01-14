# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_lifted_throw(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    maximal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = None,
) -> torch.Tensor:
    """Reward the agent for lifting the object and track the state of throwing."""
    # Get object instance
    object: RigidObject = env.scene[object_cfg.name]
    state = env.command_manager.cfg.__getattribute__(command_name).value
    object_height = object.data.root_pos_w[:, 2]
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    lifting_mask = (state == 0) & (object_height > minimal_height) & (object_height < maximal_height)
    reward[lifting_mask] = (object_height[lifting_mask] - minimal_height) / (maximal_height - minimal_height)

    throw_ready_mask = (state == 0) & (object_height >= maximal_height)
    if throw_ready_mask.any():
        env.command_manager.cfg.__getattribute__(command_name).value[throw_ready_mask] = 1

    return reward


def desired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    assert contact_sensor.data.force_matrix_w is not None, "You are not filtering any contact sensor."
    net_contact_forces = contact_sensor.data.force_matrix_w.sum(2)
    is_contact = torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0].abs() > threshold
    # sum over contacts for each environment
    return is_contact


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    if contact_sensor.data.force_matrix_w is not None:
        net_contact_forces = (
            contact_sensor.data.force_matrix_w.sum(2) - contact_sensor.data.net_forces_w
        )  # ignore filtered objects
        is_contact = torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        # sum over contacts for each bodyb element
        if len(is_contact.shape) > 1:
            is_contact = is_contact.sum(1)

        return is_contact

    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each body element
    if len(is_contact.shape) > 1:
        is_contact = is_contact.sum(1)

    return is_contact


# 0 rew at contact but 0.6 at no contact
# lift seems to trigger  state change but no reward


def desired_contacts_throw(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = None,
) -> torch.Tensor:
    """Reward based on contact state and throwing distance."""
    # Get necessary instances
    object: RigidObject = env.scene[object_cfg.name]
    state = env.command_manager.cfg.__getattribute__(command_name).value
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Initialize reward tensor
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    assert contact_sensor.data.force_matrix_w is not None, "You are not filtering any contact sensor."
    net_contact_forces = contact_sensor.data.force_matrix_w.sum(2)
    is_contact = torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1), dim=1)[0].abs() > threshold

    # Case 1: In lifting state (0) and touching
    touching_mask = (state == 0) & is_contact
    reward[touching_mask] = 1  # Fixed low reward for maintaining contact

    # Case 2: In throwing state (1) and not touching
    throwing_mask = (state == 1) & (~is_contact)
    current_pos = object.data.root_pos_w[..., :2] - env.scene.env_origins[:, :2]  # Get x,y of current position
    throwing_distance = torch.norm(current_pos, dim=1)
    # Reward proportional to throwing distance
    reward[throwing_mask] = throwing_distance[throwing_mask] * 50

    return reward


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    if hasattr(env.cfg, "reward_memory"):
        if "object_ee_distance" not in env.cfg.reward_memory:
            env.cfg.reward_memory["object_ee_distance"] = torch.ones(env.num_envs, dtype=torch.float, device=env.device)
        env.cfg.reward_memory["object_ee_distance"][torch.isnan(env.cfg.reward_memory["object_ee_distance"])] = 1

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    reward = 1 - torch.tanh(object_ee_distance / std)  # 0 bad, +1 good

    if hasattr(env.cfg, "reward_memory"):
        reward[object_ee_distance >= env.cfg.reward_memory["object_ee_distance"]] = 0
        env.cfg.reward_memory["object_ee_distance"] = torch.min(
            env.cfg.reward_memory["object_ee_distance"], object_ee_distance
        )

    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    if hasattr(env.cfg, "reward_memory"):
        if "object_goal_distance" not in env.cfg.reward_memory:
            env.cfg.reward_memory["object_goal_distance"] = torch.ones(
                env.num_envs, dtype=torch.float, device=env.device
            )
        env.cfg.reward_memory["object_goal_distance"][torch.isnan(env.cfg.reward_memory["object_goal_distance"])] = 1

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    object_goal_distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    reward = (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(object_goal_distance / std)
    )  # 0 bad, +1 good

    if hasattr(env.cfg, "reward_memory"):
        reward[(object.data.root_pos_w[:, 2] > minimal_height) & (object_goal_distance >= env.cfg.reward_memory["object_goal_distance"])] = 0
        env.cfg.reward_memory["object_goal_distance"][(object.data.root_pos_w[:, 2] > minimal_height)] = torch.min(
            env.cfg.reward_memory["object_goal_distance"], object_goal_distance
        )[(object.data.root_pos_w[:, 2] > minimal_height)]

    return reward
