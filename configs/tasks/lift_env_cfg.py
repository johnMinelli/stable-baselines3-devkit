# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import (  # CameraCfg,; ContactSensorCfg,
    FrameTransformerCfg,
    TiledCameraCfg,
)
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# from isaaclab_tasks.manager_based.manipulation.lift import mdp
from configs.tasks.envs import mdp
from configs.tasks.envs.lift_env import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class CustomFrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # self.rewards.undesired_contacts1 = RewTerm(
        #     func=mdp.undesired_contacts,
        #     weight=-0.1,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces_table1", body_names=["gripperStator"]),
        #         "threshold": 1.0,
        #     },
        # )
        # self.rewards.undesired_contacts2 = RewTerm(
        #     func=mdp.undesired_contacts,
        #     weight=-0.1,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces_table2", body_names=["gripperMover"]),
        #         "threshold": 1.0,
        #     },
        # )
        # self.rewards.desired_contacts1 = RewTerm(
        #     func=mdp.desired_contacts,
        #     weight=2.0,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces_table1", body_names=["gripperMover"]),
        #         "threshold": 1.0,
        #     },
        # )
        # self.rewards.desired_contacts2 = RewTerm(
        #     func=mdp.desired_contacts,
        #     weight=2.0,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces_table2", body_names=["gripperStator"]),
        #         "threshold": 1.0,
        #     },
        # )

        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_depth_frame",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=1.0, horizontal_aperture=50, clipping_range=(0.1, 20)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.01924, 0.24831, -0.00744), rot=(0.5784, -0.7040, 0.1144, -0.3974), convention="world")  # gripper panda
        )

        # self.scene.base = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/BaseCamera",
        #     update_period=0.1,
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=11.0, focus_distance=1.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
        #     ),
        #     offset=TiledCameraCfg.OffsetCfg(pos=(0.45, 0.5, 0.05), rot=(0.7071, 0.0, 0.0, -0.7071), convention="world")  # table base


@configclass
class FrankaCubeLiftEnvCfg_PLAY(CustomFrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
