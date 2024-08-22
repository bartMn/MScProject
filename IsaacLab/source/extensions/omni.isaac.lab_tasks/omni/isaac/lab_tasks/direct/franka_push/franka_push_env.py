# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom, Gf, Sdf

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform

from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sensors import CameraCfg, Camera
import torch
from torchvision import transforms
from PIL import Image
import sys

import numpy as np
import os
import signal
import random
import time


module_path = os.path.dirname(os.path.abspath(__file__))
#module_path = os.path.join(module_path, f"..")
if module_path not in sys.path:
    sys.path.append(module_path)
from supp import SupportClass
from configclass import FrankaPushEnvCfg


TEST_AND_SAVE_SENSORS =  os.getenv('TEST_AND_SAVE_SENSORS', 'false').lower() == 'true'
ERASE_EXISTING_DATA =  os.getenv('ERASE_EXISTING_DATA', 'false').lower() == 'true'
GATHERED_DATA_ROOT = os.getenv('RECORDED_DATA_DIR')
MAX_FRAMES_TO_SAVE = 210
SIM_FRAME_TO_START_RECORDING = 8



class FrankaPushEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaPushEnvCfg

    def __init__(self, cfg: FrankaPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.colorize_semantic_segmentation = self.cfg.colorize_semantic_segmentation
        self.camera_modes = self.cfg.camera_modes
        self.cam_sensor_width = self.cfg.cam_sensor_width
        self.cam_sensor_height = self.cfg.cam_sensor_height
        self.cubeA_size = self.cfg.cubeA_size
        self.cubeB_size = self.cfg.cubeB_size

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.current_frame = 0
        self.simulator_frame = 0
        self.reward_settings = {
            "r_dist_scale": self.cfg.r_dist_scale,
            "r_lift_scale": self.cfg.r_lift_scale ,
            "r_align_scale": self.cfg.r_align_scale,
            "r_stack_scale": self.cfg.r_stack_scale,
            "r_push_scale": self.cfg.r_push_scale,
            "r_finished_scale": self.cfg.r_finished_scale
        }
        self.handles = {
            # Franka
            "hand": self._robot.find_bodies("panda_hand")[0][0],
            "leftfinger_tip": self._robot.find_bodies("panda_leftfinger")[0][0],
            "rightfinger_tip": self._robot.find_bodies("panda_rightfinger")[0][0],
            "grip_site": self._robot.find_bodies("panda_leftfinger")[0][0],
            # Cubes
            "cubeA_body_handle": self._cubeA.find_bodies("cubeA")[0][0],
            "cubeB_body_handle": self._cubeB.find_bodies("cubeB")[0][0],
        }

        self.cubeA_size = self.cubeA_size
        self.cubeB_size = self.cubeB_size

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        #drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        #self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        #self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        #self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
        #    (self.num_envs, 1)
        #)
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        #self.drawer_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
        #    (self.num_envs, 1)
        #)

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        #self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        #self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        #self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        #print(100*"=")
        #print("INIT DONE")
        #print(100*"=")
        
        if TEST_AND_SAVE_SENSORS:
            self.CAMERA_LIST = list()
            self.MOVEABLE_CAMERAS_LIST = list()

            eyes0 = torch.tensor([-0.25, 0.0, 1.15], device=self.device, dtype=torch.float32) + self.scene.env_origins
            targets0 = torch.tensor([1.0, 0.0, 1.15], device=self.device, dtype=torch.float32) + self.scene.env_origins
            self.tiled_camera0.set_world_poses_from_view(eyes0, targets0)
            self.CAMERA_LIST.append(self.tiled_camera0)

            eyes1 = torch.tensor([0.5, 0.75, 1.5], device=self.device, dtype=torch.float32) + self.scene.env_origins
            targets1 = torch.tensor([0.5, 0.20, 1.05], device=self.device, dtype=torch.float32) + self.scene.env_origins
            self.tiled_camera1.set_world_poses_from_view(eyes1, targets1)
            self.CAMERA_LIST.append(self.tiled_camera1)

            eyes2 = torch.tensor([0.5, -0.75, 1.5], device=self.device, dtype=torch.float32) + self.scene.env_origins
            targets2 = torch.tensor([0.5, -0.20, 1.05], device=self.device, dtype=torch.float32) + self.scene.env_origins
            self.tiled_camera2.set_world_poses_from_view(eyes2, targets2)
            self.CAMERA_LIST.append(self.tiled_camera2)

            self.CAMERA_LIST.append(self.tiled_camera3)
            self.CAMERA_LIST.append(self.tiled_camera4)
            self.MOVEABLE_CAMERAS_LIST.append(self.tiled_camera4)

            

            self.sensors = SupportClass(gathered_data_root = GATHERED_DATA_ROOT,
                                        erase_exiting_data= ERASE_EXISTING_DATA,
                                        cam_sensor_width= self.cam_sensor_width,
                                        cam_sensor_height= self.cam_sensor_height,
                                        cam_data_type = self.camera_modes,
                                        num_envs = self.num_envs,
                                        num_of_cameras = len(self.CAMERA_LIST))


            
    def _setup_scene(self):
        self.set_seed_based_on_time()

        self._robot = Articulation(self.cfg.robot)
        #self._cabinet = Articulation(self.cfg.cabinet)
        self.scene.articulations["robot"] = self._robot
        #self.scene.articulations["cabinet"] = self._cabinet

        self._cubeA = RigidObject(self.cfg.cubeA)
        self.scene.rigid_objects["cubeA"] = self._cubeA
        #self._cubeA.get_prim().GetAttribute("semantic:type").Set("typeA")
        #self._cubeA.get_prim().GetAttribute("semantic:label").Set("labelA")
#
        self._cubeB = RigidObject(self.cfg.cubeB)
        self.scene.rigid_objects["cubeB"] = self._cubeB
        
        #stage = get_current_stage()
        #cubeB_prim = stage.GetPrimAtPath("/World/envs/env_.*/cubeB")
        #cubeB_prim.GetAttribute("semantic:type").Set("typeB")
        #cubeB_prim.GetAttribute("semantic:label").Set("labelB")

        #self._table = Articulation(self.cfg.table)
        #self.scene.articulations["table"] = self._table
        # spawn a usd file of a table into the scene
        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", semantic_tags = [("class", "Table")])
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=(0.5-0.025, 0.0, 1.05), orientation=(1.41/2, 0.0, 0.0, -1.41/2))

        

        if TEST_AND_SAVE_SENSORS:
            self.tiled_camera0 = Camera(self.cfg.tiled_camera0)
            self.tiled_camera1 = Camera(self.cfg.tiled_camera1)
            self.tiled_camera2 = Camera(self.cfg.tiled_camera2)
            self.tiled_camera3 = Camera(self.cfg.tiled_camera3)
            self.tiled_camera4 = Camera(self.cfg.tiled_camera4)
        

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        

        # add lights
        random.seed(time.time())
        num_of_lights = random.randint(1, 6)
        #print(100*"+=")
        #print(100*"+=")
        #print(f"num_of_lights = {num_of_lights}")
        #print(100*"+=")
        #print(100*"+=")
        #pid = os.getpid()  # Get the current process ID
        #os.kill(pid, signal.SIGKILL) 
        for i in range(num_of_lights):
            light_intensity_base = random.randint(1, 9)
            light_intensity_exp = random.randint(4, 6)
            light_cfg = sim_utils.SphereLightCfg(intensity=(light_intensity_base)*(10**light_intensity_exp),
                                                 color=(random.uniform(0.5, 1.0), random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)),
                                                 radius= 0.15)
            light_cfg.func(f"/World/Light{i}", light_cfg)
            stage = get_current_stage()
            light_prim = stage.GetPrimAtPath(f"/World/Light{i}")
            light_translation = Gf.Vec3f(random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), random.uniform(2.0, 2.2))  # Set your desired position here
            light_prim.GetAttribute("xformOp:translate").Set(light_translation)

        stage.RemovePrim("Environment/defaultLight")



    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        robot_hand_hight = self._robot.data.body_pos_w[:, self.handles["hand"], 2]
        
        
        #print(f"self._cubeA.data.body_pos_w.shape = {self._cubeA.data.body_pos_w.shape}")
        #print(f"(self._robot.data.body_pos_w[:, self.handles['grip_site'], :]).shape = {(self._robot.data.body_pos_w[:, self.handles['grip_site'], :]).shape}")
        #eef_pos = self._robot.data.body_pos_w[:, self.handles["hand"], :]
        #eef_quat = self._robot.data.body_quat_w[:, self.handles["hand"], :]
        local_points = torch.tensor([[0, 0, 0.105] for _ in range(self.num_envs)], device = self.device)
        eef_pos = self.transform_points(self._robot.data.body_pos_w[:, self.handles["hand"], :], self._robot.data.body_quat_w[:, self.handles["hand"], :], local_points)
        #print(f"(torch.ones_like(eff_pos[:, 0]) * self.cubeA_size).shape = {(torch.ones_like(eff_pos[:, 0]) * self.cubeA_size).shape}")
        #exit(0)
        states = dict()

        states["robot_hand_hight"] = robot_hand_hight

        states["cubeA_size"] = torch.ones_like(eef_pos[:, 0]) * self.cubeA_size
        states["cubeB_size"] = torch.ones_like(eef_pos[:, 0]) * self.cubeB_size
        states["cubeA_to_cubeB_pos"] = (self._cubeB.data.body_pos_w - self._cubeA.data.body_pos_w).reshape(self.num_envs, 3)
        states["cubeA_pos"] = self._cubeA.data.body_pos_w.reshape(self.num_envs, 3) - self.scene.env_origins 
        states["cubeB_pos"] = self._cubeB.data.body_pos_w.reshape(self.num_envs, 3) - self.scene.env_origins
        states["eef_pos"] = eef_pos - self.scene.env_origins

        #print(f"(self._robot.data.body_pos_w[:, self.handles['rightfinger_tip'], :]).shape = {(self._robot.data.body_pos_w[:, self.handles['rightfinger_tip'], :]).shape}")
        #print(f"robot_right_finger_pos.shape = {robot_right_finger_pos.shape}")
        states["fingers_dist"] = (robot_left_finger_pos - robot_right_finger_pos)
        states["fingers_dist"] = torch.norm(states["fingers_dist"], dim=1, keepdim=True)

        #states["cubeA_to_cubeB_pos_orig"] = self.original_dist_cubes.squeeze()
        #print(f"(states['fingers_dist']).shape = {(states['fingers_dist']).shape}")
        #exit(0)

        if TEST_AND_SAVE_SENSORS:
            self.save_data()

        return self._compute_rewards(states,
                                     self.reward_settings
                                    )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        self.set_seed_based_on_time()
        self._robot.data.default_joint_pos *= 0
        self._robot.data.default_joint_pos += torch.tensor([0.0, -0.18, 0.0, -2.0, 0.0, 2.9416, 0.7854, 0.035, 0.035], device=self.device)

        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        #base_rot = sample_uniform(
        #    0.25,
        #    0.75,
        #    (len(env_ids)),
        #    self.device,
        #)

        
        #base_idx = 0

        #joint_pos[: , base_idx] += self.robot_dof_lower_limits[base_idx] + base_rot* (self.robot_dof_upper_limits[base_idx] - self.robot_dof_lower_limits[base_idx])
        #joint_pos = 0*self._robot.data.default_joint_pos[env_ids] + \
        #            sample_uniform(0.0, 1.0, (len(env_ids), self._robot.num_joints), self.device)
#
        #joint_pos = self.robot_dof_lower_limits + joint_pos* (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
        #print(f"self.robot_dof_lower_limits = {self.robot_dof_lower_limits}")
        #print(f"self.robot_dof_upper_limits = {self.robot_dof_upper_limits}")
        #exit()
        
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        cubeA_pos = torch.Tensor([0.0, 0.0, 1.07+ self.cubeA_size/4])
        cubeA_rot = torch.Tensor([1.0, 0.0, 0.0, 0.0])
        cubeB_pos = torch.Tensor([0.0, 0.0, 1.07+ self.cubeB_size/4])
        cubeB_rot = torch.Tensor([1.0, 0.0, 0.0, 0.0])
        cubeAB_lin_vel = torch.zeros(3, dtype=torch.bool)
        cubeAB_rot_vel = torch.zeros(3, dtype=torch.bool)
        
        # Concatenate cubeA data to form one row
        cubeA_data = torch.cat((cubeA_pos, cubeA_rot, cubeAB_lin_vel.float(), cubeAB_rot_vel.float()))
        # Concatenate cubeB data to form one row
        cubeB_data = torch.cat((cubeB_pos, cubeB_rot, cubeAB_lin_vel.float(), cubeAB_rot_vel.float()))
        
        # Repeat the rows N times to form N x 13 tensor
        cubeA_tensor = cubeA_data.unsqueeze(0).repeat(len(env_ids), 1)
        cubeB_tensor = cubeB_data.unsqueeze(0).repeat(len(env_ids), 1)
        
        cubeA_tensor = cubeA_tensor.to(self.device)
        cubeB_tensor = cubeB_tensor.to(self.device)

        
        #cubeA_tensor[:, 0] += sample_uniform(0.05, 1.0, (len(env_ids)), self.device)
        #cubeA_tensor[:, 1] += sample_uniform(-0.40, 0.40, (len(env_ids)), self.device)
        
        cubeA_tensor[:, :2] += self._reset_init_cube_state(cube_name = "A", env_ids = env_ids)
        cubeB_tensor[:, :2] += self._reset_init_cube_state(cube_name = "B", env_ids = env_ids, other_cube_positions = cubeA_tensor[:, :2])

        self.cubeA_init_pos = cubeA_tensor[:, :3].clone()
        self.cubeB_init_pos = cubeB_tensor[:, :3].clone()

        #original_directions_cubes = ((self.cubeB_init_pos - self.cubeA_init_pos).reshape(self.num_envs, 3))
        #self.original_dist_cubes = torch.norm(original_directions_cubes, dim= -1).view(-1, 1) 

        cubeA_tensor[:, :3] += self.scene.env_origins
        cubeB_tensor[:, :3] += self.scene.env_origins
        #cubeA_tensor[:, 2] += 1.05
        #cubeB_tensor[:, 2] += self.scene.env_origins
        #cubeB_tensor[:, 0] += sample_uniform(0.05, 1.0, (len(env_ids)), self.device)
        #cubeB_tensor[:, 1] += sample_uniform(-0.40, 0.40, (len(env_ids)), self.device)
        
        #print(f"cubeA_tensor.shape= {cubeA_tensor.shape}")
        #print(f"self.scene.env_origins.shape= {self.scene.env_origins.shape}")
        #exit()

        self._cubeA.write_root_state_to_sim(root_state= cubeA_tensor, env_ids=env_ids)
        self._cubeB.write_root_state_to_sim(root_state= cubeB_tensor, env_ids=env_ids)
        #self._cubeA.reset(env_ids)
        #self._cubeB.reset(env_ids)
        # cabinet state
        #zeros = torch.zeros((len(env_ids), self._cabinet.num_joints), device=self.device)
        #self._cabinet.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        #self._compute_intermediate_values(env_ids)

    def _reset_init_cube_state(self, cube_name, env_ids, other_cube_positions= None):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        # If env_ids is None, we reset all the envs
        #if env_ids is None:
        #    env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values

        y_min = -0.40
        y_max = 0.40
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 2, device=self.device)

        # Get correct references depending on which one was selected
        if cube_name.lower() == 'a':
            check_valid= False
            x_min = 0.45
            x_max = 0.75
            #this_cube_state_all = self._init_cubeA_state
            #other_cube_state = self._init_cubeB_state[env_ids, :]
        elif cube_name.lower() == 'b':
            x_min = 0.45
            x_max = 0.75
        
            check_valid= True
            #this_cube_state_all = self._init_cubeB_state
            other_cube_state = other_cube_positions
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube_name}")

        
        
        # Set z value, which is fixed height
        sampled_cube_state[:, 0] = sample_uniform(x_min, x_max, (len(env_ids)), self.device)
        sampled_cube_state[:, 1] = sample_uniform(y_min, y_max, (len(env_ids)), self.device)


        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            cubeA_size_tensor = torch.ones_like(sampled_cube_state[:, 0]) * self.cubeA_size
            cubeB_size_tensor = torch.ones_like(sampled_cube_state[:, 0]) * self.cubeB_size
            # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
            min_dists = (cubeA_size_tensor+ cubeB_size_tensor) * np.sqrt(2) / 2.0

            # We scale the min dist by 2 so that the cubes aren't too close together
            min_dists = min_dists * 2.25
            # Sampling is "centered" around middle of table
            #centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, 0] = x_min + (x_max-x_min)* (torch.rand_like(sampled_cube_state[active_idx, 0])) 
                sampled_cube_state[active_idx, 1] = y_min + (y_max-y_min)* (torch.rand_like(sampled_cube_state[active_idx, 1])) 
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state - other_cube_state, dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        

        return sampled_cube_state

    def _get_observations(self) -> dict:
        
        #abs_dist_cubes = ((self._cubeB.data.body_pos_w - self._cubeA.data.body_pos_w).reshape(self.num_envs, 3)).abs()
        dist_cubes = ((self._cubeB.data.body_pos_w - self._cubeA.data.body_pos_w).reshape(self.num_envs, 3))
        #directions_cubes = ((self._cubeB.data.body_pos_w - self._cubeA.data.body_pos_w).reshape(self.num_envs, 3))
        #dist_cubes =  torch.norm(directions_cubes, dim= -1).view(-1, 1) 
        #print(f"directions_cubes.shape = {directions_cubes.shape}")
        #print(f"dist_cubes.shape = {dist_cubes.shape}")
        #print(f"norm_directions_cubes.shape = {norm_directions_cubes.shape}")
        #norm_directions_cubes = directions_cubes / dist_cubes
        

        cubeA_pos = self._cubeA.data.root_pos_w - self.scene.env_origins
        cubeA_quat = self._cubeA.data.root_quat_w
        #cubeB_pos = self._cubeB.data.root_pos_w - self.scene.env_origins
        #cubeB_quat = self._cubeB.data.root_quat_w
        franka_pos = self._robot.data.joint_pos
        franka_vel = self._robot.data.joint_vel
        #eef_pos = ((self._robot.data.body_pos_w[:, self.handles["leftfinger_tip"], :] + \
        #           self._robot.data.body_pos_w[:, self.handles["rightfinger_tip"], :]) \
        #           / 2) \
        #           - self.scene.env_origins
        #eef_pos = self._robot.data.body_pos_w[:, self.handles["hand"], :] - self.scene.env_origins

        
        #eef_pos = self._robot.data.body_pos_w[:, self.handles["grip_site"], :] - self.scene.env_origins
        eef_quat = self._robot.data.body_quat_w[:, self.handles["hand"], :]
        #abs_dist_eefCubeA = (cubeA_pos - eef_pos).abs()

        local_points = torch.tensor([[0, 0, 0.105] for _ in range(self.num_envs)], device = self.device)
        eef_pos = self.transform_points(self._robot.data.body_pos_w[:, self.handles["hand"], :] - self.scene.env_origins, eef_quat, local_points)
        
        #print(f"to_target.shape = {to_target.shape}")
        #print(f"cubeA_pos.shape = {cubeA_pos.shape}")
        #print(f"franka_pos.shape = {franka_pos.shape}")
        #print(f"eef_pos.shape = {eef_pos.shape}")
        #print(f"cubeA_quat = {cubeA_quat}")
        #print(f"cubeA_quat.shape = {cubeA_quat.shape}")
        #exit()

        obs = torch.cat(
            (
                dist_cubes,
                #abs_dist_cubes,
                #dist_cubes,
                #self.original_dist_cubes,
                #norm_directions_cubes,
                cubeA_pos,
                cubeA_quat,
                #cubeB_pos,
                #cubeB_quat,
                franka_pos,
                franka_vel,
                eef_pos,
                eef_quat,
                #abs_dist_eefCubeA,
            ),
            dim=-1,
        )
        
        #print(100*"@")
        #print("_get_observations DONE")
        #print(100*"@")

        return {"policy": obs}
        #return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # auxiliary methods

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        #drawer_pos = self._cabinet.data.body_pos_w[env_ids, self.drawer_link_idx]
        #drawer_rot = self._cabinet.data.body_quat_w[env_ids, self.drawer_link_idx]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            #self.drawer_grasp_rot[env_ids],
            #self.drawer_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            #drawer_rot,
            #drawer_pos,
            #self.drawer_local_grasp_rot[env_ids],
            #self.drawer_local_grasp_pos[env_ids],
        )

    def _compute_rewards(self, states, reward_settings):


        # Compute per-env physical parameters
        cubeA_size = states["cubeA_size"]
        #cubeB_size = states["cubeB_size"]

        # Define threshold for EE being close to cube A
        #ee_close_threshold = 0.05  # You can adjust this threshold as needed

        # Distance from cubeA to cubeB
        d_ab = torch.norm(states["cubeA_to_cubeB_pos"], dim=-1)
        move_reward = 1 - torch.tanh(10.0 * d_ab)
        #move_reward = 1 - (d_ab / states["cubeA_to_cubeB_pos_orig"])

        #print(f"d_ab.shape = {d_ab.shape}")
        #print(f"states[cubeA_to_cubeB_pos_orig].shape = {states['cubeA_to_cubeB_pos_orig'].shape}")
        #print(f"move_reward.shape = {move_reward.shape}")
        #exit(0)

        # Distance from EE to cubeA
        #d_eef_to_cubeA = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)

        # Check if cubeA is near cubeB horizontally
        cubeA_near_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.020)

        # Check if cubeA is on the ground using cubeB's height as a reference
        cubeA_on_table = torch.abs(states["cubeA_pos"][:, 2] - states["cubeB_pos"][:, 2]) < 0.05
        #cube_on_ground = torch.abs(states["cubeA_pos"][:, 2] + states["cubeB_pos"][:, 2]) < 1.5
        #fingers_apart = states["fingers_dist"] > 0.01
        #fingers_punishment = (fingers_apart * (states["fingers_dist"] * (-1000))).squeeze()
        fingers_together = states["fingers_dist"] < 0.01
        cubes_on_table = torch.abs(states["cubeA_pos"][:, 2] + states["cubeB_pos"][:, 2]) > 1.9
        #print(f"cube_on_ground = {cube_on_ground}")
        
        # Calculate the direction vector from cube A to cube B
        cubeA_to_cubeB_vec = states["cubeB_pos"] - states["cubeA_pos"]
        cubeA_to_cubeB_vec_norm = torch.norm(cubeA_to_cubeB_vec, dim=-1, keepdim=True)
        direction_vec = cubeA_to_cubeB_vec / cubeA_to_cubeB_vec_norm

        # Define the target position for the end effector
        target_pos = states["cubeA_pos"] - direction_vec * (cubeA_size.view(-1, 1) / 1.8)
        target_pos[:, 2] -= cubeA_size / 1.5

        # Distance from EE to the target position
        d_eef_to_target = torch.norm(target_pos - states["eef_pos"], dim=-1)
        ee_target_reward = 1 - torch.tanh(10.0 * d_eef_to_target)

        # Check if EE is close to cube A
        ee_close_to_cubeA = ee_target_reward > 0.65
        #print(f"sum = {torch.sum(ee_close_to_cubeA)}; max = {torch.max(ee_target_reward)}")
        push_reward = cubeA_near_cubeB & cubeA_on_table

        hand_higher_than_cube = (states["robot_hand_hight"] - states["cubeA_pos"][:, 2]) > 0.070
        #punishments = -10 * cube_on_ground

        #print(f"punishment = {punishments}")
        #exit(0)

        # Compose rewards
        rewards = torch.where(
            push_reward,
            reward_settings["r_finished_scale"] * push_reward * ee_close_to_cubeA.to(torch.float),
            reward_settings["r_push_scale"] * move_reward * ee_close_to_cubeA.to(torch.float) + reward_settings["r_align_scale"] * ee_target_reward
        )

        #rewards =(reward_settings["r_finished_scale"] * push_reward * ee_close_to_cubeA.to(torch.float) +
        #          reward_settings["r_push_scale"] * move_reward * ee_close_to_cubeA.to(torch.float) + 
        #          reward_settings["r_align_scale"] * ee_target_reward
        #)


        #print(f"rewards.shape = {rewards.shape}")
        #print(f"punishments.shape = {punishments.shape}")
        #print(f"fingers_punishment.shape = {fingers_punishment.shape}")
        #exit(0)

        return (rewards * fingers_together.squeeze() * cubes_on_table* hand_higher_than_cube) # + punishments #+ fingers_punishment

    def quaternion_to_rotation_matrix(self, quats):
        """
        Converts a batch of quaternions to rotation matrices.

        Args:
            quats: Tensor of shape (N, 4) where N is the batch size.

        Returns:
            A tensor of shape (N, 3, 3) representing rotation matrices.
        """
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        R = torch.zeros((quats.shape[0], 3, 3), device=quats.device)

        R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

        return R

    def transform_points(self, object_positions, quats, local_points):
        """
        Transforms local points to world frame using object positions and quaternions.

        Args:
            object_positions: Tensor of shape (N, 3) where N is the batch size.
            quats: Tensor of shape (N, 4) where N is the batch size.
            local_points: Tensor of shape (N, 3) where N is the batch size.

        Returns:
            A tensor of shape (N, 3) representing the transformed world points.
        """
        # Convert quaternions to rotation matrices
        R = self.quaternion_to_rotation_matrix(quats)

        # Rotate the local points
        local_points_rotated = torch.bmm(R, local_points.unsqueeze(-1)).squeeze(-1)

        # Translate to world frame
        world_points = object_positions + local_points_rotated

        return world_points


    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        #drawer_rot,
        #drawer_pos,
        #drawer_local_grasp_rot,
        #drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        #global_drawer_rot, global_drawer_pos = tf_combine(
        #    drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        #)

        return global_franka_rot, global_franka_pos#, global_drawer_rot, global_drawer_pos

    def set_seed_based_on_time(self):
        
        # Get the current time in seconds since the epoch
        seed = int(time.time())

        # Set the seed for Python random
        random.seed(seed)

        # Set the seed for NumPy random
        np.random.seed(seed)

        # Set the seed for PyTorch random
        torch.manual_seed(seed)

        # If using GPUs, set the seed for all GPU devices
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        ## Ensure deterministic behavior for some operations
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False


    def save_data(self):
        if self.simulator_frame < SIM_FRAME_TO_START_RECORDING:
            self.simulator_frame += 1
            return

        for camera in self.CAMERA_LIST:
            camera.reset([i for i in range(self.num_envs)])

        cameras_data_list = [camera.data.output for camera in self.CAMERA_LIST]
        #print(f"info = {self.CAMERA_LIST[0].data.info}")
        #exit()
        #cam_data = self.tiled_camera.data.output[self.cam_data_type]
        self.sensors.save_rendered_imgs(cameras_data_list, self.current_frame, self.colorize_semantic_segmentation)
        self.sensors.save_dof_states_and_forces(self._robot.data.computed_torque,
                                                    self._robot.data.joint_pos,
                                                    self._robot.data.joint_vel)
            
        self.sensors.save_cubes_position(self._cubeA.data.root_pos_w - self.scene.env_origins,
                                         self._cubeA.data.root_quat_w,
                                         self._cubeA.data.root_lin_vel_b,
                                         self._cubeA.data.root_ang_vel_b,
                                         0)
            
        self.sensors.save_cubes_position(self._cubeB.data.root_pos_w - self.scene.env_origins,
                                         self._cubeB.data.root_quat_w,
                                         self._cubeB.data.root_lin_vel_b,
                                         self._cubeB.data.root_ang_vel_b,
                                         1)
            
        #print(f"self.actions = {self.actions.shape}")
        #pid = os.getpid()  # Get the current process ID
        #os.kill(pid, signal.SIGKILL) 
        self.sensors.save_actions(self.actions)
            
        self.current_frame += 1
        if self.current_frame > MAX_FRAMES_TO_SAVE:
            time.sleep(1)
            pid = os.getpid()  # Get the current process ID
            os.kill(pid, signal.SIGKILL) 
