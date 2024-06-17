# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask


SAVE_SENSORS = True
FRAMES_MAX = 120
CAM_SENSOR_WIDTH = int(1920)
CAM_SENSOR_HEIGHT = int(1080)
GATHERED_DATA_ROOT = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data"

from PIL import Image as im
import threading
import cv2
import random
import time
import shutil
def save_rgb_img(rgb_filename, img_rgb):
        cv2.imwrite(rgb_filename, img_rgb)

def save_depth_img(depth_filename, depth_image):
    depth_image[depth_image == -np.inf] = 0

    # clamp depth image to 10 meters to make output image human friendly
    depth_image[depth_image < -10] = -2

    # flip the direction so near-objects are light and far objects are dark
    normalized_depth = -255.0*(depth_image)#/np.min(depth_image + 1e-4))

    # Convert to a pillow image and write it to disk
    normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
    normalized_depth_image.save(depth_filename)
    #normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (i, j, frame_count))

def save_segmented_img(segmented_filename, segmented_image):
    normalized_depth_image = im.fromarray(segmented_image.astype(np.uint8), mode="L")
    normalized_depth_image.save(segmented_filename)

def save_flow_img(flow_filename, optical_flow_image):

    optical_flow_in_pixels = optical_flow_image#np.zeros(np.shape(optical_flow_image))
    # Horizontal (u)
    #optical_flow_in_pixels[0,0] = CAM_SENSOR_WIDTH*(optical_flow_image[0,0]/2**15)
    # Vertical (v)
    #optical_flow_in_pixels[0,1] = CAM_SENSOR_WIDTH*(optical_flow_image[0,1]/2**15)
    #print(f"optical_flow_in_pixels.shape = {optical_flow_in_pixels.shape}")
    #print(f"optical_flow_in_pixels[0,1].shape = {optical_flow_in_pixels[ : , : 1920].shape}")
    #print(f"optical_flow_in_pixels[0,0].shape = {optical_flow_in_pixels[ : , 1920 : ].shape}")
    #return 

    
    result = np.sqrt(optical_flow_in_pixels[ : , : 1920]**2 + optical_flow_in_pixels[ : ,1920 : ]**2)
    normalized_depth_image = im.fromarray(optical_flow_in_pixels.astype(np.uint8), mode="L")
    normalized_depth_image.save(flow_filename)



def quaternion_from_two_vectors(v_from, v_to):
    dot_product = v_from.dot(v_to)
    if abs(dot_product) + 1 < 1e-6:
        cross_product = gymapi.Vec3(1.0, 0.0, 0.0).cross(v_from)
        if cross_product.length() < 1e-6:
            cross_product = gymapi.Vec3(0.0, 1.0, 0.0).cross(v_from)
        cross_product.normalize()
        return gymapi.Quat(0.0, cross_product.x, cross_product.y, cross_product.z)
    else:
        w = dot_product + v_from.dot(v_to)
        xyz = v_from.cross(v_to)
        return gymapi.Quat(w, xyz.x, xyz.y, xyz.z)

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaCubePush(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.start_time = time.time()
        
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
            "r_push_scale": self.cfg["env"]["pushRewardScale"]
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def erase_old_data(self):
        """
        Deletes all folders in the specified location.
    
        Args:
        path (str): The path to the directory where folders should be deleted.
    
        Returns:
        None
        """
        # Check if the specified path exists
        if not os.path.exists(GATHERED_DATA_ROOT):
            print(f"The specified path {GATHERED_DATA_ROOT} does not exist.")
            return
        
        # Iterate through the items in the specified path
        for item in os.listdir(GATHERED_DATA_ROOT):
            item_path = os.path.join(GATHERED_DATA_ROOT, item)
            # Check if the item is a directory
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                except Exception as e:
                    print(f"Failed to delete {item_path}. Reason: {e}")
    
        
        try:
            folders_to_create = ["dof_forces_data",
                                 "dof_state_data",
                                 "cube_position",
                                 "cam_data_rgb",
                                 "cam_data_depth",
                                 "cam_data_segmented",
                                 "cam_data_flow"]
            for folder_to_create in folders_to_create:
                new_folder = os.path.join(GATHERED_DATA_ROOT, folder_to_create)
                os.mkdir(new_folder)

            print(f"Folder created at: {new_folder}")
        except Exception as e:
            print(f"Failed to create folder at {new_folder}. Reason: {e}")

        #time.sleep(10)

    def save_rendered_imgs(self):
        for env_num in range(self.num_envs):
            for i, cam_handle in enumerate(self.camera_sensor_handle_lists[env_num]):
                rgb_filename = GATHERED_DATA_ROOT +os.sep+ f"cam_data_rgb{os.sep}rgb_env%d_cam%d_frame%d.png" % (env_num, i, self.frame_count)
                depth_filename = GATHERED_DATA_ROOT +os.sep+ f"cam_data_depth{os.sep}depth_env%d_cam%d_frame%d.png" % (env_num, i, self.frame_count)
                segmented_filename = GATHERED_DATA_ROOT +os.sep+ f"cam_data_segmented{os.sep}segmented_env%d_cam%d_frame%d.png" % (env_num, i, self.frame_count)
                flow_filename = GATHERED_DATA_ROOT +os.sep+ f"cam_data_flow{os.sep}flow_env%d_cam%d_frame%d.png" % (env_num, i, self.frame_count)
                
                #saving by this function takses 40 sec (16 envs, 5 frames)
                #gym.write_camera_image_to_file(sim, envs[env_num], cam_handle, gymapi.IMAGE_COLOR, rgb_filename)
                #saving by this method takses 8 sec (16 envs, 5 frames)
                camera_image = self.gym.get_camera_image(self.sim, self.envs[env_num], cam_handle, gymapi.IMAGE_COLOR)

                # Convert the image data to a NumPy array
                img_array = np.frombuffer(camera_image, dtype=np.uint8)
                img_array = img_array.reshape((CAM_SENSOR_HEIGHT, CAM_SENSOR_WIDTH, 4))  # Ensure the dimensions match the image format

                # Convert from RGBA to RGB
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

                    # Save the image using OpenCV

                #by using multithreading it takes 3.6 sec
                save_rgb_thread = threading.Thread(target=save_rgb_img, args=(rgb_filename, img_rgb))
                save_rgb_thread.start()
                
                depth_image = self.gym.get_camera_image(self.sim, self.envs[env_num], cam_handle, gymapi.IMAGE_DEPTH)
                save_rgb_thread = threading.Thread(target=save_depth_img, args=(depth_filename, depth_image))
                save_rgb_thread.start()

                segmented_image = self.gym.get_camera_image(self.sim, self.envs[env_num], cam_handle, gymapi.IMAGE_SEGMENTATION)
                save_segmented_thread = threading.Thread(target=save_segmented_img, args=(segmented_filename, segmented_image))
                save_segmented_thread.start()

                #flow_image = self.gym.get_camera_image(self.sim, self.envs[env_num], cam_handle, gymapi.IMAGE_OPTICAL_FLOW)
                #save_flow__thread = threading.Thread(target=save_flow_img, args=(flow_filename, flow_image))
                #save_flow__thread.start()

                #save_rgb_img(rgb_filename, img_rgb)
                #cv2.imwrite(rgb_filename, img_rgb)

    def save_dof_states_and_forces(self):
        for i, franka_actor in enumerate(self.frankas):
            forces = self.gym.get_actor_dof_forces(self.envs[i], franka_actor)
            forces = forces.reshape(1, -1)
            #print(f"forces = {forces}")
            #print(f"forces.shape = {forces.shape}")
            
            file_path = GATHERED_DATA_ROOT + os.sep + f"dof_forces_data{os.sep}dof_forces_data_env%d.csv" % (i)
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, forces, delimiter=',')
            dof_states = self.gym.get_actor_dof_states(self.envs[i], franka_actor, gymapi.STATE_ALL)
            #print(f"dof_states = {dof_states}")
            #print(f"dof_states['pos'] = {dof_states['pos']}")
            #print(f"dof_states['vel'] = {dof_states['vel']}")
            
            stacked_array = np.column_stack((dof_states['pos'].reshape(1, -1), dof_states['vel'].reshape(1, -1)))
            file_path = GATHERED_DATA_ROOT + os.sep + f"dof_state_data{os.sep}dof_state_data_env%d.csv" % (i)
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, stacked_array, delimiter=',')

    def save_cubes_position(self):
        
        for env_num, boxes in enumerate(self.boxes_to_track):
            for box_num, box_actor in enumerate(boxes):
                body_states = self.gym.get_actor_rigid_body_states(self.envs[env_num], box_actor, gymapi.STATE_POS)
                box_idx = self.gym.find_actor_rigid_body_index(self.envs[env_num], box_actor, "box", gymapi.DOMAIN_ACTOR)
        
                box_pos = body_states['pose']['p'][box_idx]
                box_rot = body_states['pose']['r'][box_idx]

                box_pos_nparray = np.array([box_pos['x'], box_pos['y'], box_pos['z']], dtype=np.float32).reshape(1, -1)
                box_rot_nparray = np.array([box_rot['x'], box_rot['y'], box_rot['z'], box_rot['w']], dtype=np.float32).reshape(1, -1)
                    
                stacked_array = np.column_stack((box_pos_nparray, box_rot_nparray))

                file_path = GATHERED_DATA_ROOT + os.sep + f"cube_position{os.sep}cube%d_env%d.csv" % (box_num, env_num)
                with open(file_path, 'a') as file:
                    # Write the array to the file
                    np.savetxt(file, stacked_array, delimiter=',')
        
        

    def update_camera_pos(self, env_ptr, franka_actor, cam_sensor_handle, pos_offset, axis_to_rotate, angle):
        body_states = self.gym.get_actor_rigid_body_states(env_ptr, franka_actor, gymapi.STATE_POS)
        panda_hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_actor, "panda_hand", gymapi.DOMAIN_ACTOR)
        
        look_from_transform = gymapi.Transform()
        # Convert the tuple to a Vec3 object before adding
        panda_hand_pos = gymapi.Vec3(body_states['pose']['p'][panda_hand_idx][0],
                                     body_states['pose']['p'][panda_hand_idx][1],
                                     body_states['pose']['p'][panda_hand_idx][2])
        
        panda_hand_rot = gymapi.Quat(body_states['pose']['r'][panda_hand_idx][0],
                                     body_states['pose']['r'][panda_hand_idx][1],
                                     body_states['pose']['r'][panda_hand_idx][2],
                                     body_states['pose']['r'][panda_hand_idx][3])
        
        
        rotated_offset = panda_hand_rot.rotate(pos_offset)
        look_from_transform.p = panda_hand_pos + rotated_offset
        angle_offset = gymapi.Quat.from_axis_angle(axis_to_rotate, angle)
        
        look_from_transform.r = panda_hand_rot * angle_offset
        #self.gym.set_camera_location(cam_sensor_handle, env_ptr, gymapi.Vec3(-2.0,-2.0, 2.0), gymapi.Vec3(2.0,2.0, 0.0))
        self.gym.set_camera_transform(cam_sensor_handle, env_ptr, look_from_transform)


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create floor asset
        floor_pos = [0.0, 0.0, 0.1]
        floor_thickness = 0.05
        floor_opts = gymapi.AssetOptions()
        floor_opts.fix_base_link = True
        floor_asset = self.gym.create_box(self.sim, *[10, 10, floor_thickness], floor_opts)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.050
        self.cubeB_size = 0.070

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        #############################
        if SAVE_SENSORS:
            cam_sensor_props = gymapi.CameraProperties()
            #cam_props.enable_tensors = True
            cam_sensor_props.width = CAM_SENSOR_WIDTH
            cam_sensor_props.height = CAM_SENSOR_HEIGHT
            self.camera_sensor_handle_lists = []
            self.attached_cam_sensors = []
            self.boxes_to_track = []
            self.frame_count = 0
            self.erase_old_data()

        ##############################

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for floor
        floor_start_pose = gymapi.Transform()
        floor_start_pose.p = gymapi.Vec3(*floor_pos)
        floor_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_actor)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            franka_body_count = self.gym.get_actor_rigid_body_count(env_ptr, franka_actor)

            for i in range(franka_body_count):
                self.gym.set_rigid_body_segmentation_id(env_ptr, franka_actor, i, 50)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            self.gym.set_rigid_body_segmentation_id(env_ptr, table_actor, 0, 100)
            self.gym.set_rigid_body_segmentation_id(env_ptr, table_stand_actor, 0, 150)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeA_id, 0, 200)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeB_id, 0, 250)
            self.boxes_to_track.append([self._cubeA_id, self._cubeB_id])

            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
                
            if SAVE_SENSORS:
                cam_sensor_handle0 = self.gym.create_camera_sensor(env_ptr, cam_sensor_props)
                self.gym.set_camera_location(cam_sensor_handle0, env_ptr, gymapi.Vec3(0.6, 0.0, 1.2), gymapi.Vec3(0.0, 0.0, 1.0))

                cam_sensor_handle1 = self.gym.create_camera_sensor(env_ptr, cam_sensor_props)
                self.gym.set_camera_location(cam_sensor_handle1, env_ptr, gymapi.Vec3(0.0, 0.6, 1.2), gymapi.Vec3(0.0, 0.0, 1.0))

                cam_sensor_handle2 = self.gym.create_camera_sensor(env_ptr, cam_sensor_props)
                self.gym.set_camera_location(cam_sensor_handle2, env_ptr, gymapi.Vec3(0.0, -0.6, 1.2), gymapi.Vec3(0.0, 0.0, 1.0))

                cam_sensor_handle_attached0 = self.gym.create_camera_sensor(env_ptr, cam_sensor_props)
                pos_offset = gymapi.Vec3(-0.029271, -0.000388, 0.05234)
                axis_to_rotate = gymapi.Vec3(0.0, 1.0, 0.0)
                angle = np.radians(-60.0)
                self.update_camera_pos(env_ptr, franka_actor, cam_sensor_handle_attached0, pos_offset, axis_to_rotate, angle)
                #self.gym.attach_camera_to_body(cam_sensor_handle1, env_ptr, body_handle_to_attach_cam, local_transform, gymapi.FOLLOW_TRANSFORM)
                self.attached_cam_sensors.append([env_ptr, franka_actor, cam_sensor_handle_attached0, pos_offset, axis_to_rotate, angle])
                #self.camera_sensor_handle_lists.append([cam_sensor_handle0, cam_sensor_handle1])

                cam_sensor_handle_attached1 = self.gym.create_camera_sensor(env_ptr, cam_sensor_props)
                pos_offset = gymapi.Vec3(0.02676, -0.000388, 0.05234)
                axis_to_rotate = gymapi.Vec3(0.0, 1.0, 0.0)
                angle = np.radians(180+60.0)
                self.update_camera_pos(env_ptr, franka_actor, cam_sensor_handle_attached1, pos_offset, axis_to_rotate, angle)
                self.attached_cam_sensors.append([env_ptr, franka_actor, cam_sensor_handle_attached1, pos_offset, axis_to_rotate, angle])
                self.camera_sensor_handle_lists.append([cam_sensor_handle0,
                                                        cam_sensor_handle1,
                                                        cam_sensor_handle2,
                                                        cam_sensor_handle_attached0,
                                                        cam_sensor_handle_attached1])
            
            
            #floor_actor = self.gym.create_actor(env_ptr, floor_asset, floor_start_pose, "floor", i, 8, 0)
            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
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
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set z value, which is fixed height
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        # u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
        #                               self.franka_dof_lower_limits[-2].item())
        # u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
        #                               self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        if SAVE_SENSORS:
            self.save_cubes_position()
            #self.gym.refresh_force_sensor_tensor(self.sim)
            self.save_dof_states_and_forces()

            for env_ptr, franka_actor, cam_sensor_handle, pos_offset, axis_to_rotate, angle in self.attached_cam_sensors:
                self.update_camera_pos(env_ptr, franka_actor, cam_sensor_handle, pos_offset, axis_to_rotate, angle)

            self.gym.step_graphics(self.sim)
            self.gym.sync_frame_time(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            print(f"rendering frame num: {self.frame_count}")

            self.save_rendered_imgs()
            
            if self.frame_count > FRAMES_MAX:
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                print(f"RENDERING {FRAMES_MAX} FREMES TOOK: {elapsed_time} seconds")
                exit(0)
            
            self.frame_count += 1

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
        reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    # Define threshold for EE being close to cube A
    ee_close_threshold = 0.05  # You can adjust this threshold as needed

    # Distance from cubeA to cubeB
    d_ab = torch.norm(states["cubeA_to_cubeB_pos"], dim=-1)
    move_reward = 1 - torch.tanh(10.0 * d_ab)

    # Distance from EE to cubeA
    d_eef_to_cubeA = torch.norm(states["cubeA_pos"] - states["eef_pos"], dim=-1)

    # Check if cubeA is near cubeB horizontally
    cubeA_near_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)

    # Check if cubeA is on the ground using cubeB's height as a reference
    cubeA_on_ground = torch.abs(states["cubeA_pos"][:, 2] - states["cubeB_pos"][:, 2]) < 0.01

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
    ee_close_to_cubeA = ee_target_reward > 0.75
    if torch.any(progress_buf >= max_episode_length - 1):
        print(f"EE close to A: {torch.sum(ee_close_to_cubeA, dim=0)}")

    push_reward = cubeA_near_cubeB & cubeA_on_ground

    # Compose rewards
    rewards = torch.where(
        push_reward,
        reward_settings["r_stack_scale"] * push_reward,
        reward_settings["r_align_scale"] * move_reward * ee_close_to_cubeA.to(torch.float) +
        reward_settings["r_align_scale"] * ee_target_reward
    )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (push_reward > 0), torch.ones_like(reset_buf),
                            reset_buf)

    return rewards, reset_buf

