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




@configclass
class FrankaPushEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.5  # 500 timesteps
    decimation = 8
    num_actions = 9
    num_observations = 43
    num_states = 0

    #max_episode_length = 250
    action_scale = 1.0
    start_position_noise = 0.25
    start_rotation_noise = 0.785
    franka_position_noise =0.0
    franka_rotation_noise = 0.0
    franka_dof_noise = 0.25
    aggregate_mode = 3


    camera_modes = ["rgb", "distance_to_camera", "semantic_segmentation", "motion_vectors"]
    semantic_filter = "class : cubeA ; class : cubeB; class : Robot; class : Table"
    cubeA_size = 0.050
    cubeB_size = 0.070
    cam_sensor_width = int(1920/4)
    cam_sensor_height = int(1080/4) 
    
    
    r_dist_scale = 0.1
    r_lift_scale = 1.5
    r_align_scale = 0.1
    r_stack_scale = 4.0
    r_push_scale = 8.0
    r_finished_scale = 16.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            semantic_tags = [("class", "Robot")],
            activate_contact_sensors= True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
                "panda_finger_joint.*": 0.035,
            },
            pos=(1.0, 0.0, 1.05),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=150.0,
                damping=24.5,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=150.0,
                damping=24.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )


    cubeA = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cubeA",
        
        spawn= sim_utils.CuboidCfg(
            #usd_path= None,
                size = (cubeA_size, cubeA_size, cubeA_size),
                semantic_tags = [("class", "cubeA")],
                #radius=0.15,
                #height=0.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.25, 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cubeB = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cubeB",
        
        spawn= sim_utils.CuboidCfg(
            #usd_path= None,
                size = (cubeB_size, cubeB_size, cubeB_size),
                semantic_tags = [("class", "cubeB")],
                #radius=0.15,
                #height=0.5,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.25, 2.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


    tiled_camera0: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Camera0",
    offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
    data_types= camera_modes,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
    width=cam_sensor_width,
    height=cam_sensor_height,
    semantic_filter= semantic_filter,
    
    )

    tiled_camera1: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Camera1",
    offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
    data_types= camera_modes,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
    width=cam_sensor_width,
    height=cam_sensor_height,
    semantic_filter= semantic_filter,
    
    )

    tiled_camera2: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Camera2",
    offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
    data_types= camera_modes,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    ),
    width=cam_sensor_width,
    height=cam_sensor_height,
    semantic_filter= semantic_filter,
    
    )

    tiled_camera3: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/panda_hand/Camera3",
    offset=CameraCfg.OffsetCfg(pos=(0.029271, -0.000388, 0.03234), rot=(0.0, 0.707, 0.0, 0.707), convention="world"),
    data_types= camera_modes, 
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.005, 20.0)
    ),
    width=cam_sensor_width,
    height=cam_sensor_height,
    semantic_filter= semantic_filter,
    
    )

    tiled_camera4: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/panda_hand/Camera4",
    offset=CameraCfg.OffsetCfg(pos=(-0.029271, -0.000388, 0.03234), rot=(0.707, 0.0, -0.707, 0.0), convention="world"),
    data_types= camera_modes,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.005, 20.0)
    ),
    width=cam_sensor_width,
    height=cam_sensor_height,
    semantic_filter= semantic_filter,
    
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 2.0
    rot_reward_scale = 0.5
    around_handle_reward_scale = 0.0
    open_reward_scale = 7.5
    action_penalty_scale = 0.01
    finger_dist_reward_scale = 0.0
    finger_close_reward_scale = 10.0
