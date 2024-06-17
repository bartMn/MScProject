from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import random
import numpy as np
import torch
import time
import cv2


NUM_ENVS = 16    
ENVS_PER_ROW = 4
RENDER_FREQ = 1
FRAMES_MAX = RENDER_FREQ * 50
GPU_PIPELINE = False
CAM_SENSOR_WIDTH = 1920
CAM_SENSOR_HEIGHT = 1080

gym = gymapi.acquire_gym()
args = gymutil.parse_arguments(description="This is a playground",
                               custom_parameters=[
                                   {"name": "--headless", "action": "store_true", "help": "no viewer window"},
                                   {"name": "--save", "action": "store_true", "help": "save rendered images"}
                                   ])


"""
sim_params = gymapi.SimParams()
sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
"""

if args.save:
    from PIL import Image as im
    import threading

    def save_img(rgb_filename, img_rgb):
        cv2.imwrite(rgb_filename, img_rgb)

    def save_img2(rgb_filename, camera_image):
        # Convert the image data to a NumPy array
        img_array = np.frombuffer(camera_image, dtype=np.uint8)
        img_array = img_array.reshape((CAM_SENSOR_HEIGHT, CAM_SENSOR_WIDTH, 4))  # Ensure the dimensions match the image form   
        # Convert from RGBA to RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        # Save the image using Open 
        cv2.imwrite(rgb_filename, img_rgb)
        #save_img(rgb_filename, img_rgb)
        #cv2.imwrite(rgb_filename, img_rgb)

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.use_gpu_pipeline = GPU_PIPELINE
sim_params.physx.use_gpu = True

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.num_threads = 15

"""
# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
"""
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# create sim with these parameters
print("PRINTING ARGS")
print(f"args.compute_device_id = {args.compute_device_id}")
print(f"args.graphics_device_id = {args.graphics_device_id}")
print(f"args.physics_engine = {args.physics_engine}")
print("END ARGS")
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
print("sim created sucessfully :)")

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)
print("GROUND ADDED")

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
asset_options.disable_gravity = False

franka_asset_root = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
franka_asset = gym.load_asset(sim, franka_asset_root, franka_asset_file, asset_options)
print("ASSET ADDED")

"""
spacing = 2.0
lower = gymapi.Vec3(-spacing, -spacing, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 8)
"""

cam_props = gymapi.CameraProperties()
cam_props.enable_tensors = True
cam_props.width = int(1920)
cam_props.height = int(1080)
#viewer = gym.create_viewer(sim, cam_props)

if not args.headless:
    # create viewer using the default camera properties
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')


cam_sensor_props = gymapi.CameraProperties()

cam_props.enable_tensors = True
cam_sensor_props.width = CAM_SENSOR_WIDTH
cam_sensor_props.height = CAM_SENSOR_HEIGHT



# set up the env grid
num_envs = NUM_ENVS
envs_per_row = ENVS_PER_ROW
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []
camera_sensor_handle_lists = []

width = 0.3
height = 0.4
radius = 0.3
depth = 0.3
length = 0.3

asset_options1 = gymapi.AssetOptions()
asset_options1.default_dof_drive_mode = gymapi.DOF_MODE_POS
box_asset = gym.create_box(sim, width, height, depth, asset_options1)
sphere_asset = gym.create_sphere(sim, radius, asset_options1)
capsule_asset = gym.create_capsule(sim, radius, width, asset_options1)


# create and populate the environments
for i in range(num_envs):
    camera_sensor_handles = []
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    #height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-1.5, 0.0, 0.0)
    franka_actor_handle = gym.create_actor(env, franka_asset, pose, f"MyActor", i, 1)
    actor_handles.append(franka_actor_handle)

    num_bodies = gym.get_actor_rigid_body_count(env, franka_actor_handle)
    body_handle_to_attach_cam = gym.get_actor_rigid_body_handle(env, franka_actor_handle, num_bodies-6)

    # Iterate through each body to get their handles and names
    """
    body_handles = []
    body_names = []
    for i in range(num_bodies):
        body_handle = gym.get_actor_rigid_body_handle(env, franka_actor_handle, i)
        #body_name = gym.get_actor_rigid_body_name(env, franka_actor_handle, i)
        body_handles.append(body_handle)
        #body_names.append(body_name)
        print(f"Body {i}: Handle = {body_handle}, Name = ")#body_name}")

    exit(0)
    """
    
    pose.p = gymapi.Vec3(1.0, -1.0, 0.0)
    actor_handle = gym.create_actor(env, box_asset, pose, f"box_asset", i, 2)
    actor_handles.append(actor_handle)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(random.random(), random.random(), random.random()))

    pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
    actor_handle = gym.create_actor(env, sphere_asset, pose, f"sphere_asset", i, 3)
    actor_handles.append(actor_handle)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(random.random(), random.random(), random.random()))
    
    pose.p = gymapi.Vec3(1.0, 1.0, 0.0)
    actor_handle = gym.create_actor(env, capsule_asset, pose, f"capsule_asset", i, 4)
    actor_handles.append(actor_handle)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(random.random(), random.random(), random.random()))

    cam_sensor_handle0 = gym.create_camera_sensor(env, cam_sensor_props)
    gym.set_camera_location(cam_sensor_handle0, env, gymapi.Vec3(-2.0,-2.0, 1.0), gymapi.Vec3(2.0,2.0, 0.0))
    
    cam_sensor_handle1 = gym.create_camera_sensor(env, cam_sensor_props)
    gym.set_camera_location(cam_sensor_handle1, env, gymapi.Vec3(2.0,2.0, 1.0), gymapi.Vec3(-2.0, -2.0, 0.0))

    local_transform = gymapi.Transform()
    cam_sensor_handle2 = gym.create_camera_sensor(env, cam_sensor_props)
    local_transform.p = gymapi.Vec3(1.0, 0.0, 0.0)
    local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1.0, 0.0, 0.0), np.radians(0.0))
    gym.attach_camera_to_body(cam_sensor_handle2, env, body_handle_to_attach_cam, local_transform, gymapi.FOLLOW_TRANSFORM)

    camera_sensor_handle_lists.append([cam_sensor_handle0, cam_sensor_handle1, cam_sensor_handle2])

gym.prepare_sim(sim)

frame_count = 0
# Record the start time
start_time = time.time()


while not args.save or frame_count < FRAMES_MAX:
    
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
 
    # update the viewer
    gym.step_graphics(sim)
    #gym.draw_viewer(viewer, sim, True)
 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.

    

    print(f"CURRENT FRAME: {frame_count}")
    if args.save and not np.mod(frame_count, RENDER_FREQ):
        gym.render_all_camera_sensors(sim)
        #color_image = gym.get_camera_image(sim, envs[0], camera_sensor_handles[0], gymapi.IMAGE_COLOR)
        for env_num in range(len(envs)):
            for i, cam_handle in enumerate(camera_sensor_handle_lists[env_num]):
                rgb_filename = "/home/bart/project/IsaacGym_Preview_4_Package/testPrj/graphics_images/rgb_env%d_cam%d_frame%d.png" % (env_num, i, frame_count)
                
                #saving by this function takses 40 sec (16 envs, 5 frames)
                #gym.write_camera_image_to_file(sim, envs[env_num], cam_handle, gymapi.IMAGE_COLOR, rgb_filename)
                

                #saving by this method takses 8 sec (16 envs, 5 frames)
                camera_image = gym.get_camera_image(sim, envs[env_num], cam_handle, gymapi.IMAGE_COLOR)
                
                """
                save_thread = threading.Thread(target=save_img2, args=(rgb_filename, camera_image))
                save_thread.start()

                """
                # Convert the image data to a NumPy array
                img_array = np.frombuffer(camera_image, dtype=np.uint8)
                img_array = img_array.reshape((CAM_SENSOR_HEIGHT, CAM_SENSOR_WIDTH, 4))  # Ensure the dimensions match the image format

                # Convert from RGBA to RGB
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

                # Save the image using OpenCV

                #by using multithreading it takes 3.6 sec
                save_thread = threading.Thread(target=save_img, args=(rgb_filename, img_rgb))
                save_thread.start()
                #save_img(rgb_filename, img_rgb)
                #cv2.imwrite(rgb_filename, img_rgb)
                
            #rgb_filename = "/home/bart/project/IsaacGym_Preview_4_Package/testPrj/graphics_images/rgb_env%d_cam%d_frame%d.png" % (env_num, 1, frame_count)
            #gym.write_camera_image_to_file(sim, envs[env_num], camera_sensor_handle_lists[env_num][1], gymapi.IMAGE_COLOR, rgb_filename)
#
            #rgb_filename = "/home/bart/project/IsaacGym_Preview_4_Package/testPrj/graphics_images/rgb_env%d_cam%d_frame%d.png" % (env_num, 2, frame_count)
            #gym.write_camera_image_to_file(sim, envs[env_num], camera_sensor_handle_lists[env_num][2], gymapi.IMAGE_COLOR, rgb_filename)


    if not args.headless:
        # render the viewer
        gym.draw_viewer(viewer, sim, True)

        gym.sync_frame_time(sim)
        # Check for exit condition - user closed the viewer window
        if gym.query_viewer_has_closed(viewer):
            break

    frame_count += 1


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

if args.save:
    print(f"RENDERING {FRAMES_MAX / RENDER_FREQ} FREMES TOOK: {elapsed_time} seconds")

if not args.headless:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)