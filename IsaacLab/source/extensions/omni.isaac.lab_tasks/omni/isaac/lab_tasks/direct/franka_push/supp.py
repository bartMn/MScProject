from PIL import Image as im
import threading
import cv2
import random
import time
import shutil
import numpy as np
import os

class SupportClass():

    def __init__(self,
                 gathered_data_root,
                 erase_exiting_data,
                 cam_sensor_width,
                 cam_sensor_height,
                 cam_data_type,
                 num_envs,
                 num_of_cameras):
        
        self.gathered_data_root = gathered_data_root
        self.erase_exiting_data = erase_exiting_data
        self.cam_sensor_width = cam_sensor_width
        self.cam_sensor_height = cam_sensor_height
        self.cam_data_type = cam_data_type
        self.num_envs = num_envs
        self.next_env_to_save = 0
        self.create_folders_for_data(num_of_cameras)
        self.data_frame_count = 0
        

    def save_rgb_img(self, rgb_filename, img_rgb):
        cv2.imwrite(rgb_filename, img_rgb)
    
    def save_depth_img(self, depth_filename, depth_image):
        
        max_distance = 1.5
        depth_image[depth_image == np.inf] = max_distance
    
        # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image > max_distance] = max_distance
    
        # flip the direction so near-objects are light and far objects are dark
        normalized_depth = (255.0/max_distance)*(depth_image)#/np.min(depth_image + 1e-4))
        #normalized_depth= depth_image
        # Convert to a pillow image and write it to disk
        normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
        normalized_depth_image.save(depth_filename)
        #normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (i, j, frame_count))
    
    def save_segmented_img(self, segmented_filename, segmented_image):
        cv2.imwrite(segmented_filename, segmented_image)
        return
        normalized_depth_image = im.fromarray(segmented_image.astype(np.uint8), mode="L")
        normalized_depth_image.save(segmented_filename)
    
    def save_flow_img(self, flow_filename, optical_flow_image):
        
        optical_flow_image = optical_flow_image.astype(np.float32)
    
        # Scale brightness by multiplying each pixel by the factor
        optical_flow_image = optical_flow_image * 6

        # Clip the values to be in the valid range [0, 255] and convert back to uint8
        optical_flow_image = np.clip(optical_flow_image, 0, 255).astype(np.uint8)
        #optical_flow_image = cv2.convertScaleAbs(optical_flow_image, alpha=1, beta=30)
    
        cv2.imwrite(flow_filename, optical_flow_image)
    
    def create_folders_for_data(self, num_of_cameras):
        """
        Deletes all folders in the specified location.
    
        Args:
        path (str): The path to the directory where folders should be deleted.
    
        Returns:
        None
        """
        # Check if the specified path exists
        if not os.path.exists(self.gathered_data_root):
            print(f"The specified path {self.gathered_data_root} does not exist.")
            return
        
        self.next_env_to_save = None
        self.data_dirs = {"cam_rgb"         : f"{self.gathered_data_root}{os.sep}cameras{os.sep}rgb",
                          "cam_depth"       : f"{self.gathered_data_root}{os.sep}cameras{os.sep}depth",
                          "cam_seg"         : f"{self.gathered_data_root}{os.sep}cameras{os.sep}segmented",
                          "cam_flow"        : f"{self.gathered_data_root}{os.sep}cameras{os.sep}flow",
                          "franka_forces"   : f"{self.gathered_data_root}{os.sep}franka_robot{os.sep}dof_forces",
                          "franka_state"    : f"{self.gathered_data_root}{os.sep}franka_robot{os.sep}dof_state",
                          "franka_actions"  : f"{self.gathered_data_root}{os.sep}franka_robot{os.sep}actions",
                          #"boxes_pos"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}position",
                          "boxes_pos0"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}position{os.sep}box0",
                          "boxes_pos1"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}position{os.sep}box1",
                          #"boxes_vel"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}velocity",
                          "boxes_vel0"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}velocity{os.sep}box0",
                          "boxes_vel1"       : f"{self.gathered_data_root}{os.sep}boxes{os.sep}velocity{os.sep}box1"
                        }
        
        if self.erase_exiting_data:
            usr_input = "y"#input("this will erase all existind data, continue? [y]: ")
            if usr_input != "y":
                print("EXITING...")
                exit(0)
            self.next_env_to_save = 0
            # Iterate through the items in the specified path
            for item in os.listdir(self.gathered_data_root):
                item_path = os.path.join(self.gathered_data_root, item)
                # Check if the item is a directory
                if os.path.isdir(item_path):
                    try:
                        shutil.rmtree(item_path)
                        print(f"Deleted folder: {item_path}")
                    except Exception as e:
                        print(f"Failed to delete {item_path}. Reason: {e}")
            data_actors_dirs = [f"{self.gathered_data_root}{os.sep}cameras",
                                f"{self.gathered_data_root}{os.sep}franka_robot",
                                f"{self.gathered_data_root}{os.sep}boxes",
                                f"{self.gathered_data_root}{os.sep}boxes{os.sep}position",
                                f"{self.gathered_data_root}{os.sep}boxes{os.sep}velocity"
                               ]
            
            folders_to_create = data_actors_dirs + list(self.data_dirs.values())
            try:
                for folder_to_create in folders_to_create:
                    os.mkdir(folder_to_create)
                    print(f"Folder created at: {folder_to_create}")
                    if f"cameras{os.sep}" in folder_to_create:
                        for i in range(num_of_cameras):
                            new_folder = f"{folder_to_create}{os.sep}cam{i}"
                            os.mkdir(new_folder)
            except Exception as e:
                print(f"Failed to create folder at {new_folder}. Reason: {e}")
    
        else:
            files = [int(f.replace("env", "")) for f in os.listdir(f"{self.gathered_data_root}{os.sep}cameras{os.sep}rgb{os.sep}cam0")]
            self.next_env_to_save = max(files) + 1
        try:
            for final_folder in self.data_dirs.values():
                for env_num in range(self.next_env_to_save, self.next_env_to_save + self.num_envs):
                    if "cameras" in final_folder:
                        for i in range(num_of_cameras):
                            new_folder = os.path.join(f"{final_folder}{os.sep}cam{i}", f"env{env_num}")
                            os.mkdir(new_folder)
                    else:
                        new_folder = os.path.join(final_folder, f"env{env_num}")
                        os.mkdir(new_folder)
                    print(f"Folder created at: {new_folder}")
        except Exception as e:
            print(f"Failed to create folder at {new_folder}. Reason: {e}")
        #time.sleep(10)
    
    
    def save_rendered_imgs(self, cameras_data_list, current_frame):
        
        self.data_frame_count = current_frame

        for cam_num, cam_handle in enumerate(cameras_data_list):
            for env_num in range(self.num_envs):
                   
                rgb_filename =       f"{self.data_dirs['cam_rgb']}{os.sep}cam%d{os.sep}env%d{os.sep}frame%d.png" %   (cam_num, env_num + self.next_env_to_save, self.data_frame_count)
                depth_filename =     f"{self.data_dirs['cam_depth']}{os.sep}cam%d{os.sep}env%d{os.sep}frame%d.png" % (cam_num, env_num + self.next_env_to_save, self.data_frame_count)
                segmented_filename = f"{self.data_dirs['cam_seg']}{os.sep}cam%d{os.sep}env%d{os.sep}frame%d.png" %   (cam_num, env_num + self.next_env_to_save, self.data_frame_count)
                flow_filename =      f"{self.data_dirs['cam_flow']}{os.sep}cam%d{os.sep}env%d{os.sep}frame%d.png" %  (cam_num, env_num + self.next_env_to_save, self.data_frame_count)
                
                #saving by this function takses 40 sec (16 envs, 5 frames)
                #gym.write_camera_image_to_file(sim, envs[env_num], cam_handle, gymapi.IMAGE_COLOR, rgb_filename)
                #saving by this method takses 8 sec (16 envs, 5 frames)
                if "rgb" in self.cam_data_type:
                    camera_image = cam_handle["rgb"][env_num]
                    
                    if camera_image.is_cuda:
                        camera_image = camera_image.cpu()
                    numpy_array = camera_image.numpy()
                    # Convert from RGBA to RGB
                    #print(f"numpy_array,shape = {numpy_array.shape}")
                    img_rgb = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2RGB)
                    #print(f"img_rgb,shape = {img_rgb.shape}")
                    #by using multithreading it takes 3.6 sec
                    save_rgb_thread = threading.Thread(target=self.save_rgb_img, args=(rgb_filename, img_rgb))
                    save_rgb_thread.start()
                if "distance_to_camera" in self.cam_data_type:
                    depth_image = cam_handle["distance_to_camera"][env_num]

                    if depth_image.is_cuda:
                        depth_image = depth_image.cpu()
                    depth_image = depth_image.numpy()
                    save_rgb_thread = threading.Thread(target=self.save_depth_img, args=(depth_filename, depth_image))
                    save_rgb_thread.start()

                if "semantic_segmentation" in self.cam_data_type:
                    segmented_image = cam_handle["semantic_segmentation"][env_num]

                    #print(f"segmented_image.shape = {segmented_image.shape}")
                    #exit()

                    if segmented_image.is_cuda:
                        segmented_image = segmented_image.cpu()
                    segmented_image = segmented_image.numpy()
                    #print(f"segmented_image,shape = {segmented_image.shape}")
                    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGBA2RGB)
                    save_segmented_thread = threading.Thread(target=self.save_segmented_img, args=(segmented_filename, segmented_image))
                    save_segmented_thread.start()

                if "motion_vectors" in self.cam_data_type:
                    motion_image = cam_handle["motion_vectors"][env_num]

                    if motion_image.is_cuda:
                        motion_image = motion_image.cpu()
                    motion_image = motion_image.numpy()
                    #print(f"segmented_image,shape = {segmented_image.shape}")
                    motion_image = cv2.cvtColor(motion_image, cv2.COLOR_RGBA2RGB)
                    save_segmented_thread = threading.Thread(target=self.save_flow_img, args=(flow_filename, motion_image))
                    save_segmented_thread.start()
                #print(100*"+")
                #exit()
        print(f"rendered frame {current_frame}")
        

    def save_dof_states_and_forces(self, joint_torques, joint_pos, joint_vel):
        for i, franka_actor in enumerate(joint_torques):
            forces = joint_torques[i]
            if forces.is_cuda:
                forces = forces.cpu()
            forces = forces.numpy()
            forces = forces.reshape(1, -1)
            file_path = f"{self.data_dirs['franka_forces']}{os.sep}env%d{os.sep}data.csv" % (i + self.next_env_to_save)
            with open(file_path, 'a') as file:
                np.savetxt(file, forces, delimiter=',')
            

            joint_pos_env = self.cuda_tensor_to_numpy(joint_pos[i])
            joint_vel_env = self.cuda_tensor_to_numpy(joint_vel[i])
            
            stacked_array = np.column_stack((joint_pos_env.reshape(1, -1), joint_vel_env.reshape(1, -1)))
            file_path = f"{self.data_dirs['franka_state']}{os.sep}env%d{os.sep}data.csv" % (i + self.next_env_to_save)
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, stacked_array, delimiter=',')

    def cuda_tensor_to_numpy(self, tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()

    def save_cubes_position(self,
                            box_pos_lin,
                            box_pos_quat,
                            box_vel_lin,
                            box_vel_rot,
                            box_name):
        
        for env_num, boxes in enumerate(box_pos_lin):

            box_pos = self.cuda_tensor_to_numpy(box_pos_lin[env_num])
            box_rot = self.cuda_tensor_to_numpy(box_pos_quat[env_num])
  
            stacked_array = np.column_stack((box_pos.reshape(1, -1), box_rot.reshape(1, -1)))
            file_path = f"{self.data_dirs[f'boxes_pos{box_name}']}{os.sep}env{env_num + self.next_env_to_save}{os.sep}data.csv"
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, stacked_array, delimiter=',')
                
            box_lin_vel = self.cuda_tensor_to_numpy(box_vel_lin[env_num])
            box_ang_vel = self.cuda_tensor_to_numpy(box_vel_rot[env_num])

            stacked_array = np.column_stack((box_lin_vel.reshape(1, -1), box_ang_vel.reshape(1, -1)))
            file_path = f"{self.data_dirs[f'boxes_vel{box_name}']}{os.sep}env{env_num + self.next_env_to_save}{os.sep}data.csv"
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, stacked_array, delimiter=',')

    def save_actions(self, actions):

        for env_num, franka_actions in enumerate(actions):

            actions_row = self.cuda_tensor_to_numpy(franka_actions).reshape(1, -1)
            
            file_path = f"{self.data_dirs['franka_actions']}{os.sep}env%d{os.sep}data.csv" % (env_num + self.next_env_to_save)
            with open(file_path, 'a') as file:
                # Write the array to the file
                np.savetxt(file, actions_row, delimiter=',')
