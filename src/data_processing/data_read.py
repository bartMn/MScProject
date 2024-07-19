import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


class singleSampleDataset(Dataset):
    def __init__(self, data_dict_dir, transform=None, transform_output_imgs = None, output_data_key= ""):

        self.transform = transform
        self.transform_output_imgs= transform_output_imgs
        self.output_data_key = output_data_key
    
        # Collect all env directories
        self.env_dirs = sorted(os.listdir(data_dict_dir["cam0_rgb"]))
        
        # Initialize lists to hold image paths and csv data
        self.data = dict()
        self.key_to_check_len = None
        for key in data_dict_dir:
            self.data[key] = []

        for env_num in self.env_dirs:
            for dict_key in data_dict_dir:
                self.key_to_check_len = dict_key
                if "cam" in dict_key:
                    cam_env_dir = os.path.join(data_dict_dir[dict_key], env_num)
                    # Get image files
                    cam = sorted(os.listdir(cam_env_dir), key = natural_sort_key)

                    for img in cam:
                        self.data[dict_key].append(os.path.join(cam_env_dir, img))

                else:
                    # Read csv file
                    csv_env_file = os.path.join(data_dict_dir[dict_key], env_num, 'data.csv')
                    df = pd.read_csv(csv_env_file, header=None)
                    self.data[dict_key].extend(df.values.tolist())  
                
        self.csv_min_max = {}
        for key, data in self.data.items():
            if "cam" in key:
                continue
            data = np.array(data)
            self.csv_min_max[key] = (data.min(axis=0), data.max(axis=0))



    def __len__(self):
        return len(self.data[self.key_to_check_len])



    def __getitem__(self, idx):
        
        mulitisensory_sample = dict()
        epsilon=1e-8
        for dict_key in self.data:
            
            if "cam" in dict_key:
                # Read images
                mulitisensory_sample[dict_key] = Image.open(self.data[dict_key][idx]).convert('RGB')
                #sensor2_image = Image.open(self.data["cam1_rgb"][idx]).convert('RGB')

                if self.output_data_key == dict_key:
                    if "seg" in dict_key:
                        mulitisensory_sample[dict_key]= rgb_to_class_index(mulitisensory_sample[dict_key])
                        mulitisensory_sample[dict_key] = class_index_to_one_hot(mulitisensory_sample[dict_key])
                       
                    if self.transform_output_imgs:
                        mulitisensory_sample[dict_key] = self.transform_output_imgs(mulitisensory_sample[dict_key])
                   

                elif self.transform:
                    mulitisensory_sample[dict_key] = self.transform(mulitisensory_sample[dict_key])
                    #sensor2_image = self.transform(sensor2_image)

            else:

                min_val, max_val = self.csv_min_max[dict_key]
                # Read CSV row
                min_val = np.array(min_val, dtype=np.float32)
                max_val = np.array(max_val, dtype=np.float32)
                denominator = max_val - min_val
                denominator = np.where(denominator == 0, epsilon, denominator)
                
                read_data = self.data[dict_key][idx]
                read_data = np.array(read_data, dtype=np.float32)
                read_data = (read_data - min_val) / denominator
                # Convert CSV row to tensor
                mulitisensory_sample[dict_key] = torch.tensor(read_data, dtype=torch.float)
        
        return mulitisensory_sample
  

    def remove_unused_keys(self, used_keys):

        all_keys = list()
        for key in self.data:
            all_keys.append(key)
        for key in all_keys:
            if key not in used_keys:
                self.data.pop(key)
        
        self.key_to_check_len = used_keys[0]

  
class sequentialSampleDataset(Dataset):
    def __init__(self, data_dict_dir, transform=None, transform_output_imgs = None, sequence_length=3, output_data_key = "boxes_pos0"):
        self.transform = transform
        self.transform_output_imgs = transform_output_imgs
        self.output_data_key = output_data_key
        self.sequence_length = sequence_length
        self.env_boundaries = list()
        # Collect all env directories
        self.env_dirs = sorted(os.listdir(data_dict_dir["cam0_rgb"]))
        
        # Initialize lists to hold image paths and csv data
        self.data = dict()
        for key in data_dict_dir:
            self.data[key] = []

        for env_num in self.env_dirs:

            start_idx = len(self.data["cam0_rgb"])

            for dict_key in data_dict_dir:
                if "cam" in dict_key:
                    cam_env_dir = os.path.join(data_dict_dir[dict_key], env_num)
                    # Get image files
                    cam = sorted(os.listdir(cam_env_dir), key=natural_sort_key)
                    for img in cam:
                        self.data[dict_key].append(os.path.join(cam_env_dir, img))
                else:
                    # Read csv file
                    csv_env_file = os.path.join(data_dict_dir[dict_key], env_num, 'data.csv')
                    df = pd.read_csv(csv_env_file, header=None)
                    self.data[dict_key].extend(df.values.tolist())
            
            end_idx = len(self.data["cam0_rgb"]) - 1
            self.env_boundaries.append((start_idx, end_idx))

        
        self.csv_min_max = {}
        for key, data in self.data.items():
            if "cam" in key:
                continue
            data = np.array(data)
            self.csv_min_max[key] = (data.min(axis=0), data.max(axis=0))




    def __len__(self):
        total_length = 0
        for start, end in self.env_boundaries:
            total_length += max(0, end - start + 1 - self.sequence_length + 1)
        return total_length



    def __getitem__(self, idx):
        
        for start, end in self.env_boundaries:
            env_length = end - start + 1 - self.sequence_length + 1
            if idx < env_length:
                idx = start + idx
                break
            idx -= env_length

        mulitisensory_sample = dict()
        for dict_key in self.data:
            if "cam" in dict_key:
                images = []
                for i in range(self.sequence_length):
                    img = Image.open(self.data[dict_key][idx + i]).convert('RGB')

                    if self.output_data_key == dict_key:
                        if "seg" in dict_key:
                            img = rgb_to_class_index(img)
                            img = class_index_to_one_hot(img)
                       
                        if self.transform_output_imgs:
                            img = self.transform_output_imgs(img)
                   
                    elif self.transform:
                        img = self.transform(img)
                    images.append(img)
                mulitisensory_sample[dict_key] = torch.stack(images)
            else:
                rows = []
                for i in range(self.sequence_length):
                    row = self.data[dict_key][idx + i]
                    rows.append(row)
                mulitisensory_sample[dict_key] = torch.tensor(rows, dtype=torch.float)
        

        mulitisensory_sample[self.output_data_key] = mulitisensory_sample[self.output_data_key][-1]

        return mulitisensory_sample


    def remove_unused_keys(self, used_keys):

        all_keys = list()
        for key in self.data:
            all_keys.append(key)
        for key in all_keys:
            if key not in used_keys:
                self.data.pop(key)
        
        self.key_to_check_len = used_keys[0]


def test_single_samples():

    NUM_OF_CAMERAS = 5
    camera_types = ["depth", "rgb", "segmented", "flow"]
    #data_root = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data"
    data_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(data_root, "..", "..", "isaacgym", "IsaacGymEnvs-main", "isaacgymenvs", "recorded_data")

    data_dict_dir = dict()
    data_dict_dir["franka_actions"] = f"{data_root}{os.sep}franka_robot{os.sep}actions"
    data_dict_dir["franka_forces"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_forces"
    data_dict_dir["franka_state"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_state"
    data_dict_dir["boxes_pos0"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box0"
    data_dict_dir["boxes_pos1"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box1"
    data_dict_dir["boxes_vel0"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box0"
    data_dict_dir["boxes_vel1"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box1"
    

    for cam_num in range(NUM_OF_CAMERAS):
        for camera_type in camera_types:
            data_dict_dir[f"cam{cam_num}_{camera_type}"] = f"{data_root}{os.sep}cameras{os.sep}{camera_type}{os.sep}cam{cam_num}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = singleSampleDataset(data_dict_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
    # Iterate over the DataLoader
    for batch in dataloader:
        mulitisensory_sample = batch
        for sample_key in mulitisensory_sample:
            print(f"mulitisensory_sample['{sample_key}'].shape = {mulitisensory_sample[sample_key].shape}")
            
            if "cam" in sample_key:
                
                cam_sensor_img = mulitisensory_sample[sample_key][0]
                
                # Convert tensors to numpy arrays for displaying
                cam_sensor_img = cam_sensor_img.permute(1, 2, 0).cpu().numpy()
                
                # Convert from float [0, 1] to uint8 [0, 255]
                cam_sensor_img = (cam_sensor_img * 255).astype('uint8')
                
                # Convert RGB to BGR for OpenCV
                cam_sensor_img = cv2.cvtColor(cam_sensor_img, cv2.COLOR_RGB2BGR)
                
                # Display the images using OpenCV
                cv2.imshow(f"{sample_key} Image", cam_sensor_img)
                
            else:
                csv_data = mulitisensory_sample[sample_key][0]
                print(f"{sample_key} data:")
                print(csv_data)
        
        # Wait until a key is pressed to close the images
       
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OK")
        print("EXITING...")
        exit(0)



def test_sequential_samples():
    NUM_OF_CAMERAS = 5
    camera_types = ["depth", "rgb", "segmented"]
    data_root = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data"
    sequence_length = 10

    data_dict_dir = {
        "franka_actions": f"{data_root}{os.sep}franka_robot{os.sep}actions",
        "franka_forces": f"{data_root}{os.sep}franka_robot{os.sep}dof_forces",
        "franka_state": f"{data_root}{os.sep}franka_robot{os.sep}dof_state",
        "boxes_pos0": f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box0",
        "boxes_pos1": f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box1",
        "boxes_vel0": f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box0",
        "boxes_vel1": f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box1"
    }
    
    for cam_num in range(NUM_OF_CAMERAS):
        for camera_type in camera_types:
            data_dict_dir[f"cam{cam_num}_{camera_type}"] = f"{data_root}{os.sep}cameras{os.sep}{camera_type}{os.sep}cam{cam_num}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = sequentialSampleDataset(data_dict_dir, transform=transform, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
    # Iterate over the DataLoader
    for batch in dataloader:
        mulitisensory_sample = batch
        for sample_key in mulitisensory_sample:
            print(f"mulitisensory_sample['{sample_key}'].shape = {mulitisensory_sample[sample_key].shape}")
            
            if "cam" in sample_key:
                cam_sensor_img_sequence = mulitisensory_sample[sample_key][0]
                for i in range(sequence_length):
                    cam_sensor_img = cam_sensor_img_sequence[i]
                    cam_sensor_img = cam_sensor_img.permute(1, 2, 0).cpu().numpy()
                    cam_sensor_img = (cam_sensor_img * 255).astype('uint8')
                    cam_sensor_img = cv2.cvtColor(cam_sensor_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"{sample_key} Image {i+1}", cam_sensor_img)
            else:
                csv_data_sequence = mulitisensory_sample[sample_key][0]
                print(f"{sample_key} data sequence:")
                print(csv_data_sequence)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OK")
        print("EXITING...")
        exit(0)


def rgb_to_class_index(rgb_image):

    colors = [(0, 0, 0),  # Background
              (25, 255, 255),  # Class 1
              (140, 25, 140),  # Class 2
              (140, 255, 25),  # Class 3
              (255, 25, 25)
            ]
    color_map = {tuple(color): idx for idx, color in enumerate(colors)}
 
    rgb_image = np.array(rgb_image)
    #print(f"rgb_image = {rgb_image}")
    height, width, _ = rgb_image.shape
    class_index_image = np.zeros((height, width), dtype=np.int64)
    
    for color, class_index in color_map.items():
        mask = np.all(rgb_image == color, axis=-1)
        class_index_image[mask] = class_index
    
    return class_index_image


def class_index_to_one_hot(class_index_image):
    num_classes = 5
    
    height, width = class_index_image.shape
    one_hot = np.zeros((num_classes, height, width), dtype=np.float32)
    
    for c in range(num_classes):
        one_hot[c, :, :] = (class_index_image == c)
    
    return torch.tensor(one_hot)

if __name__ == "__main__":
    #test_single_samples()
    test_sequential_samples()