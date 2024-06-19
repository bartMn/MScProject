import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import cv2
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


class CustomDataset(Dataset):
    def __init__(self, data_dict_dir, transform=None):
        #self.sensor1_dir = sensor1_dir
        #self.sensor2_dir = sensor2_dir
        #self.sensor3_dir = sensor3_dir
        self.transform = transform
        

        sensor1_dir = data_dict_dir["cam0_rgb"]
        sensor2_dir = data_dict_dir["cam1_rgb"]
        sensor3_dir = data_dict_dir["boxes_pos"]
        # Collect all env directories
        self.env_dirs = sorted(os.listdir(sensor1_dir))
        
        # Initialize lists to hold image paths and csv data
        self.data = dict()
        self.data["cam0_rgb"] = []
        self.data["cam1_rgb"] = []
        self.data["boxes_pos"] = []

        for env_num in self.env_dirs:
            sensor1_env_dir = os.path.join(sensor1_dir, env_num)
            sensor2_env_dir = os.path.join(sensor2_dir, env_num)
            sensor3_env_file = os.path.join(sensor3_dir, env_num, 'cube0.csv')
            
            # Get image files
            cam0_rgb = sorted(os.listdir(sensor1_env_dir), key = natural_sort_key)
            cam1_rgb = sorted(os.listdir(sensor2_env_dir), key = natural_sort_key)
            
            # Read csv file
            sensor3_df = pd.read_csv(sensor3_env_file)
            
            for img in cam0_rgb:
                self.data["cam0_rgb"].append(os.path.join(sensor1_env_dir, img))
            for img in cam1_rgb:
                self.data["cam1_rgb"].append(os.path.join(sensor2_env_dir, img))
            
            self.data["boxes_pos"].extend(sensor3_df.values.tolist())
    
    def __len__(self):
        return len(self.data["cam0_rgb"])

    def __getitem__(self, idx):
        # Read images
        sensor1_image = Image.open(self.data["cam0_rgb"][idx]).convert('RGB')
        sensor2_image = Image.open(self.data["cam1_rgb"][idx]).convert('RGB')
        
        # Read CSV row
        sensor3_row = self.data["boxes_pos"][idx]
        
        if self.transform:
            sensor1_image = self.transform(sensor1_image)
            sensor2_image = self.transform(sensor2_image)

        # Convert CSV row to tensor
        sensor3_tensor = torch.tensor(sensor3_row, dtype=torch.float)
        
        return sensor1_image, sensor2_image, sensor3_tensor


# Usage
def main():

    NUM_OF_CAMERAS = 5
    camera_types = ["depth", "rgb", "segmented"]#, "flow"]
    data_root = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data"

    data_dict_dir = dict()
    data_dict_dir["franka_actions"] = f"{data_root}{os.sep}franka_robot{os.sep}actions"
    data_dict_dir["franka_forces"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_forces"
    data_dict_dir["franka_state"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_state"
    data_dict_dir["boxes_pos"] = f"{data_root}{os.sep}boxes{os.sep}position"
    data_dict_dir["boxes_vel"] = f"{data_root}{os.sep}boxes{os.sep}velocity"
    

    for cam_num in range(NUM_OF_CAMERAS):
        for camera_type in camera_types:
            data_dict_dir[f"cam{cam_num}_{camera_type}"] = f"{data_root}{os.sep}cameras{os.sep}{camera_type}{os.sep}cam{cam_num}"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(data_dict_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
    # Iterate over the DataLoader
    for batch in dataloader:
        sensor1_imgs, sensor2_imgs, sensor3_data = batch
        print(f"sensor1_imgs.shape = {sensor1_imgs.shape}")
        print(f"sensor2_imgs.shape = {sensor2_imgs.shape}")
        print(f"sensor3_data.shape = {sensor3_data.shape}")

        first_sensor1_img = sensor1_imgs[0]
        first_sensor2_img = sensor2_imgs[0]
        first_sensor3_data = sensor3_data[0]
        
        # Convert tensors to numpy arrays for displaying
        first_sensor1_img_np = first_sensor1_img.permute(1, 2, 0).cpu().numpy()
        first_sensor2_img_np = first_sensor2_img.permute(1, 2, 0).cpu().numpy()
        
        # Convert from float [0, 1] to uint8 [0, 255]
        first_sensor1_img_np = (first_sensor1_img_np * 255).astype('uint8')
        first_sensor2_img_np = (first_sensor2_img_np * 255).astype('uint8')
        
        # Convert RGB to BGR for OpenCV
        first_sensor1_img_np = cv2.cvtColor(first_sensor1_img_np, cv2.COLOR_RGB2BGR)
        first_sensor2_img_np = cv2.cvtColor(first_sensor2_img_np, cv2.COLOR_RGB2BGR)
        
        # Display the images using OpenCV
        cv2.imshow("Sensor 1 Image", first_sensor1_img_np)
        cv2.imshow("Sensor 2 Image", first_sensor2_img_np)
        
        # Print the corresponding row from sensor3
        print("First row from sensor3 data:")
        print(first_sensor3_data)
        
        # Wait until a key is pressed to close the images
       
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("OK")
        print("EXITING...")
        exit(0)


if __name__ == "__main__":
    main()