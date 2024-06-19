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

        self.transform = transform
    
        # Collect all env directories
        self.env_dirs = sorted(os.listdir(data_dict_dir["cam0_rgb"]))
        
        # Initialize lists to hold image paths and csv data
        self.data = dict()
        for key in data_dict_dir:
            self.data[key] = []

        for env_num in self.env_dirs:
            for dict_key in data_dict_dir:

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

    
    
    def __len__(self):
        return len(self.data["cam0_rgb"])


    def __getitem__(self, idx):
        
        mulitisensory_sample = dict()

        for dict_key in self.data:
                
            if "cam" in dict_key:
                # Read images
                mulitisensory_sample[dict_key] = Image.open(self.data[dict_key][idx]).convert('RGB')
                #sensor2_image = Image.open(self.data["cam1_rgb"][idx]).convert('RGB')

                if self.transform:
                    mulitisensory_sample[dict_key] = self.transform(mulitisensory_sample[dict_key])
                    #sensor2_image = self.transform(sensor2_image)

            else:
                # Read CSV row
                read_data = self.data[dict_key][idx]
                # Convert CSV row to tensor
                mulitisensory_sample[dict_key] = torch.tensor(read_data, dtype=torch.float)
        
        return mulitisensory_sample
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

    dataset = CustomDataset(data_dict_dir, transform=transform)
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


if __name__ == "__main__":
    main()