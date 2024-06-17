
from isaacgym import gymapi
import torch 
#import mujoco

def check_gpu_available():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
        print("Device name:", torch.cuda.get_device_name(0))  # Prints the name of the GPU
        print("CUDA Version:", torch.version.cuda)  # Prints the CUDA version
        print("Number of GPU(s) available:", torch.cuda.device_count())  # Prints the number of available GPUs
    else:
        print("GPU is not available. Using CPU instead.")

#def check_mujoco():
    #Verify MuJoCo installation
#    print(f"MuJoCo version: {mujoco.__version__}")


if __name__ == "__main__":
    check_gpu_available()
#    check_mujoco()
