import numpy as np
import cv2 as cv
import os
from itertools import product
from tqdm import tqdm
import threading


def get_flow_and_save(source, dest, env_num, cam_num):
    """
    Computes optical flow between consecutive frames and saves the resulting flow images.

    Args:
        source (str): Path to the source directory containing RGB frames.
        dest (str): Path to the destination directory to save flow images.
        env_num (int): Environment number (e.g., env0, env1).
        cam_num (int): Camera number (e.g., cam0, cam1).

    Returns:
        None
    """
    # Path to the first frame in the sequence
    frame1_source = source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}{os.sep}frame0.png"
    
    # Read the first frame and convert it to grayscale
    frame1 = cv.imread(frame1_source)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    # Initialize an HSV image for visualizing optical flow
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Create and save a black image for the first frame in the destination directory
    black_image = np.zeros_like(prvs)
    cv.imwrite(f'{dest+os.sep}cam{cam_num}{os.sep}env{env_num}{os.sep}frame0.png', black_image)

    # Get the total number of frames in the current environment and camera
    frames_num = max([int(f.replace("frame", "").replace(".png", "")) for f in os.listdir(source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}")])
    
    # Iterate over all frames starting from the second frame
    for i in tqdm(range(1, frames_num+1)):
        # Path to the current frame
        frame2_source = source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}{os.sep}frame{i}.png"

        # Read the current frame and convert it to grayscale
        frame2 = cv.imread(frame2_source)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Compute optical flow using the Farneback method
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert flow to HSV format for visualization
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

        # Convert HSV to BGR for saving as an image
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        # Save the flow image to the destination directory
        cv.imwrite(f'{dest+os.sep}cam{cam_num}{os.sep}env{env_num}{os.sep}frame{i}.png', bgr)
        
        # Update the previous frame for the next iteration
        prvs = next



def main():
    """
    Main function to compute and save optical flow for all environments and cameras.

    Args:
        None

    Returns:
        None
    """

    # Define the root directory for data
    data_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(data_root, "..", "..", "..", "recorded_data_isaac_lab")
    
    # Define source and destination directories for RGB and flow images
    source = os.path.join(data_root, "cameras", "rgb")
    dest = os.path.join(data_root, "cameras", "flow")
    
    # Get the total number of environments
    envs_num = max([int(f.replace("env", "")) for f in os.listdir(source +os.sep+ "cam0")])
    
    # Generate all combinations of environments and cameras
    camera_env_combinations = list(product([env_n for env_n in range(envs_num+1)], [cam_n for cam_n in range(5)]))

    # Process each environment-camera combination in parallel
    for env_num, cam_num in tqdm(camera_env_combinations):
        # Start a new thread for each combination to compute and save optical flow
        thread = threading.Thread(target=get_flow_and_save, args=(source, dest, env_num, cam_num))
        thread.start()

    # Close all OpenCV windows
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()