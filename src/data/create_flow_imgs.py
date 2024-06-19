import numpy as np
import cv2 as cv
import os
from itertools import product
from tqdm import tqdm
import threading


def get_flow_and_save(source, dest, env_num, cam_num):
    frame1_source = source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}{os.sep}frame0.png"
    #frame2_source = source +os.sep+ "rgb_env0_cam0_frame20.png"

    frame1 = cv.imread(frame1_source)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Create and save a black image with the same shape as prvs
    black_image = np.zeros_like(prvs)
    cv.imwrite(f'{dest+os.sep}cam{cam_num}{os.sep}env{env_num}{os.sep}frame0.png', black_image)

    frames_num = max([int(f.replace("frame", "").replace(".png", "")) for f in os.listdir(source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}")])
    
    for i in tqdm(range(1, frames_num+1)):
        frame2_source = source +os.sep+ f"cam{cam_num}{os.sep}env{env_num}{os.sep}frame{i}.png"

        frame2 = cv.imread(frame2_source)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        cv.imwrite(f'{dest+os.sep}cam{cam_num}{os.sep}env{env_num}{os.sep}frame{i}.png', bgr)
        prvs = next



def main():
    source = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data/cameras/rgb"
    dest = "/home/bart/project/IsaacGym_Preview_4_Package/isaacgym/IsaacGymEnvs-main/isaacgymenvs/recorded_data/cameras/flow"

    envs_num = max([int(f.replace("env", "")) for f in os.listdir(source +os.sep+ "cam0")])
    camera_env_combinations = list(product([env_n for env_n in range(envs_num+1)], [cam_n for cam_n in range(5)]))

    for env_num, cam_num in tqdm(camera_env_combinations):
        #get_flow_and_save(source, dest, env_num, cam_num)
        thread = threading.Thread(target=get_flow_and_save, args=(source, dest, env_num, cam_num))
        thread.start()

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()