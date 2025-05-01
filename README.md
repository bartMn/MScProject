# MSc Project: Multisensory Data Processing and Machine Learning for Robotics

This repository contains the codebase for MSc project, which focuses on multisensory data processing and machine learning for robotics applications. The project leverages data collected from simulated environments to train and evaluate machine learning models for robotic tasks.

---

## Project Overview

This project integrates simulation-based data collection, preprocessing pipelines, and machine learning models to solve robotic tasks. The key components include:
- **Data Collection**: Using Isaac Sim to simulate robotic environments and collect multisensory data (e.g., RGB, depth, segmentation, and force data).
- **Data Processing**: Preprocessing and organizing the collected data for training machine learning models.
- **Model Training**: Training deep learning models for tasks such as object manipulation and trajectory prediction.

---

## Installation

1. **Install Isaac Sim**  
   Follow the official [Isaac Sim installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).

2. **Set Up the Project**  
   Place the `IsaacLab` folder in the Isaac Sim installation directory (e.g., `~/.local/share/ov/pkg/isaac-sim-4.0.0`).

3. **Install Python Dependencies**  
   Use the following command to install the required Python packages:
   ```bash
   pip install -r requirements.txt


## Data Collection
To collect data using the Franka Push experiment:

1. Set the following environment variables before running the experiment:
    TEST_AND_SAVE_SENSORS=true
    ERASE_EXISTING_DATA=true (set to false after the first run)
    RECORDED_DATA_DIR=/path/to/recorded_data

2. Run the following command (You can use the povided model in "nn"):
    python source/standalone/workflows/rl_games/play.py --task=Isaac-Franka-Push-Direct-v0
    --num_envs 1
    --checkpoint {path_to_isaac_sim}/isaac-sim-4.0.0/IsaacLab/logs/rl_games/franka_push_direct/2024-07-21_21-44-10/nn/franka_push_direct.pth
    --enable_cameras. 

3. Run the script to generate RGB-flow images:
    python src/data_processing/create_flow_imgs.py

## Model Training
The notebooks/train_models.py script is used to train machine learning models. It supports various data modalities (e.g., RGB, depth, segmentation) and different tasks (generating RGB camera image, generating RGB-flow camera image, generating RGB-segmented camera image, future object position).

Steps:
1. Update the dataset paths in the script.
2. Run the training script:
    python notebooks/train_models.py

## Results and Analysis
The results of the trained models, including metrics and visualizations, are saved in the results/ directory. 

## Disclaimer
These codes were tesed only on Ubuntu 20.04 LTS operating system.