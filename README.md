To make use of the codes please follow the following steps:
1. install isaac sim according to the instruciotns at https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html

2. use the IsaacLab folder provided in the projectCode (should beplaced in the installation directory of isaac sim e.g. ~/.local/share/ov/pkg/isaac-sim-4.0.0)

3. run the franka push expetinment to gater data (exmaple when running from location of issac lab: python source/standalone/workflows/rl_games/play.py --task=Isaac-Franka-Push-Direct-v0 --num_envs 1 --checkpoint /home/bart/.local/share/ov/pkg/isaac-sim-4.0.0/IsaacLab/logs/rl_games/franka_push_direct/2024-07-21_21-44-10/nn/franka_push_direct.pth --enable_cameras). You can use the povided model in "nn"
Note: before running the exeriemnt set the environemnt variables:
TEST_AND_SAVE_SENSORS to true
ERASE_EXISTING_DATA to true for the first run and then change it to false
RECORDED_DATA_DIR to the location where gathered data will be saved (when you go to the projectCode -> src -> ml -> model_classes.py it can be seen that this file looks for training data in "data_root = os.path.join(data_root, "..", "..", "..", "recorded_data_isaac_lab")" so place the recorded data in such a location. The same applies to the test set "os.path.join(data_root, "..", "..", "..", "test_set")")

4. Run to test different modalities run the projectCode -> notebooks -> train_models.py

5. recorded_data_isaac_lab shows what the training dataset looked like. In this folder there are only 2 episodes recorded.

Other sctipts in the "notebooks" folder were being modified as required. For exmaple the test_models.py is used to run tests on the test dataset, make_plots.py is used to make plots of predicions of cube positions, remove.py is used to remove a specifed episode, sort_models.py is used to sort models by loss on a validation set and makes xlsx fies.

All logic is in the projectCode -> src folder in the "ml" folder there are codes used to create machine learning models and in "data_processing" there are codes used to preprocess, read and supply the data to the machine learing models

These codes were tesed only on Ubuntu 20.04 LTS operating system.