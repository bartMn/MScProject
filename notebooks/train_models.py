import os
import sys
from itertools import product

# Add the `src` directory to the system path to allow importing modules
module_path = os.path.abspath(os.path.join(os.path.realpath(__file__), f"..{os.sep}..{os.sep}src"))
print(100*"@")
print(f"module_path = {module_path}")
if module_path not in sys.path:
    sys.path.append(module_path)

# Import model classes from the `ml` module
from ml.model_classes import ModelClass, twoCamsModelClass, multiModalClass#, fiveDepthCamsModel, depthAndAllFrankaDataModel

# Define input data keys for different model configurations
# Update configurations as needed
input_data_keys_dict = {"fiveRgbCamsModel": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "cam3_rgb", "cam4_rgb", "franka_actions", "franka_forces", "franka_state"],
                        "fiveDepthCamsModel": ["cam0_depth", "cam1_depth", "cam2_depth", "cam3_depth", "cam4_depth", "franka_actions", "franka_forces", "franka_state"],
                        "rgb1": ["cam0_rgb", "franka_actions", "franka_forces", "franka_state"],
                        "rgb2": ["cam1_rgb", "cam2_rgb", "franka_actions", "franka_forces", "franka_state"],
                        "rgb3": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "franka_actions", "franka_forces", "franka_state"],
                        "rgb2_v2": ["cam0_rgb", "cam1_rgb", "franka_actions", "franka_forces", "franka_state"],
                        "depth1": ["cam0_depth", "franka_actions", "franka_forces", "franka_state"],
                        "depth2": ["cam1_depth", "cam2_depth", "franka_actions", "franka_forces", "franka_state"],
                        "depth2_v2": ["cam0_depth", "cam1_depth", "franka_actions", "franka_forces", "franka_state"],
                        "depth3": ["cam0_depth", "cam1_depth", "cam2_depth", "franka_actions", "franka_forces", "franka_state"],
                        "rgb1depth1": ["cam0_rgb", "cam0_depth", "franka_actions", "franka_forces", "franka_state"],
                        "rgb1depth1_v2": ["cam0_rgb", "cam1_depth", "franka_actions", "franka_forces", "franka_state"],
                        ##"frankaDataOnly": ["franka_actions", "franka_forces", "franka_state"],
                        "rgb2depth2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth", "franka_actions", "franka_forces", "franka_state"],
                        ##"rgb2flow2": ["cam1_rgb", "cam2_rgb", "cam1_flow", "cam2_flow", "franka_actions", "franka_forces", "franka_state"],
                        ##"depth2flow2": ["cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow", "franka_actions", "franka_forces", "franka_state"],
                        ##"rgb2depth2flow2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow", "franka_actions", "franka_forces", "franka_state"],
                        ##
                        "noActionsfiveDepthCamsModel": ["cam0_depth", "cam1_depth", "cam2_depth", "cam3_depth", "cam4_depth" , "franka_forces", "franka_state"],
                        "noActionsfiveRgbCamsModel": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "cam3_rgb", "cam4_rgb", "franka_forces", "franka_state"],
                        "noActionsrgb1": ["cam0_rgb" , "franka_forces", "franka_state"],
                        "noActionsrgb2": ["cam1_rgb", "cam2_rgb" , "franka_forces", "franka_state"],
                        "noActionsrgb2_v2": ["cam0_rgb", "cam1_rgb", "franka_forces", "franka_state"],
                        "noActionsrgb3": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "franka_forces", "franka_state"],
                        "noActionsdepth1": ["cam0_depth", "franka_forces", "franka_state"],
                        "noActionsdepth2": ["cam1_depth", "cam2_depth" , "franka_forces", "franka_state"],
                        "noActionsdepth2_v2": ["cam0_depth", "cam1_depth", "franka_forces", "franka_state"],
                        "noActionsdepth3": ["cam0_depth", "cam1_depth", "cam2_depth", "franka_forces", "franka_state"],
                        ##"noActionsfrankaDataOnly": ["franka_forces", "franka_state"],
                        "noActionsrgb1depth1": ["cam0_rgb", "cam0_depth", "franka_forces", "franka_state"],
                        "noActionsrgb1depth1_v2": ["cam0_rgb", "cam1_depth", "franka_forces", "franka_state"],
                        "noActionsrgb2depth2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth" , "franka_forces", "franka_state"],
                        ##"noActionsrgb2flow2": ["cam1_rgb", "cam2_rgb", "cam1_flow", "cam2_flow" , "franka_forces", "franka_state"],
                        ##"noActionsdepth2flow2": ["cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow" , "franka_forces", "franka_state"],
                        ##"noActionsrgb2depth2flow2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow" , "franka_forces", "franka_state"],
                        ##                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                        "noFrankaDatafiveDepthCamsModel": ["cam0_depth", "cam1_depth", "cam2_depth", "cam3_depth", "cam4_depth"],
                        "noFrankaDatafiveRgbCamsModel": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "cam3_rgb", "cam4_rgb"],
                        "noFrankaDatargb1": ["cam0_rgb"],
                        "noFrankaDatargb2": ["cam1_rgb", "cam2_rgb"],
                        "noFrankaDatargb2_v2": ["cam0_rgb", "cam1_rgb"],
                        "noFrankaDatargb3": ["cam0_rgb" , "cam1_rgb", "cam2_rgb"],
                        "noFrankaDatadepth1": ["cam0_depth"],
                        "noFrankaDatadepth2": ["cam1_depth", "cam2_depth"],
                        "noFrankaDatadepth2_v2": ["cam0_depth", "cam1_depth"],
                        "noFrankaDatargb1depth1": ["cam0_rgb", "cam0_depth"],
                        "noFrankaDatargb1depth1_v2": ["cam0_rgb", "cam1_depth"],
                        "noFrankaDatargb2depth2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth"],
                        ##"noFrankaDatargb2flow2": ["cam1_rgb", "cam2_rgb", "cam1_flow", "cam2_flow"],
                        ##"noFrankaDatadepth2flow2": ["cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow"],
                        ##"noFrankaDatargb2depth2flow2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth", "cam1_flow", "cam2_flow"]
                        "frankaStatefiveRgbCamsModel": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "cam3_rgb", "cam4_rgb", "franka_state"],
                        "frankaStatefiveDepthCamsModel": ["cam0_depth", "cam1_depth", "cam2_depth", "cam3_depth", "cam4_depth", "franka_state"],
                        "frankaStatergb1": ["cam0_rgb", "franka_state"],
                        "frankaStatergb2_v2": ["cam0_rgb", "cam1_rgb", "franka_state"],
                        "frankaStatergb2": ["cam1_rgb", "cam2_rgb", "franka_state"],
                        "frankaStatergb3": ["cam0_rgb", "cam1_rgb", "cam2_rgb", "franka_state"],
                        "frankaStatedepth1": ["cam0_depth", "franka_state"],
                        "frankaStatedepth2": ["cam1_depth", "cam2_depth", "franka_state"],
                        "frankaStatedepth2_v2": ["cam0_depth", "cam1_depth", "franka_state"],
                        "frankaStatedepth3": ["cam0_depth", "cam1_depth", "cam2_depth", "franka_state"],
                        ##"noActionsfrankaDataOnly": ["franka_forces", "franka_state"],
                        "frankaStatergb1depth1": ["cam0_rgb", "cam0_depth", "franka_state"],
                        "frankaStatergb1depth1_v2": ["cam0_rgb", "cam1_depth", "franka_state"],
                        "frankaStatergb2depth2": ["cam1_rgb", "cam2_rgb", "cam1_depth", "cam2_depth", "franka_state"],
                        }
    
# Define the root directory for saving results
results_root_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.path.join(f"..{os.sep}..{os.sep}results")))

# Define output data keys and model configurations
output_data_key_list = ["boxes_pos0"] #["cam0_flow", "boxes_pos0"]
sequential_modes_list = [True]
CNN_modes_list = [False] #[True, False]
sq_len_list = [2]

# Generate all combinations of model configurations
list_of_all_combinations = product(sequential_modes_list, CNN_modes_list, output_data_key_list, sq_len_list)
dicts_of_all_modes = [{"sequential": el1, "pretrained_resnet": el2, "output_data_key": el3, "sq_len": el4} for el1, el2, el3, el4 in list_of_all_combinations]

# Iterate over all model configurations
for model_modes in dicts_of_all_modes:
    model_name_prefix = ""
    sequence_length = 0

    # Check if the model is sequential and set parameters accordingly
    if model_modes["sequential"]:
        model_name_prefix += "sequential_"
        use_sequential_model = True
        batch_size_to_use = 32
        sequence_length = model_modes["sq_len"]
        model_name_prefix += str(sequence_length)
        model_name_prefix += "_"
        epochs_num = 20
    else:
        use_sequential_model = False
        batch_size_to_use = 64
        epochs_num = 20
        
    # Configure ResNet models
    if model_modes["pretrained_resnet"]:
        model_name_prefix += "fineTunedResnet18_"
        useResnet = True
        usePretrainedResnet = False
        
    else:
        useResnet = False
        usePretrainedResnet = False

    # Configure output data key
    output_data_key = model_modes["output_data_key"]
    if output_data_key == "cam0_flow":
        learing_rate = 0.0002
        batch_size_to_use = 32
    else:
        learing_rate = 0.001
    
    
    # Print model configuration details
    print(f"output_data_key = {output_data_key}")
    print(f"model_modes = {model_modes}")
    print(f"learing_rate = {learing_rate}")
    print(f"batch_size_to_use = {batch_size_to_use}")

    # Create results directory for the current output data key
    results_dir = os.path.join(results_root_dir, output_data_key)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Iterate over all model names
    for model_name in input_data_keys_dict:
        model_under_test_name = model_name_prefix + model_name

        # Skip models that have already been tested
        if os.path.exists(os.path.join(results_dir, model_under_test_name, "model.pt")):
            print(f"SKIPPIG: model {model_under_test_name} has already been tested")
            continue

        # Initialize the model
        model_under_test = multiModalClass(batch_size= batch_size_to_use,
                                           epochs_num = epochs_num,
                                           save_path= os.path.join(results_dir, model_under_test_name),
                                           input_data_keys = input_data_keys_dict[model_name],
                                           output_data_key = output_data_key,
                                           train_val_split = 0.8,
                                           sequential_data= use_sequential_model,
                                           sequence_length= sequence_length,
                                           useResnet = useResnet,
                                           usePretrainedResnet = usePretrainedResnet,
                                           do_segmentation = False,
                                           fusing_before_RNN = False)

        # Initialize the model's architecture
        model_under_test.init_model(learning_rate=learing_rate, input_data_keys = input_data_keys_dict[model_name])

        # Train the model
        print(f"training model {model_under_test_name}...")
        model_under_test.train_model(save_pt_model= True)
        print(f"training {model_under_test_name} finished")
