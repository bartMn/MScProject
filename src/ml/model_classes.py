"""
Module: model_classes.py
Description: Defines the ModelClass and its variants for training and evaluating machine learning models.
This module includes functionality for data loading, model initialization, training, validation, and testing.
"""

import torch
from torch.optim import SGD, Adam, Adagrad, RMSprop, Adadelta, AdamW, SparseAdam, Adamax, LBFGS
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Tuple, List, Dict
from copy import deepcopy

import os
import sys
from PIL import Image as im
#module_path = os.path.abspath(os.path.join("../data"))
module_path = os.path.dirname(os.path.abspath(__file__))

module_path = os.path.join(module_path, f"..")
if module_path not in sys.path:
    sys.path.append(module_path)
from data_processing.data_read import singleSampleDataset, sequentialSampleDataset
from ml.models import *

torch.backends.cudnn.enabled = False



    
class ModelClass():
    """
    A base class for managing machine learning models, including data loading, training, and evaluation.

    Attributes:
        kwargs (dict): Additional keyword arguments for customization.
        output_data_key (str): Key for the output data.
        size_of_output (int): Size of the output data (e.g., 3 for position, 5 for segmentation).
        sequential_data (bool): Whether the data is sequential.
        device (torch.device): Device to run the model on (CPU or GPU).
        save_path (str): Path to save the trained model.
        epochs_num (int): Number of training epochs.
        training_loader (DataLoader): DataLoader for the training set.
        validation_loader (DataLoader): DataLoader for the validation set.
    """

    def __init__(self,
                 data_path:str= None,
                 train_val_split:float=0.9,
                 batch_size:int=32,
                 epochs_num:int=20,
                 save_path:str=None,
                 path_to_load_model:str=None,
                 input_data_keys:str = None,
                 output_data_key:str = None,
                 sequential_data:bool = False,
                 sequence_length:int = 2,
                 **kwargs) -> None:
        
        """
        Initializes the ModelClass with data paths, training parameters, and configurations.

        Args:
            data_path (str): Path to the dataset.
            train_val_split (float): Ratio of training to validation data.
            batch_size (int): Batch size for training.
            epochs_num (int): Number of training epochs.
            save_path (str): Path to save the trained model.
            path_to_load_model (str): Path to load a pre-trained model.
            input_data_keys (str): Keys for input data.
            output_data_key (str): Key for output data.
            sequential_data (bool): Whether the data is sequential.
            sequence_length (int): Length of the sequence for sequential data.
            **kwargs: Additional keyword arguments for customization.
        """
      

        self.kwargs = kwargs
        self.output_data_key = output_data_key if output_data_key else "boxes_pos0"
        if "seg" in self.output_data_key:
            self.size_of_output = 5
        else:
            self.size_of_output= 3

        self.sequential_data = sequential_data

        # Handle sequential data configurations
        if sequential_data:
            if 'future_sample_num' in self.kwargs:
                self.future_sample_num = self.kwargs['future_sample_num']
            else:
                self.future_sample_num = 0

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory if it doesn't exist
        self.save_path = save_path
        if self.save_path and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.epochs_num = epochs_num
              
        # Define data directories
        NUM_OF_CAMERAS = 5
        camera_types = ["depth", "rgb", "segmented", "flow", "flow_no_robot"]
        data_root = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(data_root, "..", "..", "..", "recorded_data_isaac_lab")
        

        # Map data keys to their respective directories
        data_dict_dir = dict()
        data_dict_dir["franka_actions"] = f"{data_root}{os.sep}franka_robot{os.sep}actions"
        data_dict_dir["franka_forces"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_forces"
        data_dict_dir["franka_state"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_state"
        data_dict_dir["boxes_pos0"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box0"
        data_dict_dir["boxes_pos1"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box1"
        data_dict_dir["boxes_vel0"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box0"
        data_dict_dir["boxes_vel1"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box1"


        # Add camera data directories
        for cam_num in range(NUM_OF_CAMERAS):
            for camera_type in camera_types:
                data_dict_dir[f"cam{cam_num}_{camera_type}"] = f"{data_root}{os.sep}cameras{os.sep}{camera_type}{os.sep}cam{cam_num}"
        
        # Handle output data key conflicts
        if self.output_data_key in input_data_keys:
            new_key = self.output_data_key + "_out"
            data_dict_dir[new_key] = data_dict_dir[self.output_data_key]
            self.output_data_key = new_key

            #print(f"data_dict_dir = {data_dict_dir}")
        #transform = transforms.Compose([
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])

        if 'do_segmentation' in self.kwargs:
            one_hot_for_segmentation = self.kwargs['do_segmentation']
        else:
            one_hot_for_segmentation = False


        # Define transformations for input and output data
        transform = transforms.Compose([
                                    transforms.Resize(256),         # Resize the image to 256x256 pixels
                                    transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
                                    transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],  # Normalize the image based on ImageNet statistics
                                                        std=[0.229, 0.224, 0.225]
                                                        )
                                    ])
        
        if one_hot_for_segmentation:
            transform_output_imgs = transforms.Compose([
                                        transforms.Resize(256),         # Resize the image to 256x256 pixels
                                        transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                        #transforms.ToTensor()           # Convert the image to a PyTorch tensor
                                        ])
        elif "cam" in self.output_data_key:
            transform_output_imgs = transforms.Compose([
                                    #transforms.Resize(224),
                                    transforms.Resize(256),         # Resize the image to 256x256 pixels
                                    transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                    transforms.ToTensor()           # Convert the image to a PyTorch tensor
                                    ])
        else:
            transform_output_imgs = None
        
        # Set random seed for reproducibility
        torch.manual_seed(0)

        # Prepare datasets and DataLoaders
        self.used_keys = input_data_keys + [self.output_data_key]
        if self.sequential_data:
            self.sequence_length = sequence_length
            full_training_set = sequentialSampleDataset(data_dict_dir,
                                                        transform=transform,
                                                        sequence_length=sequence_length,
                                                        transform_output_imgs = transform_output_imgs,
                                                        output_data_key = self.output_data_key,
                                                        used_keys = self.used_keys,
                                                        one_hot_for_segmentation = one_hot_for_segmentation,
                                                        future_sample_num = self.future_sample_num)
        else:
            full_training_set = singleSampleDataset(data_dict_dir,
                                                    transform=transform,
                                                    transform_output_imgs = transform_output_imgs,
                                                    output_data_key =output_data_key,
                                                    used_keys = self.used_keys,
                                                    one_hot_for_segmentation = one_hot_for_segmentation)
            
        self.csv_min_max = full_training_set.csv_min_max.copy()

        # Split dataset into training and validation sets
        train_size = int(train_val_split * len(full_training_set))
        val_size = len(full_training_set) - train_size
        training_set, validation_set = random_split(full_training_set, [train_size, val_size])


        # Create DataLoaders
        self.training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=7)
        self.validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=3)



    def load_model(self, path_to_load_model):
        """
        Loads a pre-trained model from the specified path.

        Args:
            path_to_load_model (str): Path to the pre-trained model file.

        Returns:
            None
        """
        self.model = torch.load(path_to_load_model, map_location=self.device) # Load the model
        self.model.to(self.device) # Move the model to the appropriate device


    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
        """
        Initializes the model with a ResNet18 backbone and a custom fully connected layer.

        Args:
            neueons_in_hidden_layer (int): Number of neurons in the hidden layer of the custom fully connected layer.
            dropout (float): Dropout rate for the custom fully connected layer.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            None
        """

        # Load a pre-trained ResNet18 model
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the fully connected layer with a custom layer
        self.model.fc = customFullyConnectedLayer(neueons_in_hidden_layer = neueons_in_hidden_layer, dropout= dropout, size_of_output= self.size_of_output)

        # Get parameters of the custom fully connected layer
        custom_fc_params = list(self.model.fc.parameters()) 

        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False 

        # Unfreeze parameters in the custom fully connected layer
        for param in custom_fc_params:
            param.requires_grad = True  

        # Define the optimizer and loss function
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        
        # Move the model to the appropriate device
        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        """
        Extracts inputs and labels from the dataset.

        Args:
            data (dict): A dictionary containing input data and labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Inputs (torch.Tensor): The input data (e.g., RGB images) moved to the appropriate device.
                - Labels (torch.Tensor): The corresponding labels (e.g., positions or segmentation masks) moved to the appropriate device.
        """
        
        return data["cam0_rgb"].to(self.device), data[self.output_data_key][:,  : self.size_of_output].to(self.device)


    def train_epoch(self, epoch_index: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch_index (int): The index of the current epoch.

        Returns:
            float: The average training loss for the epoch.
        """
       
        running_loss = 0.0 # Initialize the running loss for the epoch

        # Iterate over the training DataLoader
        for i, data in tqdm(enumerate(self.training_loader), total=len(self.training_loader)):
            # Extract inputs and labels from the current batch
            inputs, labels = self.get_inputs_and_labels(data)

            # Zero the gradients for the optimizer
            self.optimizer.zero_grad()
            
            # Perform a forward pass to compute predictions
            outputs = self.model(inputs)

            # Compute the loss between predictions and labels
            loss = self.loss_fn(outputs, labels)

            # Perform a backward pass to compute gradients
            loss.backward()
            
            # Update the model parameters using the optimizer
            self.optimizer.step()
            
            # Accumulate the loss for reporting
            running_loss += loss.item()
            
        # Compute the average loss for the epoch
        return running_loss / (i+1)
   
    
    def denormalize(self, data, key, epsilon=1e-8):
        """
        Denormalizes the given data using the min and max values for the specified key.

        Args:
            data (torch.Tensor): The normalized data to be denormalized.
            key (str): The key corresponding to the data in the dataset.
            epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-8.

        Returns:
            torch.Tensor: The denormalized data.
        """
        # Retrieve the min and max values for the specified key
        min_val, max_val = self.csv_min_max[key]

        # Convert min_val and max_val to tensors if they are not already
        if not isinstance(min_val, torch.Tensor):
            min_val = torch.tensor(min_val[ : self.size_of_output], device=self.device, dtype=data.dtype)
        if not isinstance(max_val, torch.Tensor):
            max_val = torch.tensor(max_val[ : self.size_of_output], device=self.device, dtype=data.dtype)

        # Compute the range (max - min) and handle cases where the range is zero
        numerator = max_val - min_val
        numerator = torch.where(numerator == 0, epsilon, numerator)
        
        # Denormalize the data using the formula: data * (max - min) + min
        return data * numerator + min_val


    def train_model(self, epochs: int = None, save_pt_model:bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains the model for a specified number of epochs and optionally saves the best model.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to `self.epochs_num`.
            save_pt_model (bool): Whether to save the best model as a `.pt` file. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays containing training and validation losses for each epoch.
        """
        
        if epochs is None:
            epochs = self.epochs_num
            
        train_loss = np.zeros(epochs)
        valid_loss = np.zeros(epochs)
        best_vloss = float('inf') # Track the best validation loss
        best_avg_vloss_n_original_scale = float('inf') # Best validation loss in original scale
        best_epoch_model = deepcopy(self.model) # Save the best model
        best_epochs = 0 # Track the epoch with the best validation loss
        
        for current_epoch in range(epochs):
            print(f'EPOCH {current_epoch}:')    

            # Enable training mode
            self.model.train(True)
            avg_loss = self.train_epoch(current_epoch)  # Train for one epoch  
            
            # Disable training mode for validation
            self.model.train(False)  

            epoch_vloss = 0.0
            epoch_vloss_in_original_scale = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = self.get_inputs_and_labels(vdata)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    epoch_vloss += vloss

                    # Compute validation loss in original scale if applicable
                    if "cam" not in self.output_data_key:
                        vloss_in_original_scale = self.loss_fn(self.denormalize(voutputs, self.output_data_key), self.denormalize(vlabels, self.output_data_key))
                    else:
                        vloss_in_original_scale = 0
                    epoch_vloss_in_original_scale += vloss_in_original_scale

            avg_vloss = epoch_vloss / (i + 1)
            avg_vloss_n_original_scale = epoch_vloss_in_original_scale / (i + 1)
            print(f'LOSS train: {avg_loss} valid: {avg_vloss}, valid (in m): {avg_vloss_n_original_scale}')
            train_loss[current_epoch] = avg_loss
            valid_loss[current_epoch] = avg_vloss   
            
            # Update the best model if the current validation loss is better
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_avg_vloss_n_original_scale = avg_vloss_n_original_scale
                best_epoch_model = deepcopy(self.model)
                best_epochs = current_epoch

        # Save the best model if required
        if  self.save_path and save_pt_model:
            torch.save(best_epoch_model, os.path.join(self.save_path, "model.pt"))
        
        # Plot training and validation losses
        plot_training(train_loss, valid_loss, best_vloss, best_epochs, best_avg_vloss_n_original_scale, self.save_path)    

        return train_loss, valid_loss


    def test_model(self,
                   model: torch.nn.Module = None,
                   use_test_set:bool = False,
                   predictions_dir:str = None):
                 
        """
        Tests the model on the validation or test set and optionally saves predictions.

        Args:
            model (torch.nn.Module, optional): The model to test. Defaults to `self.model`.
            use_test_set (bool): Whether to use the test set instead of the validation set. Defaults to False.
            predictions_dir (str, optional): Directory to save predictions. Defaults to None.

        Returns:
            float: Average validation/test loss.
        """

        
        if model is None:
            model = self.model

        # Use the validation loader by default    
        if use_test_set is False:
            test_loader = self.validation_loader

        else:
            # Prepare the test set if specified
            NUM_OF_CAMERAS = 5
            camera_types = ["depth", "rgb", "segmented", "flow", "flow_no_robot"]
            data_root = "/home/bart/project/test_set"

            # Define data directories for the test set
            data_dict_dir = dict()
            data_dict_dir["franka_actions"] = f"{data_root}{os.sep}franka_robot{os.sep}actions"
            data_dict_dir["franka_forces"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_forces"
            data_dict_dir["franka_state"] = f"{data_root}{os.sep}franka_robot{os.sep}dof_state"
            data_dict_dir["boxes_pos0"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box0"
            data_dict_dir["boxes_pos1"] = f"{data_root}{os.sep}boxes{os.sep}position{os.sep}box1"
            data_dict_dir["boxes_vel0"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box0"
            data_dict_dir["boxes_vel1"] = f"{data_root}{os.sep}boxes{os.sep}velocity{os.sep}box1"

            # Add camera data directories
            for cam_num in range(NUM_OF_CAMERAS):
                for camera_type in camera_types:
                    data_dict_dir[f"cam{cam_num}_{camera_type}"] = f"{data_root}{os.sep}cameras{os.sep}{camera_type}{os.sep}cam{cam_num}"
            

            # Handle output data key conflicts
            self.output_data_key = self.output_data_key.replace("_out", "")
            print(f"self.output_data_key = {self.output_data_key}")
            if self.output_data_key in self.input_data_keys:
                new_key = self.output_data_key + "_out"
                data_dict_dir[new_key] = data_dict_dir[self.output_data_key]
                self.output_data_key = new_key
                print(f"data_dict_dir = {data_dict_dir}")
            
            self.used_keys = self.input_data_keys + [self.output_data_key]
            #transform = transforms.Compose([
            #    transforms.Resize((224, 224)),
            #    transforms.ToTensor()
            #])

            if 'do_segmentation' in self.kwargs:
                one_hot_for_segmentation = self.kwargs['do_segmentation']
            else:
                one_hot_for_segmentation = False

            transform = transforms.Compose([
                                        transforms.Resize(256),         # Resize the image to 256x256 pixels
                                        transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                        transforms.ToTensor(),           # Convert the image to a PyTorch tensor
                                        transforms.Normalize(
                                                            mean=[0.485, 0.456, 0.406],  # Normalize the image based on ImageNet statistics
                                                            std=[0.229, 0.224, 0.225]
                                                            )
                                        ])

            if one_hot_for_segmentation:
                transform_output_imgs = transforms.Compose([
                                            transforms.Resize(256),         # Resize the image to 256x256 pixels
                                            transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                            #transforms.ToTensor()           # Convert the image to a PyTorch tensor
                                            ])
            elif "cam" in self.output_data_key:
                transform_output_imgs = transforms.Compose([
                                        #transforms.Resize(224),
                                        transforms.Resize(256),         # Resize the image to 256x256 pixels
                                        transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                        transforms.ToTensor()           # Convert the image to a PyTorch tensor
                                        ])
            else:
                transform_output_imgs = None
            

            # Prepare the test dataset
            if self.sequential_data:
                test_set = sequentialSampleDataset(data_dict_dir,
                                                   transform=transform,
                                                   sequence_length= self.sequence_length,
                                                   transform_output_imgs = transform_output_imgs,
                                                   output_data_key = self.output_data_key,
                                                   used_keys = self.used_keys,
                                                   one_hot_for_segmentation = one_hot_for_segmentation,
                                                   future_sample_num = self.future_sample_num)
            else:
                test_set = singleSampleDataset(data_dict_dir,
                                               transform=transform,
                                               transform_output_imgs = transform_output_imgs,
                                               output_data_key = self.output_data_key,
                                               used_keys = self.used_keys,
                                               one_hot_for_segmentation = one_hot_for_segmentation)

            self.csv_min_max = test_set.csv_min_max.copy()
            test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=10)


        # Create predictions directory if it doesn't exist
        if predictions_dir and not os.path.exists(predictions_dir):
            os.mkdir(predictions_dir)

        # Evaluate the model
        model.train(False)
        epoch_vloss = 0.0
        epoch_vloss_in_original_scale = 0.0
        
        frame_counter = 0
        label_box_pos_orig_scale_list = list()
        predictions_box_pos_orig_scale_list = list()

        label_box_pos_list = list()
        predictions_box_pos_list = list()

        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = self.get_inputs_and_labels(vdata)
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                epoch_vloss += vloss

                # Handle predictions for segmentation or other tasks
                if one_hot_for_segmentation:
                    voutputs = one_hot_to_rgb_batch(voutputs)
                    vlabels = one_hot_to_rgb_batch(vlabels)
                    loop_range = len(vlabels)

                else:
                    loop_range = vlabels.shape[0]

                if "cam" in self.output_data_key:
                    vloss_in_original_scale = 0
                    for i in range(loop_range):
                        show_img(voutputs, vlabels, i, frame_counter, one_hot_for_segmentation, predictions_dir)
                        frame_counter += 1
                        

                else:
                    vloss_in_original_scale = self.loss_fn(self.denormalize(voutputs, self.output_data_key), self.denormalize(vlabels, self.output_data_key))

                    out_orig_scale = self.denormalize(voutputs, self.output_data_key)
                    labes_orig_scale = self.denormalize(vlabels, self.output_data_key)
                    label_box_pos_orig_scale_list.append(labes_orig_scale)
                    predictions_box_pos_orig_scale_list.append(out_orig_scale)

                    label_box_pos_list.append(vlabels)
                    predictions_box_pos_list.append(voutputs)


                epoch_vloss_in_original_scale += vloss_in_original_scale

                    
        avg_vloss = epoch_vloss / (i + 1)
        avg_vloss_n_original_scale = epoch_vloss_in_original_scale / (i + 1)
        print(f'LOSS test: {avg_vloss}, loss (in m): {avg_vloss_n_original_scale}')

        if "cam" not in self.output_data_key:
            label_tensor_orig_scale = torch.cat(label_box_pos_orig_scale_list, dim=0)
            out_tensor_orig_scale = torch.cat(predictions_box_pos_orig_scale_list, dim=0)

            final_tensor_orig_scale = torch.cat([label_tensor_orig_scale, out_tensor_orig_scale], dim=1)
            final_tensor_orig_scale = final_tensor_orig_scale.cpu()
            final_array_orig_scale = final_tensor_orig_scale.numpy()

            np.savetxt(predictions_dir +os.sep+ "output_orig_scale.csv", final_array_orig_scale, delimiter=",")


            label_tensor = torch.cat(label_box_pos_list, dim=0)
            out_tensor = torch.cat(predictions_box_pos_list, dim=0)

            final_tensor = torch.cat([label_tensor, out_tensor], dim=1)
            final_tensor = final_tensor.cpu()
            final_array = final_tensor.numpy()

            np.savetxt(predictions_dir +os.sep+ "output.csv", final_array, delimiter=",")

            
        return avg_vloss
        

def show_img(voutputs, vlabels, idx, frame_counter, one_hot_for_segmentation, predictions_dir):
    """
    Combines and saves a side-by-side comparison of predicted and ground truth images.

    Args:
        voutputs (torch.Tensor): Predicted output tensor.
        vlabels (torch.Tensor): Ground truth tensor.
        idx (int): Index of the image in the batch.
        frame_counter (int): Frame counter for naming the saved image.
        one_hot_for_segmentation (bool): Whether the output is one-hot encoded for segmentation.
        predictions_dir (str): Directory to save the combined image.

    Returns:
        None
    """

    # Convert the first tensor to a NumPy array and then to a PIL image
    if one_hot_for_segmentation:
        image1 = voutputs[idx]
    else:        
        tensor1 = voutputs[idx] * 255  # This creates a tensor with values in range [0, 255]
        tensor1 = tensor1.byte()  # Convert to byte format
        tensor1 = tensor1.cpu()
        np_array1 = tensor1.numpy()
        np_array1 = np.transpose(np_array1, (1, 2, 0))  # Transpose to match the shape (M, N, 3)
        image1 = im.fromarray(np_array1)

    # Convert the second tensor to a NumPy array and then to a PIL image
    if one_hot_for_segmentation:
        image2 = vlabels[idx]
    
    else:
        tensor2 = vlabels[idx] * 255  # This creates a tensor with values in range [0, 255]
        tensor2 = tensor2.byte()  # Convert to byte format
        tensor2 = tensor2.cpu()
        np_array2 = tensor2.numpy()
        np_array2 = np.transpose(np_array2, (1, 2, 0))  # Transpose to match the shape (M, N, 3)
        image2 = im.fromarray(np_array2)

    # Ensure both images have the same height
    if image1.size[1] != image2.size[1]:
        # Resize images to the same height if necessary
        common_height = min(image1.size[1], image2.size[1])
        image1 = image1.resize((image1.size[0], common_height))
        image2 = image2.resize((image2.size[0], common_height))

    # Concatenate the images horizontally
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    combined_image = im.new('RGB', (total_width, max_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # Display the combined image
    #combined_image.show()
    if predictions_dir:
        filename = predictions_dir +os.sep+ f"{frame_counter}.png"
    else:
        filename = f"flow_comparison{frame_counter}.png"
    combined_image.save(filename)

def one_hot_to_rgb_batch(one_hot_tensor):
    """
    Converts a batch of one-hot encoded tensors to RGB images.

    Args:
        one_hot_tensor (torch.Tensor): Batch of one-hot encoded tensors.

    Returns:
        List[PIL.Image.Image]: List of RGB images.
    """
    to_return = list()
    for img in one_hot_tensor:
        rgb_img = one_hot_to_rgb(img.cpu())
        rgb_img = im.fromarray(rgb_img)
        to_return.append(rgb_img)

    return to_return


def one_hot_to_rgb(one_hot_tensor):
    """
    Converts a one-hot encoded tensor to an RGB image.

    Args:
        one_hot_tensor (torch.Tensor): One-hot encoded tensor.

    Returns:
        np.ndarray: RGB image as a NumPy array.
    """

    colors = [(25, 255, 140),   # Class 0
              (255, 255, 25),   # Class 1
              (25, 25, 255),    # Class 2
              (0, 0, 0),        # Class 3
              (140, 25, 140)    # Class 4
            ]
    
    # Get the height and width from the one-hot tensor
    num_classes, height, width = one_hot_tensor.shape
    
    # Create an empty RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert one-hot tensor back to class indices
    class_indices = torch.argmax(one_hot_tensor, dim=0).numpy()
    
    # Map class indices back to RGB colors
    for class_index, color in enumerate(colors):
        mask = (class_indices == class_index)
        rgb_image[mask] = color
    
    return rgb_image


def plot_training(train_loss: np.ndarray, valid_loss: np.ndarray, best_valid_loss:float, best_valid_loss_epoch_num: int, best_valid_loss_epoch_num_in_m:float,  save_path: str = None) -> None:
    """
    Plots training and validation losses over epochs and saves the plot.

    Args:
        train_loss (np.ndarray): Array of training losses.
        valid_loss (np.ndarray): Array of validation losses.
        best_valid_loss (float): Best validation loss.
        best_valid_loss_epoch_num (int): Epoch number with the best validation loss.
        best_valid_loss_epoch_num_in_m (float): Best validation loss in original scale.
        save_path (str, optional): Directory to save the plot. If None, the plot is displayed.

    Returns:
        None
    """
   
    
    epochs_arr = np.arange(train_loss.size)
    # Create line plots for y1_values and y2_values
    plt.plot(epochs_arr, train_loss, label='train_loss', marker='o')
    plt.plot(epochs_arr, valid_loss, label='valid_loss', marker='o')

    # Add labels and a title
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.ylim(0, 0.1)
    #plt.title('loss over epochs')

    # Add a legend
    plt.legend()
    plt.grid(True)

    if save_path:
       graph_path = os.path.join(save_path, "training_over_epochs.png")
       plt.savefig(graph_path)
       plt.close()

       #print(f"valid_loss sh = {valid_loss.shape}")
       #print(f"train_loss sh = {train_loss.shape}")
       losses_array = np.column_stack((train_loss, valid_loss))
       #print(f"losses_array sh = {losses_array.shape}")
       np.savetxt(os.path.join(save_path, "losses.csv"), losses_array, delimiter=",")

    else:
        plt.show()

    with open(os.path.join(save_path, "best_valid_loss.txt"), "w") as file:
        file.write(f"best validation loss: {best_valid_loss}" "\n")
        file.write(f"best validation loss: (in m): {best_valid_loss_epoch_num_in_m}" "\n")
        file.write(f"best validation epoch num: {best_valid_loss_epoch_num}" "\n")
        

class twoCamsModelClass(ModelClass):
    """
    A specialized model class for handling data from two cameras.
    Inherits from the base ModelClass.
    """

    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
        """
        Initializes the model with a custom architecture for two-camera input.

        Args:
            neueons_in_hidden_layer (int): Number of neurons in the hidden layer.
            dropout (float): Dropout rate for the fully connected layer.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            None
        """
        
        num_of_resnets = 2  # Number of ResNet backbones
        self.model = CombinedResNet18(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                      dropout = dropout,
                                      size_of_input = num_of_resnets*512,
                                      size_of_output = 3, 
                                      num_of_resnets= num_of_resnets)

        # Define optimizer and loss function
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        
        # Move the model to the appropriate device
        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        """
        Extracts inputs and labels from the dataset for two-camera input.

        Args:
            data (dict): Dictionary containing input data and labels.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Inputs from two cameras and corresponding labels.
        """
        return ((data["cam0_rgb"].to(self.device), data["cam1_rgb"].to(self.device)), data[self.output_data_key][:,  : self.size_of_output].to(self.device))
    


class multiModalClass(ModelClass):
    """
    A specialized model class for handling multimodal data.
    Inherits from the base ModelClass.
    """

    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1, input_data_keys = None, path_to_load_model = None, transfer = False):
        """
        Initializes the model with a custom architecture for multimodal input.

        Args:
            neueons_in_hidden_layer (int): Number of neurons in the hidden layer.
            dropout (float): Dropout rate for the fully connected layer.
            learning_rate (float): Learning rate for the optimizer.
            input_data_keys (list): List of keys for input data.
            path_to_load_model (str, optional): Path to load a pre-trained model. Defaults to None.
            transfer (bool): Whether to transfer weights from a pre-trained model. Defaults to False.

        Returns:
            None
        """
        self.input_data_keys = input_data_keys

        # Determine the number of camera inputs and non-camera data sizes
        num_of_cam_data = 0
        size_for_non_cam_data = list()
        if self.sequential_data:
            shape_to_check = 3
            dim_to_take = 2
            if "fusing_before_RNN" not in self.kwargs:
                self.kwargs['fusing_before_RNN'] = False
        else:
            shape_to_check = 2
            dim_to_take = 1
        
        for i, data in enumerate(self.training_loader):
            read_data, _= self.get_inputs_and_labels(data)
            for data_in in read_data:
                
                if len(data_in.shape) > shape_to_check:
                   num_of_cam_data  += 1 
                else:
                    size_for_non_cam_data.append(data_in.shape[dim_to_take])
            break

        # Define loss function
        self.loss_fn = torch.nn.MSELoss()

        if 'useResnet' not in self.kwargs:
            self.kwargs['useResnet'] = False
        if 'usePretrainedResnet' not in self.kwargs:
            self.kwargs['usePretrainedResnet'] = True
        if 'cam' in self.output_data_key:
            self.kwargs['generateImage'] = True
        else:
            self.kwargs['generateImage'] = False

        if 'do_segmentation' not in self.kwargs:
            self.kwargs['do_segmentation'] = False
            
        if self.kwargs['do_segmentation']:
            self.loss_fn = torch.nn.CrossEntropyLoss()
            #self.loss_fn = torch.nn.MSELoss()

        
        # Load pre-trained model if specified
        if path_to_load_model and not transfer:
            self.load_model(path_to_load_model)
            beta1 = 0.5
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))
            return
        

        # Initialize the model based on the data type
        if self.sequential_data and self.kwargs['fusing_before_RNN']:
            self.model = sequentialModelFusingBeforeRNN(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                          dropout = dropout,
                                          size_of_output = 3,
                                          num_of_resnets= num_of_cam_data,
                                          linear_inputs_sizes = size_for_non_cam_data,
                                          device= self.device,
                                          useResnet = self.kwargs['useResnet'],
                                          usePretrainedResnet = self.kwargs['usePretrainedResnet'],
                                          generateImage = self.kwargs['generateImage'],
                                          do_segmentation = self.kwargs["do_segmentation"]
                                          )
            
        elif self.sequential_data: 
            self.model = sequentialModel(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                          dropout = dropout,
                                          size_of_output = 3,
                                          num_of_resnets= num_of_cam_data,
                                          linear_inputs_sizes = size_for_non_cam_data,
                                          device= self.device,
                                          useResnet = self.kwargs['useResnet'],
                                          usePretrainedResnet = self.kwargs['usePretrainedResnet'],
                                          generateImage = self.kwargs['generateImage'],
                                          do_segmentation = self.kwargs["do_segmentation"]
                                          )
            

        else:
            self.model = multimodalMoldel(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                          dropout = dropout,
                                          size_of_output = 3,
                                          num_of_resnets= num_of_cam_data,
                                          linear_inputs_sizes = size_for_non_cam_data,
                                          useResnet = self.kwargs['useResnet'],
                                          usePretrainedResnet = self.kwargs['usePretrainedResnet'],
                                          generateImage = self.kwargs['generateImage'],
                                          do_segmentation = self.kwargs["do_segmentation"]
                                          )

        # Define optimizer
        if 'cam' in self.output_data_key:
            beta1 = 0.5
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))
        else:
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)


        # Move the model to the appropriate device
        self.model.to(self.device)

        # Transfer weights if specified
        if path_to_load_model and transfer:
            print("USING WEIGHTS OF OLD MODEL...")
            self.transfer_model(path_to_load_model)
            print("WEIGHTS LOADED :)")


    def get_inputs_and_labels(self, data):
        """
        Extracts inputs and labels from the dataset for multimodal input.

        Args:
            data (dict): Dictionary containing input data and labels.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: List of input tensors and corresponding labels.
        """
        return ([data[data_key].to(self.device) for data_key in self.input_data_keys], data[self.output_data_key][:, : self.size_of_output].to(self.device))
    

    def transfer_model(self, path_to_load_model):
        """
        Transfers weights from a pre-trained model to the current model.

        Args:
            path_to_load_model (str): Path to the pre-trained model file.

        Returns:
            None
        """
        old_model = torch.load(path_to_load_model, map_location=self.device)
        old_model.to(self.device)


        # Extract the state dict from the loaded model
        pretrained_dict = old_model.state_dict()

        # Copy weights except for `fc`
        new_model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
        new_model_dict.update(pretrained_dict)
        self.model.load_state_dict(new_model_dict)

    
    def freeze_encoders(self):
        """
        Freezes all layers except the last fully connected layer (`fc`).

        Returns:
            None
        """
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_encoders(self):
        """
        Unfreezes all layers in the model.

        Returns:
            None
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = True