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
from data_processing.data_read import singleSampleDataset, sequentialSampleDataset,  test_single_samples
from ml.models import *

torch.backends.cudnn.enabled = False



    
class ModelClass():
    """
    This class is used to create, train and evaluate a resnet18 model
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
        Initialize the ResNet model for image classification.

        Args:
            data_path (str, optional): The path to the data directory. Defaults to None.
            train_val_split (float, optional): Ratio of the size of training set to the sum of sizes of training and validation sets. Defaults to 0.9.
            batch_size (int, optional): The batch size. Defaults to 20.
            epochs_num (int, optional): The number of epochs for training. Defaults to 20.
            neueons_in_hidden_layer (int, optional): The number of neurons in the hidden layer of the custom fully connected layer. Defaults to 50.
            learning_rate (float, optional): The learning rate. Defaults to 0.1.
            dropout (float, optional): The dropout rate for regularization. Defaults to 0.4.
            save_path (str, optional): The path to save the trained model. Defaults to None.
            path_to_load_model (str, optional): The path to the pre-trained model to load. Defaults to None.
        """
        self.kwargs = kwargs
        self.output_data_key = output_data_key if output_data_key else "boxes_pos0"
        if "seg" in self.output_data_key:
            self.size_of_output = 5
        else:
            self.size_of_output= 3

        self.sequential_data = sequential_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        if self.save_path and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.epochs_num = epochs_num
              

        NUM_OF_CAMERAS = 5
        camera_types = ["depth", "rgb", "segmented", "flow"]
        data_root = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(data_root, "..", "..", "..", "recorded_data_isaac_lab")
        
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
        
        torch.manual_seed(0) #added maual seed to make sure the random split is the same every time

        self.used_keys = input_data_keys + [self.output_data_key]
        
        if self.sequential_data:
            self.sequence_length = sequence_length
            full_training_set = sequentialSampleDataset(data_dict_dir,
                                                        transform=transform,
                                                        sequence_length=sequence_length,
                                                        transform_output_imgs = transform_output_imgs,
                                                        output_data_key = self.output_data_key,
                                                        used_keys = self.used_keys,
                                                        one_hot_for_segmentation = one_hot_for_segmentation)
        else:
            full_training_set = singleSampleDataset(data_dict_dir,
                                                    transform=transform,
                                                    transform_output_imgs = transform_output_imgs,
                                                    output_data_key =output_data_key,
                                                    used_keys = self.used_keys,
                                                    one_hot_for_segmentation = one_hot_for_segmentation)
            
        self.csv_min_max = full_training_set.csv_min_max.copy()

        #full_training_set.remove_unused_keys(input_data_keys + [self.output_data_key])
        train_size = int(train_val_split * len(full_training_set))
        val_size = len(full_training_set) - train_size
        training_set, validation_set = random_split(full_training_set, [train_size, val_size])

        self.training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=7)
        self.validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=3)



    def load_model(self, path_to_load_model):
        self.model = torch.load(path_to_load_model, map_location=self.device)
        self.model.to(self.device)
            #return

    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        #self.size_of_output = 3
        self.model.fc = customFullyConnectedLayer(neueons_in_hidden_layer = neueons_in_hidden_layer, dropout= dropout, size_of_output= self.size_of_output)

        custom_fc_params = list(self.model.fc.parameters()) 

        #freezing all parmeters
        for param in self.model.parameters():
            param.requires_grad = False 

        #unfreezing parmeters in the fully connnected layer
        for param in custom_fc_params:
            param.requires_grad = True  

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        
        return data["cam0_rgb"].to(self.device), data[self.output_data_key][:,  : self.size_of_output].to(self.device)


    def train_epoch(self, epoch_index: int) -> float:
        """
        Performs training for one epoch.
    
        Args:
            epoch_index (int): The index of the current epoch.
    
        Returns:
            float: The average loss value for the epoch.
        """
        running_loss = 0.0

        for i, data in tqdm(enumerate(self.training_loader), total=len(self.training_loader)):
            # Every data instance is an input + label pair
            inputs, labels = self.get_inputs_and_labels(data)

            #print(f"max = {labels.max()}")
            #inputs = inputs.to(self.device)
            #labels = labels.to(self.device)
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self.model(inputs)
            # Compute the loss and its gradients
            #print(f"outputs.shape = {outputs.shape}")
            #print(f"labels.shape = {labels.shape}")
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            
        return running_loss / (i+1)
   
    
    def denormalize(self, data, key, epsilon=1e-8):
        min_val, max_val = self.csv_min_max[key]
        if not isinstance(min_val, torch.Tensor):
            min_val = torch.tensor(min_val[ : self.size_of_output], device=self.device, dtype=data.dtype)
        if not isinstance(max_val, torch.Tensor):
            max_val = torch.tensor(max_val[ : self.size_of_output], device=self.device, dtype=data.dtype)

        numerator = max_val - min_val
        numerator = torch.where(numerator == 0, epsilon, numerator)
        
        return data * numerator + min_val


    def train_model(self, epochs: int = None, save_pt_model:bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trains the model for a specified number of epochs and optionally saves the trained model.

        Args:
            epochs (int, optional): The number of epochs to train the model. Defaults to None.
            save_pt_model (bool, optional): Flag indicating whether to save the PyTorch model after training. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                the returned tuple contains the following:
                - train_loss (np.ndarray): An array containing the training loss values over epochs.
                - valid_loss (np.ndarray): An array containing the validation loss values over epochs.
        """
        
        if epochs is None:
            epochs = self.epochs_num
            
        train_loss = np.zeros(epochs)
        valid_loss = np.zeros(epochs)
        best_vloss = float('inf')
        best_avg_vloss_n_original_scale = float('inf')
        best_epoch_model = deepcopy(self.model)
        best_epochs = 0
        
        for current_epoch in range(epochs):
            print(f'EPOCH {current_epoch}:')    

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_epoch(current_epoch)   
            # We don't need gradients on to do reporting
            self.model.train(False)  

            epoch_vloss = 0.0
            epoch_vloss_in_original_scale = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = self.get_inputs_and_labels(vdata)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    epoch_vloss += vloss

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
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_avg_vloss_n_original_scale = avg_vloss_n_original_scale
                best_epoch_model = deepcopy(self.model)
                best_epochs = current_epoch

        if  self.save_path and save_pt_model:
            torch.save(best_epoch_model, os.path.join(self.save_path, "model.pt"))
        
        plot_training(train_loss, valid_loss, best_vloss, best_epochs, best_avg_vloss_n_original_scale, self.save_path)    

        return train_loss, valid_loss


    def test_model(self,
                   model: torch.nn.Module = None,
                   use_test_set:bool = False):
                 

        if model is None:
            model = self.model
            
        if use_test_set is False:
            test_loader = self.validation_loader

        else:

            ###################################################################################################
            NUM_OF_CAMERAS = 5
            camera_types = ["depth", "rgb", "segmented", "flow"]
            data_root = "/home/bart/project/test_set"

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
            ####################################################################################################
            if self.sequential_data:
                test_set = sequentialSampleDataset(data_dict_dir,
                                                            transform=transform,
                                                            sequence_length= self.sequence_length,
                                                            transform_output_imgs = transform_output_imgs,
                                                            output_data_key = self.output_data_key,
                                                            used_keys = self.used_keys,
                                                            one_hot_for_segmentation = one_hot_for_segmentation)
            else:
                test_set = singleSampleDataset(data_dict_dir,
                                                        transform=transform,
                                                        transform_output_imgs = transform_output_imgs,
                                                        output_data_key = self.output_data_key,
                                                        used_keys = self.used_keys,
                                                        one_hot_for_segmentation = one_hot_for_segmentation)

            self.csv_min_max = test_set.csv_min_max.copy()

            #full_training_set.remove_unused_keys(input_data_keys + [self.output_data_key])
            #train_size = int(train_val_split * len(full_training_set))
            #val_size = len(full_training_set) - train_size
            #training_set, validation_set = random_split(full_training_set, [train_size, val_size])

            test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=3)



        model.train(False)
        epoch_vloss = 0.0
        epoch_vloss_in_original_scale = 0.0
        
        frame_counter = 0
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = self.get_inputs_and_labels(vdata)
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                epoch_vloss += vloss

                if "cam" in self.output_data_key:
                    for i in range(vlabels.shape[0]):
                        show_img(voutputs, vlabels, i, frame_counter)
                        frame_counter += 1

                if "cam" not in self.output_data_key:
                    vloss_in_original_scale = self.loss_fn(self.denormalize(voutputs, self.output_data_key), self.denormalize(vlabels, self.output_data_key))
                else:
                    vloss_in_original_scale = 0
                epoch_vloss_in_original_scale += vloss_in_original_scale

                    
        avg_vloss = epoch_vloss / (i + 1)
        avg_vloss_n_original_scale = epoch_vloss_in_original_scale / (i + 1)
        print(f'LOSS test: {avg_vloss}, loss (in m): {avg_vloss_n_original_scale}')
        

def show_img(voutputs, vlabels, idx, frame_counter):

    # Convert the first tensor to a NumPy array and then to a PIL image
    tensor1 = voutputs[idx] * 255  # This creates a tensor with values in range [0, 255]
    tensor1 = tensor1.byte()  # Convert to byte format
    tensor1 = tensor1.cpu()
    np_array1 = tensor1.numpy()
    np_array1 = np.transpose(np_array1, (1, 2, 0))  # Transpose to match the shape (M, N, 3)
    image1 = im.fromarray(np_array1)

    # Convert the second tensor to a NumPy array and then to a PIL image
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
    filename = f"/home/bart/project/test_set/flow_predicitons{os.sep}flow_comparison{frame_counter}.png"
    combined_image.save(filename)


def plot_training(train_loss: np.ndarray, valid_loss: np.ndarray, best_valid_loss:float, best_valid_loss_epoch_num: int, best_valid_loss_epoch_num_in_m:float,  save_path: str = None) -> None:
    """
    Plot the training and validation loss over epochs.

    Args:
        train_loss (np.ndarray): An array containing the training loss values over epochs.
        valid_loss (np.ndarray): An array containing the validation loss values over epochs.
        save_path (str, optional): The path to save the plot as an image file. Defaults to None.
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


    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
 
        #self.size_of_output = 3
        num_of_resnets = 2
        self.model = CombinedResNet18(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                      dropout = dropout,
                                      size_of_input = num_of_resnets*512,
                                      size_of_output = 3, 
                                      num_of_resnets= num_of_resnets)

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        
        return ((data["cam0_rgb"].to(self.device), data["cam1_rgb"].to(self.device)), data[self.output_data_key][:,  : self.size_of_output].to(self.device))
    


class multiModalClass(ModelClass):


    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1, input_data_keys = None, path_to_load_model = None):
 
        self.input_data_keys = input_data_keys
        #self.size_of_output = 3

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

        
        if path_to_load_model:
            self.load_model(path_to_load_model)
            #beta1 = 0.5
            #self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))
            return


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
            #beta1 = 0.5
            #self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))
            
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
            #beta1 = 0.5
            #self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))

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
            #self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)

        
        if 'cam' in self.output_data_key:
            beta1 = 0.5
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate, betas=(beta1, 0.999))
        else:
            self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)


        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        #print(f"data[self.output_data_key].shape = {data[self.output_data_key].shape}")
        #print(f"data[self.output_data_key][:,  : self.size_of_output].shape = {data[self.output_data_key][:,  : self.size_of_output].shape}")
        #exit(0)
        #print(f"data[self.output_data_key].shape = {data[self.output_data_key].shape}")
        #print(f"data[self.output_data_key] = {data[self.output_data_key]}")
        #exit(0)
        return ([data[data_key].to(self.device) for data_key in self.input_data_keys], data[self.output_data_key][:, : self.size_of_output].to(self.device))