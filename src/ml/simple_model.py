import torch
from torch.optim import SGD, Adam, Adagrad, RMSprop, Adadelta, AdamW, SparseAdam, Adamax, LBFGS
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import Tuple, List, Dict
from copy import deepcopy

import os
import sys
#module_path = os.path.abspath(os.path.join("../data"))
module_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(module_path, f"..")
if module_path not in sys.path:
    sys.path.append(module_path)
from data.data_read import singleSampleDataset, test_single_samples


class customFullyConnectedLayer(torch.nn.Module):
    """
    This class is used to create a fully conected
    suited for the problem of classifiation a ball into one of 15 classes 
    """
    
    def __init__(self, neueons_in_hidden_layer: int = 50, dropout:float = 0.4, size_of_input = 512, size_of_output = 1):
        """
        Initializes the fully connected layer.

        Args:
            neurons_in_hidden_layer (int, optional): The number of neurons in the hidden layer. Defaults to 50.
            dropout (float, optional): The dropout rate for regularization in the hidden layer. Defaults to 0.4.
        """
        
        super(customFullyConnectedLayer, self).__init__()
        
        neurons_in = size_of_input #number of outpus from the resnet18 to the fully connected layer
        neurons_out = size_of_output #number of image classes
        
        self.lin1 = torch.nn.Linear(neurons_in, 512)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.lin2 = torch.nn.Linear(512, 128)
        self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.lin_out = torch.nn.Linear(128, neurons_out)
        #self.softmax_out = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Defines the forward pass of the fully connected layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the fully connected layer.
        """
        
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.lin_out(x)
        #x = self.softmax_out(x)
        
        return x
    

class customCNN(torch.nn.Module):
    """
    This class is used to create a fully conected
    suited for the problem of classifiation a ball into one of 15 classes 
    """
    
    def __init__(self):
        """
        Initializes the fully connected layer.

        Args:
            neurons_in_hidden_layer (int, optional): The number of neurons in the hidden layer. Defaults to 50.
            dropout (float, optional): The dropout rate for regularization in the hidden layer. Defaults to 0.4.
        """
        
        super(customCNN, self).__init__()
        
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5))
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5))
        self.relu3 = torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 5))
        self.relu4 = torch.nn.ReLU()
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5))
        self.relu5 = torch.nn.ReLU()
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        
    def forward(self, x):
        
        #x = torch.flatten(x, 1)
        #print(f"combined.shape = {x.shape}")
        #exit()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        

        x = torch.flatten(x, 1)
        #print(f"combined.shape = {x.shape}")
        #exit()
        return x
    


class CombinedResNet18(torch.nn.Module):
    def __init__(self, neueons_in_hidden_layer, dropout, size_of_input, size_of_output, num_of_resnets = 2):
        super(CombinedResNet18, self).__init__()
        
        self.num_of_resnets = num_of_resnets
        self.resnets_list = torch.nn.ModuleList()
        for _ in range(num_of_resnets):
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.fc = torch.nn.Identity()
            self.resnets_list.append(resnet)
        
        
        # Define a new fully connected layer
        self.fc = customFullyConnectedLayer(size_of_input = 512 * num_of_resnets, size_of_output = 3)  # Adjust the output size as needed
        


    def forward(self, x):
        # Pass through the two ResNet18 models
        out_combined = [self.resnets_list[i](x[i]) for i in range(self.num_of_resnets)]

        # Concatenate the outputs
        combined = torch.cat(out_combined, dim=1)
        
        # Pass through the fully connected layer
        out = self.fc(combined)
        
        return out
    

class multimodalMoldel(torch.nn.Module):
    def __init__(self, neueons_in_hidden_layer, dropout, size_of_output, num_of_resnets, linear_inputs_sizes):
        super(multimodalMoldel, self).__init__()
        
        self.num_of_resnets = num_of_resnets
        self.linear_inputs_sizes = linear_inputs_sizes
        
        self.resnets_list = torch.nn.ModuleList()
        for _ in range(self.num_of_resnets):
            #resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            #for param in resnet.parameters():
            #    param.requires_grad = False
            #resnet.fc = torch.nn.Identity()
            resnet = customCNN()
            self.resnets_list.append(resnet)
        
        
        # Define a new fully connected layer
        self.fc = customFullyConnectedLayer(size_of_input = 144 * self.num_of_resnets + sum(linear_inputs_sizes), size_of_output = size_of_output)
        

    def forward(self, x):
        # Pass through the two ResNet18 models
        out_combined = list()
        out_combined = [self.resnets_list[i](x[i]) for i in range(self.num_of_resnets)]
        
        for i in range(len(self.linear_inputs_sizes)):
            out_combined.append(x[i+self.num_of_resnets])

        # Concatenate the outputs
        combined = torch.cat(out_combined, dim=1)
        # Pass through the fully connected layer
        out = self.fc(combined)
        
        return out

    
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
                 input_data_keys:str = None) -> None:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.epochs_num = epochs_num
        
        if path_to_load_model:
            self.training_loader = None
            self.validation_loader = None
            self.test_loader = None
            self.classes = None
            self.optimizer = None
            self.loss_fn = None
            self.model = torch.load(path_to_load_model, map_location=self.device)
            self.model.to(self.device)
            return
      

        NUM_OF_CAMERAS = 5
        camera_types = ["depth", "rgb", "segmented", "flow"]
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

        #transform = transforms.Compose([
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()
        #])

        transform = transforms.Compose([
                                    transforms.Resize(256),         # Resize the image to 256x256 pixels
                                    transforms.CenterCrop(224),     # Crop the center to 224x224 pixels
                                    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
                                    transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],  # Normalize the image based on ImageNet statistics
                                                        std=[0.229, 0.224, 0.225]
                                                        )
                                    ])

        
        torch.manual_seed(0) #added maual seed to make sure the random split is the same every time

        full_training_set = singleSampleDataset(data_dict_dir, transform=transform)
        self.csv_min_max = full_training_set.csv_min_max.copy()

        full_training_set.remove_unused_keys(input_data_keys + ["boxes_pos0"])
        train_size = int(train_val_split * len(full_training_set))
        val_size = len(full_training_set) - train_size
        training_set, validation_set = random_split(full_training_set, [train_size, val_size])

        self.training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=7)
        self.validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=3)



    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.size_of_output = 3
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
        
        return data["cam0_rgb"].to(self.device), data["boxes_pos0"][:,  : self.size_of_output].to(self.device)


    def train_epoch(self, epoch_index: int) -> float:
        """
        Performs training for one epoch.
    
        Args:
            epoch_index (int): The index of the current epoch.
    
        Returns:
            float: The average loss value for the epoch.
        """

        running_loss = 0.

        for i, data in tqdm(enumerate(self.training_loader), total=len(self.training_loader)):
            # Every data instance is an input + label pair
            inputs, labels = self.get_inputs_and_labels(data)

            #inputs = inputs.to(self.device)
            #labels = labels.to(self.device)
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self.model(inputs)
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
            # Gather data and report
            running_loss += loss.item()
            
        return running_loss / (i+1)
   
    
    def denormalize(self, data, key):
        min_val, max_val = self.csv_min_max[key]
        if not isinstance(min_val, torch.Tensor):
            min_val = torch.tensor(min_val[ : self.size_of_output], device=self.device, dtype=data.dtype)
        if not isinstance(max_val, torch.Tensor):
            max_val = torch.tensor(max_val[ : self.size_of_output], device=self.device, dtype=data.dtype)
        return data * (max_val - min_val) + min_val


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
    
                    vloss_in_original_scale = self.loss_fn(self.denormalize(voutputs, "boxes_pos0"), self.denormalize(vlabels, "boxes_pos0"))
                    epoch_vloss_in_original_scale += vloss_in_original_scale

            avg_vloss = epoch_vloss / (i + 1)
            avg_vloss_n_original_scale = epoch_vloss_in_original_scale / (i + 1)
            print(f'LOSS train: {avg_loss} valid: {avg_vloss}, valid (in m): {avg_vloss_n_original_scale}')
            train_loss[current_epoch] = avg_loss
            valid_loss[current_epoch] = avg_vloss   
            
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                best_epoch_model = deepcopy(self.model)
                best_epochs = current_epoch

        """
        total_preds, total_accuracy, F1, total_pred_per_class, correct_pred_per_class, predictions_for_each_label = self.test_model(model = best_epoch_model,
                                                                                                                                    test_loader = self.validation_loader)

        print(f'Accuracy of the network on the {total_preds} test images: {total_accuracy} %')
        print(f'F1 Score: {F1}')
        print(f'best val loss: {best_vloss}')
        print(f'number of epochs (best): {best_epochs}\n')
        for classname, correct_count in correct_pred_per_class.items():
            accuracy = 100 * float(correct_count) / total_pred_per_class[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')    

        if self.save_path:
            with open(os.path.join(self.save_path, "accuracy_and_F1.txt"), 'w') as f:
                f.write(f'Accuracy of the network on the {total_preds} test images: {total_accuracy} %\n')
                f.write(f'F1 Score: {F1}\n')
                f.write(f'best val loss: {best_vloss}\n')
                f.write(f'number of epochs (best): {best_epochs}\n')
                for classname, correct_count in correct_pred_per_class.items():
                    accuracy = 100 * float(correct_count) / total_pred_per_class[classname]
                    f.write(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %\n')
            
            if save_pt_model:
                torch.save(best_epoch_model, os.path.join(self.save_path, "resnet18_model.pt"))
            
        """
        
        plot_training(train_loss, valid_loss, best_vloss, best_epochs, self.save_path)    

        return train_loss, valid_loss


    def test_model(self,
                   model: torch.nn.Module = None,
                   classes: List[str] = None,
                   test_loader: DataLoader = None) -> Tuple[int,
                                                            int,
                                                            float,
                                                            Dict[str, int],
                                                            Dict[str, int],
                                                            Dict[str, Dict[str, int]]]:
        """
        Tests the performance of the model on the test_loader dataset.

        Args:
            model (torch.nn.Module, optional): The PyTorch model to be evaluated. Defaults to None, in which case the stored model is used.
            classes (List[str], optional): A list of class labels. Defaults to None, in which case the stored class labels are used.
            test_loader (DataLoader, optional): The data loader for the test/evaluation dataset. Defaults to None, in which case the stored test loader is used.

        Returns:
            Tuple[int, int, float, Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]]]:
                the returned tuple contains the following:
                - total_preds (int): The total number of predictions made.
                - total_accuracy (int): The accuracy of the model on the test dataset.
                - F1 (float): The F1 score of the model on the test dataset.
                - total_pred_per_class (Dict[str, int]): A dictionary containing the total number of predictions per class.
                - correct_pred_per_class (Dict[str, int]): A dictionary containing the number of correct predictions per class.
                - predictions_for_each_label (Dict[str, Dict[str, int]]): A nested dictionary that is a confusion matrix of predicitons made.
        """               

        if model is None:
            model = self.model
            
        if classes is None:
            classes = self.classes
            
        if test_loader is None:
            test_loader = self.test_loader


        model.train(False)

        correct_pred_per_class = {classname: 0 for classname in classes}
        total_pred_per_class = {classname: 0 for classname in classes}
        predictions_for_each_label = {real_classname: {predicted_classname: 0 for predicted_classname in classes} for real_classname in classes}
        correct_total = 0
        total_preds = 0
        all_labels = []
        all_predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # again no gradients needed
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                total_preds += labels.size(0)
                correct_total += (predictions == labels).sum().item()
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    all_labels.append(label.item())
                    all_predictions.append(prediction.item())
                    if label == prediction:
                        correct_pred_per_class[classes[label]] += 1
                    total_pred_per_class[classes[label]] += 1
                    predictions_for_each_label[classes[label]][classes[prediction]] += 1    

        total_accuracy = 1.0 * correct_total / total_preds
        F1 = f1_score(all_labels, all_predictions, average='weighted')

        return total_preds, total_accuracy, F1, total_pred_per_class, correct_pred_per_class, predictions_for_each_label


def plot_training(train_loss: np.ndarray, valid_loss: np.ndarray, best_valid_loss:float, best_valid_loss_epoch_num: int,  save_path: str = None) -> None:
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
    plt.ylim(0, 300)
    #plt.title('loss over epochs')

    # Add a legend
    plt.legend()
    plt.grid(True)

    if save_path:
       graph_path = os.path.join(save_path, "training_over_epochs.png")
       plt.savefig(graph_path)
       plt.close()
    else:
        plt.show()

    with open(os.path.join(save_path, "best_valid_loss.txt"), "w") as file:
        file.write(f"best validation loss: {best_valid_loss}" "\n")
        file.write(f"best validation epoch num: {best_valid_loss_epoch_num}" "\n")
        

class twoCamsModelClass(ModelClass):


    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1):
 
        self.size_of_output = 3
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
        
        return ((data["cam0_rgb"].to(self.device), data["cam1_rgb"].to(self.device)), data["boxes_pos0"][:,  : self.size_of_output].to(self.device))
    


class multiModalClass(ModelClass):


    def init_model(self, neueons_in_hidden_layer = 50, dropout = 0.4, learning_rate = 0.1, input_data_keys = None):
 
        self.input_data_keys = input_data_keys
        self.size_of_output = 3

        num_of_cam_data = 0
        size_for_non_cam_data = list()

        for i, data in enumerate(self.training_loader):
            read_data, _= self.get_inputs_and_labels(data)
            for data_in in read_data:
                
                if len(data_in.shape) > 2:
                   num_of_cam_data  += 1 
                else:
                    size_for_non_cam_data.append(data_in.shape[1])
            break


        self.model = multimodalMoldel(neueons_in_hidden_layer = neueons_in_hidden_layer,
                                      dropout = dropout,
                                      size_of_output = 3,
                                      num_of_resnets= num_of_cam_data,
                                      linear_inputs_sizes = size_for_non_cam_data
                                      )

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.model.to(self.device)


    def get_inputs_and_labels(self, data):
        
        return ([data[data_key].to(self.device) for data_key in self.input_data_keys], data["boxes_pos0"][:,  : self.size_of_output].to(self.device))
    