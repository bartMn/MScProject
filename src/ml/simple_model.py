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
    
    def __init__(self, neueons_in_hidden_layer: int = 50, dropout:float = 0.4):
        """
        Initializes the fully connected layer.

        Args:
            neurons_in_hidden_layer (int, optional): The number of neurons in the hidden layer. Defaults to 50.
            dropout (float, optional): The dropout rate for regularization in the hidden layer. Defaults to 0.4.
        """
        
        super(customFullyConnectedLayer, self).__init__()
        
        neurons_in = 512 #number of outpus from the resnet18 to the fully connected layer
        neurons_out = 7 #number of image classes
        
        self.lin1 = torch.nn.Linear(neurons_in, neueons_in_hidden_layer)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.lin2 = torch.nn.Linear(neueons_in_hidden_layer, neurons_out)
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
        #x = self.softmax_out(x)
        
        return x
    
class ModelClass():
    """
    This class is used to create, train and evaluate a resnet18 model
    """

    def __init__(self,
                 data_path:str= None,
                 train_val_split:float=0.9,
                 batch_size:int=32,
                 epochs_num:int=20,
                 neueons_in_hidden_layer = 50,
                 learning_rate:float = 0.1,
                 dropout:float = 0.4,
                 save_path:str=None,
                 path_to_load_model:str=None) -> None:
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
        
        #the rest is executed only when path_to_load_model is not given
        
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = customFullyConnectedLayer(neueons_in_hidden_layer = neueons_in_hidden_layer, dropout= dropout)

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

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        
        torch.manual_seed(0) #added maual seed to make sure the random split is the same every time

        full_training_set = singleSampleDataset(data_dict_dir, transform=transform)
        train_size = int(train_val_split * len(full_training_set))
        val_size = len(full_training_set) - train_size
        training_set, validation_set = random_split(full_training_set, [train_size, val_size])

        self.training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=7)
        self.validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=3)

    

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
            inputs, labels = data["cam0_rgb"], data["boxes_pos0"]

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
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
            for i, vdata in enumerate(self.validation_loader):
                vinputs, vlabels = vdata["cam0_rgb"], vdata["boxes_pos0"]
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                epoch_vloss += vloss  

            avg_vloss = epoch_vloss / (i + 1)
            print(f'LOSS train: {avg_loss} valid: {avg_vloss}')
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
        
        plot_training(train_loss, valid_loss, self.save_path)    

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


def plot_training(train_loss: np.ndarray, valid_loss: np.ndarray, save_path: str = None) -> None:
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
