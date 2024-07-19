import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F


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
    def __init__(self, neueons_in_hidden_layer, dropout, size_of_output, num_of_resnets, linear_inputs_sizes, **kwargs):
        super(multimodalMoldel, self).__init__()
        
        self.num_of_resnets = num_of_resnets
        self.linear_inputs_sizes = linear_inputs_sizes

        if kwargs["useResnet"]:
            self.cnn_encoding_size = 512
        else:
            self.cnn_encoding_size = 144


        self.resnets_list = torch.nn.ModuleList()
        for _ in range(self.num_of_resnets):
            if kwargs["useResnet"]:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                if kwargs["usePretrainedResnet"]:
                    for param in resnet.parameters():
                        param.requires_grad = False
                resnet.fc = torch.nn.Identity()
            else:
                resnet = customCNN()
            
            self.resnets_list.append(resnet)
        
        
        # Define a new fully connected layer
        if kwargs["generateImage"]:
            self.fc = CNNToImage(size_of_input = self.cnn_encoding_size * self.num_of_resnets + sum(linear_inputs_sizes), do_segmentation= kwargs["do_segmentation"])
        else:
            self.fc = customFullyConnectedLayer(size_of_input = self.cnn_encoding_size * self.num_of_resnets + sum(linear_inputs_sizes), size_of_output = size_of_output)
        

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


class sequentialModel(torch.nn.Module):
    def __init__(self, neueons_in_hidden_layer, dropout, size_of_output, num_of_resnets, linear_inputs_sizes, device, **kwargs):
        super(sequentialModel, self).__init__()
        self.num_of_resnets = num_of_resnets
        self.linear_inputs_sizes = linear_inputs_sizes
        self.sum_of_linear_inputs_sizes = sum(linear_inputs_sizes)
        self.device = device
        self.lstms_num_layers = 1
        self.lstm_franka_hidden_size = 50
        self.lstm_camera_hidden_size = 128

        if kwargs["useResnet"]:
            self.cnn_encoding_size = 512
        else:
            self.cnn_encoding_size = 144

        
        if self.sum_of_linear_inputs_sizes:
            self.lstm_franka = torch.nn.LSTM(input_size= self.sum_of_linear_inputs_sizes,
                                             hidden_size=self.lstm_franka_hidden_size,
                                             num_layers= self.lstms_num_layers,
                                             batch_first=True)
        else:
            self.lstm_franka_hidden_size = 0

        self.lstm_cameras = torch.nn.LSTM(input_size=self.cnn_encoding_size * self.num_of_resnets,
                                          hidden_size=self.lstm_camera_hidden_size,
                                          num_layers= self.lstms_num_layers,
                                          batch_first=True)
        
        
        self.resnets_list = torch.nn.ModuleList()
        for _ in range(self.num_of_resnets):
            if kwargs["useResnet"]:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                if kwargs["usePretrainedResnet"]:
                    for param in resnet.parameters():
                        param.requires_grad = False
                resnet.fc = torch.nn.Identity()
            else:
                resnet = customCNN()
            
            self.resnets_list.append(resnet)

        
        # Define a new fully connected layer
        if kwargs["generateImage"]:
            self.fc = CNNToImage(size_of_input = self.lstm_franka_hidden_size + self.lstm_camera_hidden_size, do_segmentation= kwargs["do_segmentation"])
        else:
            self.fc = customFullyConnectedLayer(size_of_input = self.lstm_franka_hidden_size + self.lstm_camera_hidden_size, size_of_output = size_of_output)
        

    def forward(self, x):
        # Pass through the two ResNet18 models
        
        batch_size, seq_length, C, H, W = x[0].size()
        cnn_tensor = torch.zeros(batch_size, seq_length, self.num_of_resnets * self.cnn_encoding_size, device= self.device)
        
        #cnn_features = list()
        for i in range(self.num_of_resnets):
            for t in range(seq_length):
                cnn_tensor[:, t , self.cnn_encoding_size*i : self.cnn_encoding_size* (i+1)] = self.resnets_list[i](x[i][:, t, :, :, :])
    
        h0_cam = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_camera_hidden_size).to(self.device)
        c0_cam = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_camera_hidden_size).to(self.device)
        cam_lstm_out, (hidden_state, cell_state) = self.lstm_cameras(cnn_tensor, (h0_cam, c0_cam))


        if self.sum_of_linear_inputs_sizes:
            franka_data_tensor = torch.zeros(batch_size, seq_length, self.sum_of_linear_inputs_sizes, device= self.device)
            offset = 0
            for i, size in enumerate(self.linear_inputs_sizes):
                franka_data_tensor[:, :, offset: offset+ size] = (x[i+self.num_of_resnets])

            h0_franka = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_franka_hidden_size).to(self.device)
            c0_franka = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_franka_hidden_size).to(self.device)
            franka_lstm_out, (h_franka, c_franka) = self.lstm_franka(franka_data_tensor, (h0_franka, c0_franka))

            fused_features = torch.cat((cam_lstm_out[:, -1, :], franka_lstm_out[:, -1, :]), dim=1)
        
        else:
            fused_features = cam_lstm_out[:, -1, :]

        output = self.fc(fused_features)
        
        return output
        

class sequentialModelFusingBeforeRNN(torch.nn.Module):
    def __init__(self, neueons_in_hidden_layer, dropout, size_of_output, num_of_resnets, linear_inputs_sizes, device, **kwargs):
        super(sequentialModelFusingBeforeRNN, self).__init__()
        self.num_of_resnets = num_of_resnets
        self.linear_inputs_sizes = linear_inputs_sizes
        self.sum_of_linear_inputs_sizes = sum(linear_inputs_sizes)
        self.device = device
        self.lstms_num_layers = 1
        self.lstm_franka_hidden_size = 50
        self.lstm_camera_hidden_size = 128

        if kwargs["useResnet"]:
            self.cnn_encoding_size = 512
        else:
            self.cnn_encoding_size = 144

        
        if not self.sum_of_linear_inputs_sizes:
            self.lstm_franka_hidden_size = 0
        
        self.lstm_hidden_size = self.lstm_franka_hidden_size + self.lstm_camera_hidden_size
        self.lstm_fused = torch.nn.LSTM(input_size=self.cnn_encoding_size * self.num_of_resnets + self.sum_of_linear_inputs_sizes,
                                          hidden_size=self.lstm_hidden_size,
                                          num_layers= self.lstms_num_layers,
                                          batch_first=True)
        
        
        self.resnets_list = torch.nn.ModuleList()
        for _ in range(self.num_of_resnets):
            if kwargs["useResnet"]:
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                if kwargs["usePretrainedResnet"]:
                    for param in resnet.parameters():
                        param.requires_grad = False
                resnet.fc = torch.nn.Identity()
            else:
                resnet = customCNN()
            
            self.resnets_list.append(resnet)

        
        # Define a new fully connected layer
        if kwargs["generateImage"]:
            self.fc = CNNToImage(size_of_input = self.lstm_hidden_size, do_segmentation= kwargs["do_segmentation"])
        else:
            self.fc = customFullyConnectedLayer(size_of_input = self.lstm_hidden_size, size_of_output = size_of_output)
        

    def forward(self, x):
        # Pass through the two ResNet18 models
        
        batch_size, seq_length, C, H, W = x[0].size()
        cnn_tensor = torch.zeros(batch_size, seq_length, self.num_of_resnets * self.cnn_encoding_size, device= self.device)
        
        #cnn_features = list()
        for i in range(self.num_of_resnets):
            for t in range(seq_length):
                cnn_tensor[:, t , self.cnn_encoding_size*i : self.cnn_encoding_size* (i+1)] = self.resnets_list[i](x[i][:, t, :, :, :])
    
        h0_cam = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_camera_hidden_size).to(self.device)
        c0_cam = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_camera_hidden_size).to(self.device)

        if self.sum_of_linear_inputs_sizes:
            franka_data_tensor = torch.zeros(batch_size, seq_length, self.sum_of_linear_inputs_sizes, device= self.device)
            offset = 0
            for i, size in enumerate(self.linear_inputs_sizes):
                franka_data_tensor[:, :, offset: offset+ size] = (x[i+self.num_of_resnets])

            h0_franka = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_franka_hidden_size).to(self.device)
            c0_franka = torch.zeros(self.lstms_num_layers, batch_size, self.lstm_franka_hidden_size).to(self.device)

            fused_tensor = torch.cat((cnn_tensor, franka_data_tensor), dim = 2)
            fused_h0_cam = torch.cat((h0_cam, h0_franka), dim = 2)
            fused_c0_cam = torch.cat((c0_cam, c0_franka), dim = 2)


        else: 
            fused_tensor = cnn_tensor
            fused_h0_cam = h0_cam
            fused_c0_cam = c0_cam


        lstm_out, (hidden_state, cell_state) = self.lstm_fused(fused_tensor, (fused_h0_cam, fused_c0_cam))
        output = self.fc(lstm_out[:, -1, :])
        
        return output
        




class CNNToImage(torch.nn.Module):
    def __init__(self, size_of_input, do_segmentation):
        super(CNNToImage, self).__init__()


        if do_segmentation:
            self.output_channels = 5
            align_corners_mode = None
            upscaling_mode = 'nearest'
        else:
            self.output_channels = 3
            upscaling_mode = 'bilinear' 
            align_corners_mode = True

        input_vector_size = size_of_input
        self.mapped_image_channels = 1
        self.mapped_image_height = 32
        self.mapped_image_width = 32

        self.lin_decoder = torch.nn.Linear(input_vector_size, self.mapped_image_channels * self.mapped_image_height * self.mapped_image_width)
        # Define your CNN layers (convolutions, pooling, etc.)
        self.upconv1 = torch.nn.ConvTranspose2d(self.mapped_image_channels, 16, kernel_size=3, stride=2, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(16, self.output_channels, kernel_size=3, stride=2, padding=1)
        self.upconv3 = torch.nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1)
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Instead of a fully connected layer, we'll have a final Conv2d layer that outputs an RGB image
          # RGB image has 3 channels
        self.do_segmentation = do_segmentation


        self.final_upconv = torch.nn.ConvTranspose2d(64, self.output_channels, kernel_size=3, stride=2, padding=1)

        self.softmax = torch.nn.Softmax(dim=1)
        
        # Optionally, to ensure the output size matches the desired image size
        self.upsample = torch.nn.Upsample(size=(224, 224), mode= upscaling_mode, align_corners=align_corners_mode)

    def forward(self, x):

        x = self.lin_decoder(x)
        x = x.view(-1, self.mapped_image_channels, self.mapped_image_height, self.mapped_image_width)
        
        x = self.upconv1(x)
        x = torch.relu(x)
        x = self.upconv2(x)
        #x = torch.relu(x)
        #x = self.upconv3(x)
        #x = torch.relu(x)
        #x = self.final_upconv(x)

        # Produce an RGB image with final convolution
        if self.do_segmentation:
            x = self.softmax(x)
        else:
            x = torch.relu(x)
            #x = 255* torch.sigmoid(self.final_upconv(x))
        #x *= 255

        #try:
        #    ha = g
        #except:
        #    print(torch.sum(x[0, :, 0, 0]))
        x = self.upsample(x)

        return x