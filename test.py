# import time
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Output file
out = open("output.txt", "w")

# Set hyperparameters
MODEL_NUM = 1
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001


class Car(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Map textual labels to integers
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations['true_class_name'].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = os.path.join(self.img_dir, str(self.annotations.iloc[idx, 6]).strip("'"))
        label = self.annotations.iloc[idx, 4]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        
        # Pooling functions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max Pool over 2x2
        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Fully connected layers - nn.Linear(in_features, out_features)
        self.fc1 = nn.Linear(in_features=(16 * 4 * 4), out_features=200) # Reading data from kernel output
        self.fc1_model2 = nn.Linear(in_features=(16 * 2 * 2), out_features=200) # Reading data from kernel output
        self.fc2 = nn.Linear(in_features=200, out_features=100)        # Reading data from fc1
        self.fc_out = nn.Linear(in_features=100, out_features=10)      # Output data

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-2")
            exit(0)
        
        
    def model_1(self, x):
        # Apply the first convolutional layer with 40 filters of 3x3, stride 1, padding 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply the second convolutional layer with 20 filters of 5x5, stride 1, padding 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply the third convolutional layer with 10 filters of 7x7, stride 1, padding 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output of the conv layers to feed into the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Fully connected layer (100 neurons with Sigmoid)
        x = self.fc1(x)
        x = self.relu(x)
        # Fully connected layer (100 neurons with Sigmoid)
        x = self.fc2(x)
        x = self.relu(x)
        # Final output layer (10 output neurons for digit classification)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x

    def model_2(self, x):
        # Apply the first convolutional layer with 40 filters of 3x3, stride 1, padding 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply the second convolutional layer with 20 filters of 5x5, stride 1, padding 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply the third convolutional layer with 10 filters of 7x7, stride 1, padding 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        # Apply the fourth convolutional layer with 16 filters of 3x3, stride 1, padding 1
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output of the conv layers to feed into the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        # Fully connected layer (200 neurons with Sigmoid)
        x = self.fc1_model2(x)
        x = self.relu(x)
        # Fully connected layer (100 neurons with Sigmoid)
        x = self.fc2(x)
        x = self.relu(x)
        # Final output layer (10 output neurons for digit classification)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


def train(model, device, train_loader, optimizer, criterion):
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for _batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # TODO: Do I need one hot?
        # When using MSELoss as criterion need to do one_hot to resize target
        # to match MSELoss expectations from [1,10] to [10,10]
        # target_one_hot = (F.one_hot(target, 10).float()).to(device)
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # ======================================================================
        # Count correct predictions overall 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / len(train_loader.dataset)
    out.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(train_loader.dataset), train_acc))
    return train_loss, train_acc


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for _batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # TODO: Do I need one hot?
            # When using MSELoss as criterion need to do one_hot to resize target
            # to match MSELoss expectations from [1,10] to [10,10]
            # target_one_hot = (F.one_hot(target, 10).float()).to(device)
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # ======================================================================
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    out.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    out.write("Torch device selected: " + ("cuda" if use_cuda else "cpu") + "\n")
    
    # Initialize the model and send to device 
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    # model = ConvNet(FLAGS.mode).to(device)
    
    # ======================================================================
    # Define loss function.
    criterion = nn.CrossEntropyLoss()
    
    # ======================================================================
    # Define optimizer function.
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load class labels and annotations
    train_csv = 'cars_train_labels.csv'
    test_csv = 'cars_test_labels.csv'
    test_img_dir = 'cars_test/'
    train_img_dir = 'cars_train/'
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    train_dataset = Car(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
    test_dataset = Car(csv_file=test_csv, img_dir=test_img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Run training for n_epochs specified in config
    print(f'Progress 0/{FLAGS.num_epochs}', end="\r")
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        print(f'Progress {epoch}/{FLAGS.num_epochs}', end="\r")
    
    # ======================================================================
    # Plot the test accuracy change over epochs
    def plot_results(arr, title, linelabel, xlabel, ylabel):
        plt.figure(figsize=(6, 3))
        plt.plot(range(1, FLAGS.num_epochs + 1), arr, label=linelabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        # Save the plot instead of displaying it
        save_path = os.path.abspath(os.path.join(".", f"{title.replace(' ', '_')}.png"))
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

    out.write("Best accuracy is {:2.2f}%\n".format(best_accuracy))
    out.write("Training and evaluation finished\n")
    plot_results(train_losses, "Train Loss over Epochs", "Train Loss", "Epoch", "Train Loss")
    plot_results(train_accuracies, "Train Accuracy over Epochs", "Train Accuracy", "Epoch", "Train Accuracy")
    plot_results(test_losses, "Test Loss over Epochs", "Test Loss", "Epoch", "Test Loss")
    plot_results(test_accuracies, "Test Accuracy over Epoch", "Test Accuracy", "Epoch", "Test Accuracy")


if len(sys.argv) > 2:
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=60,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    run_main(FLAGS)
else:
    FLAGS = argparse.Namespace(
        mode=MODEL_NUM,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    run_main(FLAGS)
    out.close()



