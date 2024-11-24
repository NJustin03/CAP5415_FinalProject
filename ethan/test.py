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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Set hyperparameters
MODEL_NUM = 1
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
# Output file
out = open("output.txt", "w")


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


def train(model, device, train_loader, optimizer, criterion):
    # Set model to train mode before each epoch
    model.train()
    losses = []
    correct = 0
    # Iterate over entire training samples (1 epoch)
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        # Do forward pass for current set of data
        output = model(data)
        # Compute loss based on criterion
        loss = criterion(output, target)
        # Computes gradient based on final loss
        loss.backward()
        # Optimize model parameters based on learning rate and gradient
        optimizer.step()
        # Store loss
        losses.append(loss.item())

        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        # Count correct predictions overall
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / len(train_loader.dataset)
    out.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, correct, len(train_loader.dataset), train_acc))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, correct, len(train_loader.dataset), train_acc))
    return train_loss, train_acc


def test(model, device, test_loader, criterion):
    # Set model to eval mode to notify all layers.
    model.eval()
    losses = []
    correct = 0
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for _batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict classes by doing forward pass
            output = model(data)
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Count correct predictions overall
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)
    out.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


def run_main(FLAGS):
    #####################
    # Setup cuda device #
    #####################
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: " + ("cuda" if use_cuda else "cpu") + "\n")

    ######################################
    # Choose model, criterion, optimizer #
    ######################################
    # Pick the model and weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Setup the output and send to device
    num_classes = 196
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    # Set criterion for train and test, set optimizer for train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    ######################
    # Prep and load data #
    ######################
    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load class labels and annotations
    train_csv = 'cars_train_labels-adjusted.csv'
    test_csv = 'cars_test_labels-adjusted.csv'
    test_img_dir = 'cars_test/'
    train_img_dir = 'cars_train/'
    # Load datasets for training and testing
    train_dataset = Car(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
    test_dataset = Car(csv_file=test_csv, img_dir=test_img_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ############################
    # Run testing and training #
    ############################
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
    out.write("Best accuracy is {:2.2f}%\n".format(best_accuracy))
    print("Best accuracy is {:2.2f}%\n".format(best_accuracy))

    ##############################################
    # Plot the test and train, accuracy and loss #
    ##############################################
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
    plot_results(train_losses, "Train Loss over Epochs", "Train Loss", "Epoch", "Train Loss")
    plot_results(train_accuracies, "Train Accuracy over Epochs", "Train Accuracy", "Epoch", "Train Accuracy")
    plot_results(test_losses, "Test Loss over Epochs", "Test Loss", "Epoch", "Test Loss")
    plot_results(test_accuracies, "Test Accuracy over Epoch", "Test Accuracy", "Epoch", "Test Accuracy")


if __name__ == "__main__":
    FLAGS = argparse.Namespace(
        mode=MODEL_NUM,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    run_main(FLAGS)
    out.close()
