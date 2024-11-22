import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the filename and label from the CSV
        img_filename = (str(self.annotations.iloc[idx, 6])).strip("'")
        
        label = self.annotations.iloc[idx, 4]

        # Construct the image path
        img_path = os.path.join(self.img_dir, img_filename)

        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resnet_model(train_loader, test_loader):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_classes = 196
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    num_epochs = 5

    for epoch in range(num_epochs):
        train_loss, train_accuracy = resnet_train(resnet, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

        test_loss, test_accuracy = resnet_test(resnet, test_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")


def resnet_train(resnet, train_loader, criterion, optimizer):
    # Training loop
    resnet.train()
    running_loss = 0.0
    correct_predictions = 0  # Counter for correct predictions
    total_samples = 0  # Counter for total samples processed
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = resnet(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Get the predicted class labels (index of max logit)
        _, predicted = torch.max(outputs, 1)
        
        # Update the number of correct predictions
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    # Calculate accuracy for this epoch
    accuracy = 100 * correct_predictions / total_samples
    
    return running_loss, accuracy

def resnet_test(resnet, test_loader, criterion, optimizer):
    # Testing loop
    resnet.eval()
    running_loss = 0.0
    correct_predictions = 0  # Counter for correct predictions
    total_samples = 0  # Counter for total samples processed
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = resnet(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Update running loss
        running_loss += loss.item()
        
        # Get the predicted class labels (index of max logit)
        _, predicted = torch.max(outputs, 1)
        
        # Update the number of correct predictions
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    # Calculate accuracy for this epoch
    accuracy = 100 * correct_predictions / total_samples

    return running_loss, accuracy


if __name__ == '__main__':
    # Load class labels and annotations
    train_csv = 'cars_train_labels.csv'
    test_csv = 'cars_test_labels.csv'
    test_img_dir = 'cars_test/'
    train_img_dir = 'cars_train/'

    train_dataset = Car(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
    test_dataset = Car(csv_file=test_csv, img_dir=test_img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load pre-trained ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model(train_loader, test_loader)

    #device = torch.device("cpu")
    
