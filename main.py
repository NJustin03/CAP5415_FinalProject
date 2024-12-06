import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

train_accuracy_array = []
test_accuracy_array = []
test_loss_array = []
train_loss_array = []

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

def run_model(train_loader, test_loader, model_type, out):
    # Choose the model that we run
    if model_type == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        learning_rate = 0.005
    elif model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        learning_rate = 0.001
    elif model_type == 'shufflenet':
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        learning_rate = 0.001
    elif model_type == 'efficientnets':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        learning_rate = 0.001
    elif model_type == 'efficientnetm':
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        learning_rate = 0.001

    # Our final classification layer 
    num_classes = 196
    if model_type == 'efficientnets' or "efficientnetm":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer and number of epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 15

    # Run testing and training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_loop(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        train_accuracy_array.append(train_accuracy)
        train_loss_array.append(train_loss/len(train_loader))
        out.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%\n")

        test_loss, test_accuracy = test_loop(model, test_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
        test_accuracy_array.append(test_accuracy)
        test_loss_array.append(test_loss/len(train_loader))
        out.write(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

        out.write("\n")


def train_loop(model, train_loader, criterion, optimizer):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_predictions = 0  # Counter for correct predictions
    total_samples = 0  # Counter for total samples processed
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
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

def test_loop(model, test_loader, criterion, optimizer):
    # Testing loop
    model.eval()
    running_loss = 0.0
    correct_predictions = 0  # Counter for correct predictions
    total_samples = 0  # Counter for total samples processed
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
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

def vit_model(train_loader, test_loader, vit_type, out):
    # Choose our model
    if vit_type == 'base':
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        learning_rate = 0.001
    elif vit_type == 'large':
        vit = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        learning_rate = 0.0005
    
    # Our final classification layer
    num_classes = 196
    vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes)
    vit = vit.to(device)

    # Define loss and optimizer and number of epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit.parameters(), lr=learning_rate)
    num_epochs = 15

    # Run testing and training loop
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_loop(vit, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        train_accuracy_array.append(train_accuracy)
        train_loss_array.append(train_loss/len(train_loader))
        out.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%\n")

        test_loss, test_accuracy = test_loop(vit, test_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
        test_accuracy_array.append(test_accuracy)
        test_loss_array.append(test_loss/len(train_loader))
        out.write(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss/len(train_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

        out.write("\n")

def plot_results(arr, title, linelabel, xlabel, ylabel):
        plt.figure(figsize=(6, 6))
        x = list(range(1, len(train_loss_array) + 1))
        plt.plot(x, arr, label=linelabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        # Save the plot instead of displaying it
        save_path = os.path.abspath(os.path.join(".", f"{title.replace(' ', '_')}.png"))
        plt.savefig(save_path)
        plt.close()

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

    out = open("output.txt", "w")

    # Load pre-trained ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = "efficientnetm"
    run_model(train_loader, test_loader, model, out)

    vit_type = 'base'
    #vit_model(train_loader, test_loader, vit_type, out)

    out.close()

    # Show the graphs
    plot_results(train_loss_array, "Train Loss over Epochs", "Train Loss", "Epoch", "Train Loss")
    plot_results(train_accuracy_array, "Train Accuracy over Epochs", "Train Accuracy", "Epoch", "Train Accuracy")
    plot_results(test_loss_array, "Test Loss over Epochs", "Test Loss", "Epoch", "Test Loss")
    plot_results(test_accuracy_array, "Test Accuracy over Epoch", "Test Accuracy", "Epoch", "Test Accuracy")
    
    
