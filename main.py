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

        # Map textual labels to integers
        self.label_mapping = {label: idx for idx, label in enumerate(self.annotations['true_class_name'].unique())}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the image path and label
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        label = self.annotations.iloc[idx, 1]
        label = self.label_mapping[label]  # Convert text label to integer
        image = Image.open(img_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
    resnet = models.resnet18(pretrained=True)
    num_classes = len(train_dataset.label_mapping)  # Use the dataset's mapping
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        resnet.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
