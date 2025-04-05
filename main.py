# trained model on 20 epochs
# trained images = 39209
# test images = 12630

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torch.onnx
import onnx

print("Step 1 Done - Importing Libraries") #---------------------------------------------------------------------------------------

# Define the CNN model
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

print("Step 2 Done - Created TrafficSignCNN Class")  #---------------------------------------------------------------------------------------

# Dataset Class
class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx]["Path"].lstrip("/")  
        img_path = os.path.join(self.root_dir, img_filename).replace("\\", "/")
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]["ClassId"])
        if self.transform:
            image = self.transform(image)
        return image, label

print("Step 3 Done - Created TrafficSignDataset Class")  #---------------------------------------------------------------------------------------

# Define transformations
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Step 4 Done - Data Transformation")  #---------------------------------------------------------------------------------------

# Specify dataset paths (Update these paths accordingly)
data_path = 'C:/Users/1/Desktop/Dyne/Project/gtsrb-german-traffic-sign/versions/1'  
train_csv = os.path.join(data_path, 'train.csv')  
test_csv = os.path.join(data_path, 'test.csv')  
train_images_path = os.path.join(data_path)  
test_images_path = os.path.join(data_path)    

print("Step 5 Done - Path Update")   #---------------------------------------------------------------------------------------

# Load dataset
train_dataset = TrafficSignDataset(csv_file=train_csv, root_dir=train_images_path, transform=data_transform)
test_dataset = TrafficSignDataset(csv_file=test_csv, root_dir=test_images_path, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Step 6 Done - Dataset and Loader")  #---------------------------------------------------------------------------------------

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Step 7 Done - Model, Loss Function, and Optimizer")  #---------------------------------------------------------------------------------------

# Training the model 
def train_model(model, train_loader, criterion, optimizer, epochs=10, save_pth="traffic_sign_model.pth", save_onnx="traffic_sign_model.onnx"):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} started")
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)} (Epoch {epoch+1}/{epochs})")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:  
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/(batch_idx+1):.4f}')

        print(f'Epoch {epoch+1}/{epochs} completed, Average Loss: {running_loss/len(train_loader):.4f}')
    
    # Save model in .pth format
    torch.save(model.state_dict(), save_pth)
    print(f"Model saved successfully at {save_pth}")

    # Convert and save model in .onnx format
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  
    torch.onnx.export(model, dummy_input, save_onnx, 
                      input_names=["input"], 
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    
    print(f"Model saved successfully in ONNX format at {save_onnx}")

    return model

print("Step 8 Done - Model Training function")   #---------------------------------------------------------------------------------------

# Testing the model
def test_model(model, test_loader):
    model.load_state_dict(torch.load("traffic_sign_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i + 1) % 50 == 0:  
                print(f'Test Batch {i+1}/{len(test_loader)}, Accuracy: {100 * correct / total:.2f}%')
    print(f'Final Accuracy: {100 * correct / total:.2f}%')

print("Step 9 Done - Model Testing function")   #---------------------------------------------------------------------------------------

print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# Run training and testing
train_model(model, train_loader, criterion, optimizer, epochs=20)
print("Step 10 Done - Training of Model")   #---------------------------------------------------------------------------------------

test_model(model, test_loader)
print("Step 11 Done - Testing of Model")    #---------------------------------------------------------------------------------------

print("Finally - Model Trained, Tested annd Saved")   #---------------------------------------------------------------------------------------