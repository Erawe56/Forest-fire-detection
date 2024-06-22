import os
from ultralytics import YOLO

# Initialize the model
model = YOLO("yolov8n.pt")

# Define the path to your data.yaml file
data_path = "D:/virtual/test/data.yaml"  

# Train the model for 20 epochs 
results = model.train(data=data_path, epochs=20)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader

# # Define transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor(),           # Convert images to tensors
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
# ])

# # Define dataset paths
# train_path = 'D:/virtual/test/train/images'
# valid_path = 'D:/virtual/test/valid/images'
# test_path = 'D:/virtual/test/test/images'

# # Define dataset
# train_dataset = ImageFolder(root='train_path', transform=transform)
# test_dataset = ImageFolder(root='test_path', transform=transform)
# valid_dataset = ImageFolder(root='valid_path', transform=transform)

# # Define dataloaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # Define the neural network model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * 56 * 56, 512)
#         self.fc2 = nn.Linear(512, 10)  # Assuming 10 classes

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 56 * 56)  # Flatten the tensor
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Instantiate the model
# model = SimpleCNN()

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode
#     running_loss = 0.0
#     for i, (images, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if (i+1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
#             running_loss = 0.0

# # Save the trained model
# torch.save(model.state_dict(), 'trained_model.pt')

# # Validate the model
# model.eval()  # Set the model to evaluation mode
# valid_loss = 0.0
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in valid_loader:
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         valid_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Validation Loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

