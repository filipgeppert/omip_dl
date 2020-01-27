import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from src.settings import BASE_DIR

# PARAMS
num_epochs = 20
num_classes = 2
batch_size = 50
learning_rate = 0.001
crop_size = 400
read_model = True
dim = 3
classes = ('Other', 'Plans')


# Create data transformer
data_transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor()
    ])

dataset = datasets.ImageFolder(root=os.path.join(BASE_DIR, 'images', 'root_data'),
                               transform=data_transform,
                               )

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

# Device configuration
device = torch.device('cuda:0'
                      if torch.cuda.is_available()
                      else 'cpu'
                      )


class ConvNet3d(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(ConvNet3d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(int(crop_size / (2*2*2)) * int(crop_size / (2*2*2)) * 8, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = ConvNet3d(num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

running_loss = 0.0
# Train the model
total_step = len(trainloader)

for epoch in range(1):
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        running_loss += loss.item()
        running_loss = 0.0
