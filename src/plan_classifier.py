import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from settings import *
from src.architecture import ConvNet3d
from src.utilities import test_model, compare_and_save_model_checkpoint


def run_model():
    # Create data transformer
    data_transform = transforms.Compose([
            transforms.RandomResizedCrop(CROP_SIZE),
            transforms.ToTensor()
        ])

    dataset = datasets.ImageFolder(root=INPUT_IMAGES_PATH,
                                   transform=data_transform,
                                   )

    train_size = int(len(dataset) * 0.75)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True)

    gpu_access = torch.cuda.is_available()
    # Device configuration
    device = torch.device('cuda:0'
                          if gpu_access
                          else 'cpu'
                          )

    model = ConvNet3d(num_classes=NUMBER_CLASSES, crop_size=CROP_SIZE).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    running_loss = 0.0
    # Train the model
    total_step = len(trainloader)

    print("Train model...")
    for epoch in range(NUMBER_EPOCHS):
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
                  .format(epoch + 1, NUMBER_EPOCHS, i + 1, total_step, loss.item()))

            running_loss += loss.item()
            running_loss = 0.0

    print("Test model..")
    results_true, results_pred, accuracy = test_model(model=model, data_loader=testloader)

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    compare_and_save_model_checkpoint(state=checkpoint, is_best=True, model_name=MODEL_NAME,
                                      checkpoint_dir=CHECKPOINT_DIR, info_dict={"accuracy": accuracy})


if __name__ == "__main__":
    run_model()
