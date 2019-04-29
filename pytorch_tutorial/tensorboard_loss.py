import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime



class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)  # nSamples x nChannels x Height x Width
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        # Max pooling over 2x2 window
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features



def train(model, train_loader):
    # Initialize the visualization environment
    writer = SummaryWriter()


    # Training loop
    loss_values = []
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for step, data in enumerate(train_loader, 0):
        # Forward pass
        inputs, labels = data

        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        loss_values.append(loss.item())

        # Backward pass
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Visualization data
        if step % 10 == 0:
            print("loss " + str(np.mean(loss_values)))
            #vis.plot_loss(np.mean(loss_values), step)
            writer.add_scalar('data/scalar1', np.mean(loss_values), step)
            loss_values.clear()

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root="../data", train=True,
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="../data", train=False,
                                           download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                             shuffle=False, num_workers=2)
    model = Net()
    train(model, train_loader)