################################
## Imports
################################
from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import missinglink

# Change the `project_token` to your own when trying to run this.
missinglink_project = missinglink.PyTorchProject()


################################
## Constants
################################

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict()

args.seed = 321
args.batch_size = 200
args.test_batch_size = 64
args.epochs = 3
args.lr = 0.03
args.momentum = 0.5
args.log_interval = 5

mnist_mean = 0.1307
mnist_std = 0.3081

torch.manual_seed(args.seed)

ACC_METRIC = 'Accuracy'
LOSS_METRIC = 'Loss'


################################
## NN Architecture
################################

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),

            # Final linear classifier = logits
            nn.Linear(50, 10),

            # Softmax = Normalization into probability distribution
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.features(input)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


################################
## Functions
################################

def get_correct_count(output, target):
    _, indexes = output.data.max(1, keepdim=True)
    return indexes.eq(target.data.view_as(indexes)).cpu().sum().item()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    seen = 0

    # `no_grad` so we don't use these calculations in backprop
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data)
            target = Variable(target)

            # inference
            output = model(data)
            seen += len(output)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # get the index of the max log-probability
            correct += get_correct_count(output, target)

    test_loss /= seen
    test_accuracy = correct * 100.0 / seen

    print(
        'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, seen, test_accuracy))
    return test_loss, test_accuracy


def train(model, optimizer, epoch, train_loader, validation_loader):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in experiment.batch_loop(iterable=train_loader):
        data, target = Variable(data), Variable(target)

        # Inference
        output = model(data)
        loss_t = F.nll_loss(output, target)

        # The iconic grad-back-step trio
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            train_loss = loss_t.item()
            train_accuracy = get_correct_count(output, target) * 100.0 / len(target)
            experiment.add_metric(LOSS_METRIC, train_loss)
            experiment.add_metric(ACC_METRIC, train_accuracy)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), train_loss))
            with experiment.validation():
                val_loss, val_accuracy = test(model, validation_loader)
                experiment.add_metric(LOSS_METRIC, val_loss)
                experiment.add_metric(ACC_METRIC, val_accuracy)


def get_train_test():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mnist_mean,), (mnist_std,))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mnist_mean,), (mnist_std,))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=True)

    return train_loader, test_loader


################################
## Main
################################
def main():
    global experiment

    # Get Data
    train_loader, test_loader = get_train_test()

    # instatiate NN model
    model = SimpleNet()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    with missinglink_project.create_experiment(
            model=model,
            optimizer=optimizer,
            train_data_object=train_loader,
    ) as experiment:
        first_batch = next(iter(train_loader))
        # for epoch in range(args.epochs):
        for epoch in experiment.epoch_loop(args.epochs):
            train(model, optimizer, epoch, [first_batch] * 50, [first_batch])
            # train(model, optimizer, epoch, [first_batch] * 50, [next(iter(test_loader))])
            # train(model, optimizer, epoch, [first_batch] * 50, test_loader)
            # train(model, optimizer, epoch, train_loader, test_loader)

        with experiment.test():
            print("--Final test--")
            test(model, test_loader)

    return model


trained_model = main()

