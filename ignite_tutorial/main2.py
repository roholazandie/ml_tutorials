import os

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Precision, RunningAverage, Recall, Loss
from torch import nn
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader

train_batch_size = 10
val_batch_size = 5
lr = 0.01

train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)

model = Net()
optimizer = SGD(model.parameters(), lr=lr)
#loss_fct = nn.CrossEntropyLoss()
loss_fct = nn.NLLLoss()


def update(engine, batch):
    x, y = batch
    y_pred = model(x)
    loss = loss_fct(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def inference(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        y_pred = model(x)
        batch_size = x.shape[0]
        nb_digits = 10
        y_onehot = torch.FloatTensor(batch_size, nb_digits).zero_()
        y_onehot.scatter_(1, torch.argmax(y_pred, dim=1).unsqueeze(1), 1)
    return y_onehot, y


trainer = Engine(update)
evaluator = Engine(inference)


metrics = {"precision": Precision(),
           "recall": Recall(),
           "accuracy": Accuracy(),
           "loss": Loss(nn.NLLLoss())}

for name in metrics:
    metrics[name].attach(evaluator, name)


#trainer.add_event_handler(Events.STARTED, lambda engine: print("started"))
#trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: print("epoch started"))
#trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: print("iteration completed"))
#trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: print("epoch ends"))

RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda _: print(evaluator.state.metrics))

trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
progress_bar = ProgressBar(persist=True)

progress_bar.attach(trainer, metric_names=["loss"])

# @trainer.on(Events.EPOCH_COMPLETED)
# def on_trainer_epochs_completed(engine):
#     print(engine.state.epoch)
#     print(engine.state.iteration)
#     evaluator.run(val_loader)


trainer.run(train_loader, max_epochs=3)
