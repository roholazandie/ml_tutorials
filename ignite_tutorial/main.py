import os
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from torch import nn


def train_and_store_loss(engine, batch):
    # inputs, targets = batch
    # optimizer.zero_grad()
    # outputs = model(inputs)
    # loss = loss_fn(outputs, targets)
    # loss.backward()
    # optimizer.step()
    # return loss.item()
    return None

trainer = Engine(lambda engine, batch: None)
checkpoint_dir = "models"
handler = ModelCheckpoint(checkpoint_dir, "myprefix", save_interval=2, n_saved=2, create_dir=True)

model = nn.Linear(3, 3)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"mymodel": model})
trainer.run([0], max_epochs=6)
print(os.listdir(checkpoint_dir))