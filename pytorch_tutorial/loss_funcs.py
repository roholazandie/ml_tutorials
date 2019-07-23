import torch
import torch.nn.functional as F
from torch import nn

def try_cross_entropy_loss():
    num_classes = 5
    batch_size = 3
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    target = torch.randint(num_classes, (batch_size,), dtype=torch.int64) #The `target` that this loss expects should be a class index in the range [0, C-1] where C is the num_classes
    loss = F.cross_entropy(logits, target)
    loss.backward()

    print(loss.item())

    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)

def nll_loss():
    num_classes = 5
    batch_size = 3
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    target = torch.randint(num_classes, (batch_size,),
                           dtype=torch.int64)  # The `target` that this loss expects should be a class index in the range [0, C-1] where C is the num_classes

    loss = F.nll_loss(logits, target)
    loss.backward()

    print(loss.item())

if __name__ == "__main__":
    try_cross_entropy_loss()
    nll_loss()