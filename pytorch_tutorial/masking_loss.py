import torch
from torch import nn



lm_criterion = nn.CrossEntropyLoss(ignore_index=0)


input = torch.ones(6, 5, requires_grad=True)/2
target = torch.tensor([1, 2, 3, 1, 1, 1], dtype=torch.long)

#mask some target



res = lm_criterion(input, target)

print(res.item())