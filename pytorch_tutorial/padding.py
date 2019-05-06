import torch
from torch.nn.utils.rnn import pad_sequence

a = torch.ones(4, 10)
b = torch.ones(5, 10)
c = torch.ones(2, 10)
padded_result = pad_sequence([a, b, c], batch_first=True, padding_value=0)
print(padded_result.size())


x = torch.tensor([4, 5, 8, 9, 3, 2, 2, 9, 7])
mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.uint8)
value = -1
x = x.masked_fill_(mask, value)
print(x)


class S:
    def __init__(self, a):
        self.a = a

s = S(3)
l = [s]*10


l[3].a = 5

print(l[4].a)

