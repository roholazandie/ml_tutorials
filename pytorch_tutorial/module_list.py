import torch
from torch import nn



class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i](x) + l(x)
        return x


if __name__ == "__main__":
    my_module = MyModule()
    x = torch.rand((10, 10))
    out = my_module(x)
    print(out)