import torch


def foo(x, y):
    return 2*x + y

traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

print(traced_foo)