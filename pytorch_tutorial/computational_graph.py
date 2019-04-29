import torch
from torch.autograd import Variable


a = Variable(torch.rand(1, 4), requires_grad=True)
b = a**2
c = b*2
d = c.mean()
e = c.sum()


def wrong_use():
    d.backward()
    e.backward() # this gives an error because after d.backward() the memory for the graph is freed then e can't do grad
    print(d)


def right_use():
    d.backward(retain_graph=True)
    e.backward()



if __name__ == "__main__":
    #wrong_use()
    right_use()