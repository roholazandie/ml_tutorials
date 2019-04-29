import torch
from torch.autograd import Variable

def basic_fun(x):
    return 3*(x*x)

def get_grad(inp, grad_var):
    A = basic_fun(inp)
    A.backward()
    return grad_var.grad

x = Variable(torch.FloatTensor([1]), requires_grad=True)
xx = x.clone()

# Grad wrt x will work
print(x.is_leaf) # is it a leaf? Yes
print(get_grad(x, x))
print(get_grad(xx, x))

# Grad wrt xx won't work
print(xx.is_leaf) # is it a leaf? No
print(get_grad(xx, xx))
print(get_grad(x, xx))