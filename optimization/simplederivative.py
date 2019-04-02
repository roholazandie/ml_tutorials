from __future__ import division
import autograd.numpy as np
from autograd import jacobian
from autograd import grad
from autograd.util import flatten

# def func(x):
#    prev = 0
#    result = []
#    for i in range(0, 3):
#       prev = r = x**i + prev
#       result.append(r)
#    return np.array(result)
#
# print jacobian(func)(3.)

def bar(params):
    return params[0]**2*np.sin(params[1]) + np.cos(params[1])
    #params, _ = flatten([params[0]**2, np.sin(params[1])])
    #return sum(params)

gradient = grad(bar)
grad_array = gradient([1.0, np.pi/2])
print(grad_array)


# One input many output
def myfunc(x):
    return np.array([np.sin(x), np.cos(x)])

myfunc_jacobian = jacobian(myfunc)
res = myfunc_jacobian(3.0)
print(res)

# Many input many output
def func2(params):
    return np.array([np.sin(params[0])*np.cos(params[1]), np.exp(params[0])])

func2_jacobian = jacobian(func2)
result2 = func2_jacobian(np.array([10.0, np.pi/2]))
print(result2)

# gradient with respect to
def myfunc2(x, y):
    return x**2 + y**3

myfunc2_grad = grad(myfunc2, 1)
res = myfunc2_grad(3.0, 4.0)
print(res)

# Jacobian
def func3(W, X, b):
    W = np.array([W[0], W[1]])
    f, _ = flatten(np.dot(W, X)+b)
    return f

W0 = np.array([1.0, 2.0])

X = np.array([[2.0], [3.0]])

b = np.array([[2.0], [1.0]])
inp = np.array(W0)
c = func3(inp, X, b)

f_jac = jacobian(func3, 0)
m = f_jac(inp, X, b)
print(m.shape)
print(m)