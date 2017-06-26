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
print grad_array


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