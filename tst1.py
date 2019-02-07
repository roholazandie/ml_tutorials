
import numpy as np
import numdifftools as nd

#A = np.random.rand(5,3)
x = np.array([[1.0], [9.0]])
b = np.array([[1.0], [2.0]])


fun = lambda A: np.dot(x.T, A) + b.T
#fun = lambda A: np.dot(A, x) + b
A = np.array([[1.0, 2.0],[0.0, 2.0]])


print(x)
jac = nd.Jacobian(fun)(A)
print(jac)
