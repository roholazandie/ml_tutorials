from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 20, endpoint=False)
y = np.cos(-x**2/6.0)

f = signal.resample(y, 100)

xnew = np.linspace(0, 10, 100, endpoint=False)
f2 = np.interp(xnew, x, y)

plt.plot(x, y, 'go-', xnew, f2, '.-', xnew, f, 10)
plt.show()