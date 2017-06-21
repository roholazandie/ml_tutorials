import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

regularization = lambda coefficient: np.sum(np.square(coefficient))


#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
#np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))

#x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
#y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])


regs = []
n = 400
for i in range(1, n):
    coefficient = np.polyfit(x, y, i)
    regs.append(regularization(coefficient))

regs = np.log(regs)
print(regs)
#p30 = np.poly1d(coefficient30)
#print(coefficient30)
import matplotlib.pyplot as plt
x = np.linspace(1, n, n-1)
plt.plot(x, regs)
plt.show()

# import matplotlib.pyplot as plt
# xp = np.linspace(-2, 6, 100)
# _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
# plt.ylim(-2,2)
# plt.show()