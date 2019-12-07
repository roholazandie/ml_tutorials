import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

#alpha = [0.01, 0.01, 0.01] # sparse
#alpha = [10, 2, 8]
alpha = [1, 1, 1]
for i in range(25):
    pmf = np.random.dirichlet(alpha)
    plt.subplot(5, 5, i+1)
    y_pos = np.arange(len(pmf))
    plt.bar(y_pos, pmf, align='center', alpha=0.5)

plt.show()



