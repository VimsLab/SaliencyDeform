import matplotlib.pyplot as plt
import numpy as np

X = np.arange(-5, 5, 1)
Y = np.arange(-5, 5, 1)
xx, yy= np.meshgrid(X, Y)

fig, ax = plt.subplots()
plt.plot(xx, yy, marker='.', color='k', linestyle='none')
plt.show()
