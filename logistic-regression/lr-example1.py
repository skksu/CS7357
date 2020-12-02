from numpy.random import randint
# Import the random number generation function in the numpy library

import matplotlib.pyplot as plt
# Import plot library

import numpy as np

X = np.array([x for x in range(1, 101)]).reshape(-1, 1)
# X = 1,2,...,n


y = np.array(list(map(lambda x: 2 * x + randint(-30, 30), X)))
# y=2X+r, r=random(-30,+30)

y_true = np.array(list(map(lambda x: 2 * x, X))) 
# y=2x

plt.plot(X, y_true, c='b') 
#draw a line graph

plt.scatter(X, y, alpha=0.7, s=60) 
# draw the points

plt.title('Random Scatter')
plt.show()



