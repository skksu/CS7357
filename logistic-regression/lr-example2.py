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

from sklearn.linear_model import LinearRegression 
# Import the linear regression module in the sklearn library

lr = LinearRegression() # define a linear regression model
lr.fit(X, y) # Fit the model to the data
y_pred = lr.predict(X) 

plt.scatter(X, y, alpha=0.5, s=50)
plt.plot(X, y_pred, c='r')
plt.plot(X, y_true, c='b')
plt.title('Linear Regression')
plt.show()


