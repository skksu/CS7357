# Read Auto MPG data
import pandas as pd
import numpy as np

data = pd.read_csv('auto-mpg.data.csv')

# make two variables - X and y
y = data.iloc[:, 0]  # the first columne is for class label
X = data.drop('mpg', axis=1)
X = X.drop('carname', axis=1)
n, p = X.shape # number of samples and features
X = X.iloc[:, 2] # use only the second feature
X = pd.DataFrame(np.c_[np.ones(n), X])

# plot the data
import matplotlib.pyplot as plt
plt.plot(X.iloc[:, 1], y, 'o')
plt.xlabel("displacement")
plt.ylabel("MPG")
plt.show()


def SolverLinearRegression(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

b = SolverLinearRegression(X, y)
print(b)

plt.plot(X.iloc[:, 1], y, 'o')
abline(b[1], b[0])
plt.show()

