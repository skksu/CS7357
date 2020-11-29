import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

color = ("red", "blue", "green")
plt.scatter(X[:50, 0], X[:50, 1], c = color[0], marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], c = color[1], marker='^', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1], c = color[2], marker='+', label='Virginica')

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()


