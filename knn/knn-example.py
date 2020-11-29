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
"plt.show()"

clf = neighbors.KNeighborsClassifier(n_neighbors = 5,
                     algorithm = 'kd_tree', weights='uniform')
clf.fit(X, y)

new_data = [[4.8, 1.6], [2, 0.6]]
result = clf.predict(new_data)

plt.plot(new_data[0][0], new_data[0][1], c = color[result[0]], marker='x', ms= 8)
plt.plot(new_data[1][0], new_data[1][1], c = color[result[1]], marker='x', ms= 8)
plt.show()




