import matplotlib.pyplot as plt
import numpy as np

#The loadDataSet() function opens a text file separated by the tab key, 
#where the last value of each line of the default file is the target value.
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#The standRegres() function is used to calculate the best fit straight line.
#This function first reads x and y and saves them in the matrix, then calculates xTx, and then judges whether its determinant is 0.
#If the determinant is 0, an error will occur when calculating the inverse matrix.
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


xArr,yArr = loadDataSet('data.txt')

ws = standRegres(xArr,yArr)

xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat*ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()

