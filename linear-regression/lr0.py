import matplotlib.pyplot as plt
import numpy as np

#loadDataSet()函数打开一个用tab键分隔的文本文件，这里默认文件每行的最后一个值是目标值。
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

#standRegres()函数用于计算最佳拟合直线。
#该函数首先读入x和y并将它们保存到矩阵中，然后计算xTx，然后判断它的行列式是否为0。
#如果行列式为0，那么计算逆矩阵的时候将出现错误。
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

