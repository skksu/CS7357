from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

examDict={
    'Study time':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,
            2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
    'Score':    [10,  22,  13,  43,  20,  22,  33,  50,  62,  
              48,  55,  75,  62,  73,  81,  76,  64,  82,  90,  93]
}
examOrderDict=OrderedDict(examDict)
exam=pd.DataFrame(examOrderDict)

#Export labels and features from DataFrame
exam_X = exam['Study time']
exam_Y = exam['Score']

#Draw a scatter chart, judge whether it is suitable for the linear regression model by drawing, and comment out after the judgment
plt.scatter(exam_X, exam_Y, color = 'green')

#plt.ylabel('Scores')
#plt.xlabel('Times(h)')
#plt.title('Exam Data')
#plt.show()

#The ratio is 7:3
X_train, X_test, Y_train, Y_test = train_test_split(exam_X,exam_Y, train_size = 0.7)

#Import linear regression model
#sklearn requires the input features to be a two-dimensional array type.
#Data set has only 1 feature, you need to use array.reshape(-1, 1) to change the shape of the array

#Change the shape of the array
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

#Create a model
model = LinearRegression()
#Train the model
model.fit(X_train, Y_train)

a = model.intercept_
b = model.coef_
a = float(a)
b = float(b)
print('The linear regression equation of the model is y = {} + {} * x'.format(a, b))

accuracy = model.score(X_test, Y_test)
print(accuracy)

#Draw a scatter chart
plt.scatter(exam_X, exam_Y, color = 'green', label = 'train data')
#Set X, Y axis label and title
plt.ylabel('Scores')
plt.xlabel('Times(h)')

#Draw the best fit curve
Y_train_pred = model.predict(X_train)
plt.plot(X_train, Y_train_pred, color = 'black', label = 'best line')

#Output
plt.legend(loc = 2)
plt.show()