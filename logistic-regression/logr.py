import numpy as np
import matplotlib.pyplot as plt
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



# 导入数据，“_orig”代表这里是原始数据，我们还要进一步处理才能使用：
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#由数据集获取一些基本参数，如训练样本数m，图片大小：
m_train = train_set_x_orig.shape[0]  #训练集大小209
m_test = test_set_x_orig.shape[0]    #测试集大小50
num_px = train_set_x_orig.shape[1]  #图片宽度64，大小是64×64
#将图片数据向量化（扁平化）：
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#对数据进行标准化：
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)


def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def propagate(w, b, X, Y):
    """
Passage:
     w - weight, shape: (num_px * num_px * 3, 1)
     b - bias term, a scalar
     X - data set, shape: (num_px * num_px * 3, m), m is the number of samples
     Y - real label, shape: (1,m)

     return value:
     cost, dw, db, the latter two are placed in a dictionary grads
     """
     #Get the number of samples m:
    m = X.shape[1]

    # Forward propagation:
    A = sigmoid(np.dot(w.T,X)+b)    # Call the sigmoid function earlier    
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m                 

    # Backpropagation:
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m
    db = (np.sum(dZ))/m

    # Return value
    grads = {"dw": dw,
             "db": db}

    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    #Define a costs array to store the cost after every several iterations, 
    #so that you can draw a graph to see the change trend of the cost:
    costs = []
    #To iterate:
    for i in range(num_iterations):
        # Use propagate to calculate the cost and gradient after each iteration:
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        # Use the gradient obtained above to update the parameters:
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Every 100 iterations, save a cost to see:
        if i % 100 == 0:
            costs.append(cost)

        # We can print out the cost every 100 times to see the progress of the model at any time:
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #After iteration, put the final parameters into the dictionary and return:
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs



def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction




def logistic_model(X_train,Y_train,X_test,Y_test,learning_rate=0.1,num_iterations=2000,print_cost=False):
    #Obtain feature dimensions, initialization parameters:
    dim = X_train.shape[0]
    W,b = initialize_with_zeros(dim)

    #Gradient descent, iteratively find the model parameters:
    params,grads,costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    W = params['w']
    b = params['b']

    #Use the learned parameters to predict:
    prediction_train = predict(W,b,X_train)
    prediction_test = predict(W,b,X_test)

    #Calculate the accuracy rate, respectively on the training set and test set:
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:",accuracy_train )
    print("Accuracy on test set:",accuracy_test )

   #In order to facilitate analysis and inspection, we store all the parameters 
   #and hyperparameters obtained in a dictionary and return them:
    d = {"costs": costs,
         "Y_prediction_test": prediction_test , 
         "Y_prediction_train" : prediction_train , 
         "w" : W, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "train_acy":accuracy_train,
         "test_acy":accuracy_test
        }
    return d



d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)



d['costs']
plt.plot(np.squeeze(d['costs']))

plt.show()
