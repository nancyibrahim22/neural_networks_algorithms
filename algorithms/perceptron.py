import numpy as np
from utils.preprocessing import *
import pandas as pd

def signum(x):
    if x < 0:
        return -1
    elif x > 0:
         return 1
    else:
        return 0


def Perceptron(feature1, feature2, class1, class2, eta, epochs, mse_threshold,bias):
    #print('write your code here')
    epochs= int(epochs)
    eta = float(eta)
    weights = np.random.uniform(low=0.1, high=0.5, size=(1, 3))
    print(weights)
    custom_encoding = {class1: -1, class2: 1}
    x_train, x_test, y_train, y_test = DataSplit(class1, class2)

    ####################
    y_encoded = y_train.map(custom_encoding)
    train_df = pd.concat([x_train, y_encoded], axis=1)

    y_test_encoded = y_test.map(custom_encoding)
    test_df = pd.concat([x_test, y_test_encoded], axis=1)

    X_train = train_df.iloc[:, [feature1, feature2]].values
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    X_test = test_df.iloc[:, [feature1, feature2]].values
    scaler = StandardScaler()
    # Fit the scaler to your data and transform it
    X_test_scaled = scaler.fit_transform(X_test)
    if (bias == 1):
        X = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
        Xtest = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
        # print('X after scaled: ',X)
        # print('--------------------------------')
    else:
        X = np.c_[np.zeros((X_train_scaled.shape[0], 1)), X_train_scaled]
        Xtest = np.c_[np.zeros((X_test_scaled.shape[0], 1)), X_test_scaled]
        # print('X after scaled: ', X)
        # print('--------------------------------')

    yTrain = train_df.iloc[:, [-1]].values
    yTest = test_df.iloc[:, [-1]].values
    ########
    #x_train, x_test, y_train, y_test = DataSplit(class1, class2)
    # x1 = pd.DataFrame()
    # x2=  pd.DataFrame()
    # x1=  x_train.iloc[:,feature1]
    # x2 = x_train.iloc[:,feature2]
    # res = pd.concat([x1, x2], axis=1)
    # print(res.head())
    for i in range(epochs):
        for j in range(len(X)):
           # print(weights)
           # print(x)
            nagib= np.dot(weights,X[j])
            y_i= signum(nagib)
            if y_i != yTrain[j] :
                loss= yTrain[j]- y_i
                #weights+= np.dot(np.dot(eta , loss),x)
                weights+= eta*loss*X[j]
            else:
                continue
    c=0
    for j in range(len(Xtest)):
       # print(weights)
       # print(x)
        nagib= np.dot(weights,Xtest[j])
        y_i= signum(nagib)
        if y_i==yTest[j]:
            c+=1
    print("Nagib")
    print(c)

