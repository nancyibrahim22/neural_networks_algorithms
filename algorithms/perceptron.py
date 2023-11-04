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
    #print(weights)
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
    #############
    import matplotlib.pyplot as plt




    # Create a scatter plot of the points of both classes
    class1_points = X_train[yTrain.flatten() == -1]
    class2_points = X_train[yTrain.flatten() == 1]

    plt.scatter(class1_points[:, 0], class1_points[:, 1], label=class1)
    plt.scatter(class2_points[:, 0], class2_points[:, 1], label=class2)

    ###Line
    ## w1x1 + w2x2 +b=0

    # p1x=0
    # p1y=(-weights[0][0])/weights[0][2]
    #
    # p2x=0
    # p2y=(-weights[0][0])/weights[0][1]
    #
    # x_values = [p1x, p2x]
    # y_values = [p1y, p2y]
    w1 = weights[0][1]
    w2 = weights[0][2]
    b = weights[0][0]

    # Generate a range of x1 values
    x1_values = np.linspace(X_train[0], X_train[59], 100)

    # Calculate the corresponding x2 values based on the equation w1 * x1 + w2 * x2 + b = 0
    x2_values = (-w1 * x1_values - b) / w2

    # Plot the line
    plt.plot(x1_values, x2_values, label='ya 3adel yabny'.format(w1, w2, b))
    # Plot the line
    #plt.plot(x_values, y_values, marker='o', linestyle='-')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    ##############
    c=0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for j in range(len(Xtest)):
       # print(weights)
       # print(x)
        nagib= np.dot(weights,Xtest[j])
        y_i= signum(nagib)
        if y_i==yTest[j]:
            c+=1
        if y_i == yTest[j] and y_i == 1:
            true_positive += 1
        if y_i == yTest[j] and y_i == -1:
            true_negative += 1
        elif y_i != yTest[j] and y_i == 1:
            false_positive += 1
        elif y_i != yTest[j] and y_i == -1:
            false_negative+=1
    # print(true_positive)
    # print(true_negative)
    # print(false_positive)
    # print(false_negative)
    confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
    print(confusion_matrix)
    accuracy = (true_positive + true_negative) / (len(yTest))
    #print(accuracy*100)
    print(f"Acc:{accuracy*100}% ")



