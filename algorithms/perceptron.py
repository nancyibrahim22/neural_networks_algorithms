import numpy as np
from utils.preprocessing import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    weights = np.random.uniform(low=0.4, high=0.5, size=(1, 3))
    #print(weights)
    custom_encoding = {class1: -1, class2: 1}
    x_train, x_test, y_train, y_test = DataSplit(class1, class2)

    ####################
    y_encoded = y_train.map(custom_encoding)
    train_df = pd.concat([x_train, y_encoded], axis=1)

    y_test_encoded = y_test.map(custom_encoding)
    test_df = pd.concat([x_test, y_test_encoded], axis=1)

    X_train = train_df.iloc[:, [feature1, feature2]].values


    X_test = test_df.iloc[:, [feature1, feature2]].values
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.fit_transform(X_test)

    if (bias == 1):
        X = np.c_[np.ones((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.ones((x_test_scaled.shape[0], 1)), x_test_scaled]
        # print('X after scaled: ',X)
        # print('--------------------------------')
    else:
        X = np.c_[np.zeros((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.zeros((x_test_scaled.shape[0], 1)), x_test_scaled]
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

    X = np.linspace(min(x_train_scaled[:, 0]), max(x_train_scaled[:, 0]), 100)
    Y =-(weights[0][1] * X + weights[0][0]) / weights[0][2]

    # Calculate the corresponding x2 values using the decision boundary equation


    # Create a scatter plot of the points of both classes
    class1_points = x_train_scaled[yTrain.flatten() == -1]
    class2_points = x_train_scaled[yTrain.flatten() == 1]

    plt.scatter(class1_points[:, 0], class1_points[:, 1], label=class1)
    plt.scatter(class2_points[:, 0], class2_points[:, 1], label=class2)
    plt.plot(X, Y, color='red', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

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
    # x1_values = np.linspace(X_train[0], X_train[59], 100)
    #
    # # Calculate the corresponding x2 values based on the equation w1 * x1 + w2 * x2 + b = 0
    # x2_values = (-w1 * x1_values - b) / w2
    #
    # # Plot the line
    # plt.plot(x1_values, x2_values, label='ya 3adel yabny'.format(w1, w2, b))
    # Plot the line
    #plt.plot(x_values, y_values, marker='o', linestyle='-')

    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.show()

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

    confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
    print(confusion_matrix)
    plt.imshow(confusion_matrix, cmap='gray_r')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.show()
    accuracy = (true_positive + true_negative) / (len(yTest))
    #print(accuracy*100)
    print(f"Acc:{accuracy*100}% ")



