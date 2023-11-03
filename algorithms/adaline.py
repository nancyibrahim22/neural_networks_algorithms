import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.preprocessing import *
import matplotlib.pyplot as plt


def Adaline(feature1, feature2, class1, class2, eta, epochs, mse_threshold,bias):
    weights = np.random.uniform(low=0.1,high=0.5,size=(1,3))
    print('initialized weights: ',weights)
    print('----------------------------')
    x_train,x_test,y_train,y_test = DataSplit(class1,class2)

    custom_encoding = {class1: -1, class2: 1}
    y_encoded = y_train.map(custom_encoding)
    train_df = pd.concat([x_train, y_encoded], axis=1)

    y_test_encoded = y_test.map(custom_encoding)
    test_df = pd.concat([x_test,y_test_encoded],axis=1)


    X_train = train_df.iloc[:, [feature1, feature2]].values
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit the scaler to your data and transform it
    X_train_scaled = scaler.fit_transform(X_train)


    X_test = test_df.iloc[:, [feature1, feature2]].values
    scaler = StandardScaler()
    # Fit the scaler to your data and transform it
    X_test_scaled = scaler.fit_transform(X_test)

    if(bias == 1):
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
    yTest = test_df.iloc[:,[-1]].values
    # print('y: ', y)
    error_list = []
    global val

    for i in range(epochs):
        for j in range(len(X)):
            print('-------------------------------')
            print('X[j]: ', X[j])
            print('--------------------------------')
            print('y[j]: ', yTrain[j])
            y_hat = np.dot(weights, X[j])
            error = (yTrain[j])-y_hat
            weights += (eta*error*(X[j]))
            print('---------------------------------')
            print('error in first loop: ', error)
            print('---------------------------------')
            print('weights:', weights)
            print('---------------------------------')
        val = weights
        print('---------------------------------')
        print('val: ', val)
        print('---------------------------------')
        print('val[0]: ', val[0][0])
        print('---------------------------------')
        print('val[1]: ', val[0][1])
        print('---------------------------------')
        print('val[2]: ', val[0][2])
        for l in range(len(X)):
            y_hat = np.dot(val, X[l])
            error = (yTrain[l]) - y_hat
            error_list = np.append(error_list,error)
        print('----------------------------------------------')
        print('total error list: ', error_list)
        # Calculate the sum of the squares of the values
        sum_of_error = sum(x ** 2 for x in error_list)
        print('----------------------------------------------')
        print('sum after squared error list: ',sum_of_error)
        mse = (sum_of_error * (0.5))/len(error_list)
        print('----------------------------------------------')
        print('mse',mse)
        if mse <= mse_threshold:
            x2 = (-val[0][0])/val[0][2]
            x1 = (-val[0][0])/val[0][1]
            X1, Y1 = 0, x2
            X2, Y2 = x1, 0

            # Create a Matplotlib figure and axis
            fig, ax = plt.subplots()

            # Plot the line using the two points
            ax.plot([X1, X2], [Y1, Y2], marker='o', color='b')

            # Add labels for the points
            ax.text(x1, Y1, f'({X1}, {Y1})', fontsize=12, verticalalignment='bottom')
            ax.text(x2, Y2, f'({X2}, {Y2})', fontsize=12, verticalalignment='bottom')

            # Set axis limits and labels
            # ax.set_xlim(0, 7)
            # ax.set_ylim(0, 8)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

            # Show the plot
            plt.grid()
            plt.show()
            break
        else:
            continue

    true_positive = 0
    true_negative= 0
    false_positive = 0
    false_negative = 0
    for k in range(len(Xtest)):
        y_test_hat = np.dot(val, Xtest[k])
        signmum = np.sign(y_test_hat)
        if signmum == yTest[k] and signmum == 1:
            true_positive += 1
        if signmum == yTest[k] and signmum == -1:
            true_negative += 1
        elif signmum != yTest[k] and signmum == 1:
            false_positive += 1
        elif signmum != yTest[k] and signmum == -1:
            false_negative
    print(true_positive)
    print(true_negative)
    print(false_positive)
    print(false_negative)
    confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
    print(confusion_matrix)
    accuracy = (true_positive+true_negative)/(len(yTest))
    print(accuracy)