import numpy as np
from utils.preprocessing import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def signum(x):
    if x < 0:
        return -1
    elif x > 0:
         return 1
    else:
        return 0


def Perceptron(feature1, feature2, class1, class2, eta, epochs, mse_threshold,bias):

    epochs = int(epochs)
    eta = float(eta)
    weights = np.random.uniform(low=0.1, high=0.5, size=(1, 3))

    custom_encoding = {class1: -1, class2: 1}
    x_train, x_test, y_train, y_test = DataSplit(class1, class2)


    y_encoded = y_train.map(custom_encoding)
    train_df = pd.concat([x_train, y_encoded], axis=1)

    y_test_encoded = y_test.map(custom_encoding)
    test_df = pd.concat([x_test, y_test_encoded], axis=1)

    x_features_train = train_df.iloc[:, [feature1, feature2]].values
    x_features_test = test_df.iloc[:, [feature1, feature2]].values
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_features_train)
    x_test_scaled = scaler.fit_transform(x_features_test)

    if bias == 1:
        X = np.c_[np.ones((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.ones((x_test_scaled.shape[0], 1)), x_test_scaled]

    else:
        X = np.c_[np.zeros((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.zeros((x_test_scaled.shape[0], 1)), x_test_scaled]


    yTrain = train_df.iloc[:, [-1]].values
    yTest = test_df.iloc[:, [-1]].values
    for i in range(epochs):
        for j in range(len(X)):

            nagib= np.dot(weights,X[j])
            y_i= signum(nagib)
            if y_i != yTrain[j] :
                loss= yTrain[j]- y_i

                weights+= eta*loss*X[j]
            else:
                continue



    c=0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for j in range(len(Xtest)):

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
    accuracy = (true_positive + true_negative) / (len(yTest))
    plt.imshow(confusion_matrix, cmap='gray_r')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate the corresponding x2 values using the decision boundary equation
    x2_values = np.linspace(min(x_test_scaled[:, 0]), max(x_test_scaled[:, 0]), 100)
    if bias == 1:
        x1_values =(-(weights[0][1] * x2_values + weights[0][0]) / weights[0][2])
    elif bias == 0:
        x1_values = -(weights[0][1] * x2_values ) / weights[0][2]






    # Create a scatter plot of the points of both classes
    class1_points = x_test_scaled[yTest.flatten() == -1]
    class2_points = x_test_scaled[yTest.flatten() == 1]

    plt.scatter(class1_points[:, 0], class1_points[:, 1], label=class1)
    plt.scatter(class2_points[:, 0], class2_points[:, 1], label=class2)
    plt.plot(x2_values, x1_values, color='red', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    print(f"Acc:{accuracy*100}% ")
    return round(accuracy*100, 2)



