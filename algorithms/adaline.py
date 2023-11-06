import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import *
import matplotlib.pyplot as plt


def Adaline(feature1, feature2, class1, class2, eta, epochs, mse_threshold, bias):
    # Making initial weights and bias with small random values
    weights = np.random.uniform(low=0.1, high=0.5, size=(1, 3))

    # Making the values of confusion matrix with zeros
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # Making a list for errors and last_weights
    error_list = []
    global last_weights

    # Calling DataSplit function
    x_train, x_test, y_train, y_test = DataSplit(class1, class2)

    # Encoding the classes (class1 : -1) and (class2: 1) in train and test
    custom_encoding = {class1: -1, class2: 1}
    y_train_encoded = y_train.map(custom_encoding)
    y_test_encoded = y_test.map(custom_encoding)

    # Concatenating X and Y in train and test
    train_df = pd.concat([x_train, y_train_encoded], axis=1)
    test_df = pd.concat([x_test, y_test_encoded], axis=1)

    # Taking the values of specific features in train and test
    x_features_train = train_df.iloc[:, [feature1, feature2]].values
    x_features_test = test_df.iloc[:, [feature1, feature2]].values

    # Taking the values of y in train and test
    yTrain = train_df.iloc[:, [-1]].values
    yTest = test_df.iloc[:, [-1]].values

    # Scaling the values of X in train and test
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_features_train)
    x_test_scaled = scaler.fit_transform(x_features_test)

    # Checking if there is bias or not if there is bias we add ones to X in train and test if not we add zeros
    if bias == 1:
        xTrain = np.c_[np.ones((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.ones((x_test_scaled.shape[0], 1)), x_test_scaled]

    else:
        xTrain = np.c_[np.zeros((x_train_scaled.shape[0], 1)), x_train_scaled]
        Xtest = np.c_[np.zeros((x_test_scaled.shape[0], 1)), x_test_scaled]

    # Training the data
    for i in range(epochs):
        for j in range(len(xTrain)):
            # Calculating y_hat by making W.X and calculating the error
            y_hat = np.dot(weights, xTrain[j])
            error = (yTrain[j])-y_hat
            # Updating weights
            weights += (eta*error*(xTrain[j]))
        # Making last_weights = the last weights after finishing the loop that loops on all samples
        last_weights = weights
        for l in range(len(xTrain)):
            y_hat = np.dot(last_weights, xTrain[l])
            error = (yTrain[l]) - y_hat
            error_list = np.append(error_list, error)
        # Calculating the sum of the squares of the errors in error_list
        sum_of_error = sum(errors ** 2 for errors in error_list)
        # Calculating mean squared error
        mse = (sum_of_error * (0.5))/len(error_list)
        # Checking if mse that we calculate <= or > mse threshold
        if mse <= mse_threshold:
            # x2 = (-last_weights[0][0])/last_weights[0][2]
            # x1 = (-last_weights[0][0])/last_weights[0][1]
            # X1, Y1 = 0, x2
            # X2, Y2 = x1, 0

            # # Create a Matplotlib figure and axis
            # fig, ax = plt.subplots()
            #
            # # Plot the line using the two points
            # ax.plot([X1, X2], [Y1, Y2], marker='o', color='b')
            #
            # # Add labels for the points
            # ax.text(x1, Y1, f'({X1}, {Y1})', fontsize=12, verticalalignment='bottom')
            # ax.text(x2, Y2, f'({X2}, {Y2})', fontsize=12, verticalalignment='bottom')
            #
            # # Set axis limits and labels
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')
            #
            # # Show the plot
            # plt.grid()
            # plt.show()
            break
        else:
            continue
    X = np.linspace(min(x_train_scaled[:, 0]), max(x_train_scaled[:, 0]), 100)
    Y = -(weights[0][1] * X + weights[0][0]) / weights[0][2]

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

    # Testing the data
    for k in range(len(Xtest)):
        y_test_hat = np.dot(last_weights, Xtest[k])
        signum = np.sign(y_test_hat)
        if signum == yTest[k] and signum == 1:
            true_positive += 1
        if signum == yTest[k] and signum == -1:
            true_negative += 1
        elif signum != yTest[k] and signum == 1:
            false_positive += 1
        elif signum != yTest[k] and signum == -1:
            false_negative += 1

    # calculating confusion matrix
    confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
    print('Confusion Matrix: ', confusion_matrix)

    # Plotting the matrix
    plt.imshow(confusion_matrix, cmap='gray_r')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.show()

    # Calculating the accuracy from confusion matrix
    accuracy = (true_positive+true_negative)/(len(yTest))
    print('Accuracy: ', accuracy)
    return accuracy


