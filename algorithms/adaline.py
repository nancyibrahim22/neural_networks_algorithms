import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils.preprocessing import *


def Adaline(feature1, feature2, class1, class2, eta, epochs, mse_threshold,bias):
    weights = np.random.uniform(low=0.1,high=0.5,size=(1,3))
    #b=np.ones([1,60])

    no_epochs=2
    eta = 2
    x_train,x_test,y_train,y_test = DataSplit(class1,class2)

    custom_encoding = {class1: 0, class2: 1}
    y_encoded = y_train.map(custom_encoding)
    # print(category_data_encoded)
    train_df = pd.concat([x_train, y_encoded], axis=1)
    print(train_df)

    #columns_to_include = ['Area', 'roundnes']
    # # Create a list to store vectors for each
    # vector_list = []
    # for index, row in train_df.iterrows():
    #
    #     row_vector =[1]+[row[column] for column in columns_to_include]
    #     vector_list.append(row_vector)


    X = train_df.iloc[:, [feature1, feature2]].values
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = train_df.iloc[:, [-1]].values

    for i in range(1):
        for j in range(len(X)):
            print('hello')
            # print(X[j])
            # print(y[j])
            # y_hat = np.dot(weights,X[j])
            # error = (y[j])-y_hat
            # weights += (eta*error*(X[j]))
            # print(error)
            # print(weights)



    # for i in range(1):
    #     for index, row_vector in enumerate(vector_list):
    #         y_hat = np.dot(w,row_vector)
    #         error = (train_df.iloc[index,-1])-y_hat
    #         pla = w + (eta*error*row_vector)
    #         print(error)
    #         print(pla)




