import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import *
def adaline():
    w=np.random.uniform(low=0.1,high=0.5,size=(1,3))
    b=np.ones([1,60])
    no_epochs=3
    x_train,x_test,y_train,y_test = DataSplit('sira','bomay')
    # print(x_train)
    # print('-----------------------------------------------------')
    # print(y_train)
    # print('-----------------------------------------------------')
    # print(x_test)
    # print('-----------------------------------------------------')
    # print(y_test)
    # print('-----------------------------------------------------')

    custom_encoding = {'BOMBAY': 0, 'CALI': 1, 'SIRA': 2}
    y_encoded = y_train.map(custom_encoding)
    # print(category_data_encoded)
    train_df = pd.concat([x_train, y_encoded], axis=1)
    df = pd.DataFrame(train_df)

    columns_to_include = ['Area', 'roundnes']
    # Create a list to store vectors for each
    vector_list = []
    for index, row in df.iterrows():

        row_vector =[1]+[row[column] for column in columns_to_include]
        vector_list.append(row_vector)
    print(w)
    for index, row_vector in enumerate(vector_list):
        print(row_vector)
        print(np.dot(w.transpose(),row_vector))
    # for i in range(no_epochs):
    #     for index, row in train_df.iterrows():








