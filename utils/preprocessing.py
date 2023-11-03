import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook
def DataSplit(class1,class2):

    excel_file = pd.read_excel('Dry_Bean_Dataset.xlsx')
    #taking bomay only from the file and and passing it to a function to split x and y
    bomay_list = excel_file.iloc[0:50,:]
    x_bomay, y_bomay = x_and_y(bomay_list)
    x_bomay_train,x_bomay_test, y_bomay_train,  y_bomay_test = train_test_split(x_bomay, y_bomay, test_size=0.4,
                                                                                random_state=42)

    #taking cali only from the file and and passing it to a function to split x and y
    cali_list = excel_file.iloc[50:100,:]
    x_cali,y_cali=x_and_y(cali_list)
    x_cali_train,x_cali_test, y_cali_train,  y_cali_test = train_test_split(x_cali, y_cali, test_size=0.4,
                                                                                random_state=42)

    #taking sira only from the file and and passing it to a function to split x and y
    sira_list = excel_file.iloc[100:,:]
    x_sira,y_sira=x_and_y(sira_list)
    x_sira_train,x_sira_test, y_sira_train,  y_sira_test = train_test_split(x_sira, y_sira, test_size=0.4,
                                                                                random_state=42)



    if class1 == "SIRA" and class2 =='CALI'or class2 == "SIRA" and class1 =='CALI' :
        return merge_class(x_sira_train, x_sira_test, y_sira_train, y_sira_test, x_cali_train, x_cali_test,y_cali_train, y_cali_test)
    elif class1=='BOMBAY' and class2=='SIRA' or class2=='BOMBAY' and class1=='SIRA':
        return merge_class(x_bomay_train, x_bomay_test, y_bomay_train, y_bomay_test, x_sira_train, x_sira_test,y_sira_train, y_sira_test)
    elif class1=='BOMBAY' and class2=='CALI' or class2=='BOMBAY' and class1=='CALI':
        return merge_class(x_bomay_train,x_bomay_test, y_bomay_train,  y_bomay_test, x_cali_train,x_cali_test, y_cali_train,y_cali_test)



def x_and_y(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x,y


def merge_class(x_one_train,x_one_test, y_one_train,  y_one_test,x_two_train,x_two_test, y_two_train,  y_two_test):
    x_train_frames = [x_one_train, x_two_train]
    x_train = pd.concat(x_train_frames)
    y_train_frames = [y_one_train, y_two_train]
    y_train = pd.concat(y_train_frames)
    x_test_frames = [x_one_test, x_two_test]
    x_test = pd.concat(x_test_frames)
    y_test_frames = [y_one_test, y_two_test]
    y_test = pd.concat(y_test_frames)
    return x_train,x_test,y_train,y_test