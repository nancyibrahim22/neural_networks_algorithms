import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook
def DataSplit(class1,class2):
    excel_file = pd.read_excel('Dry_Bean_Dataset.xlsx')
    #taking bomay only from the file and and passing it to a function to split x and y
    bomay_list = excel_file.iloc[0:50,:]
    x_bomay, y_bomay = x_and_y(bomay_list)
    x_bomay_train, y_bomay_train, x_bomay_test, y_bomay_test = train_test_split(x_bomay, y_bomay, test_size=0.4,
                                                                                random_state=42)

    #taking cali only from the file and and passing it to a function to split x and y
    cali_list = excel_file.iloc[50:100,:]
    x_cali,y_cali=x_and_y(cali_list)
    x_cali_train, y_cali_train, x_cali_test, y_cali_test = train_test_split(x_cali, y_cali, test_size=0.4,
                                                                                random_state=42)

    #taking sira only from the file and and passing it to a function to split x and y
    sira_list = excel_file.iloc[100:,:]
    x_sira,y_sira=x_and_y(sira_list)
    x_sira_train, y_sira_train, x_sora_test, y_sira_test = train_test_split(x_sira, y_sira, test_size=0.4,
                                                                                random_state=42)



    if class1 == "sira" and class2 =='cali' or class2 == "sira" and class1 =='cali':
        return x_sira_train, y_sira_train, x_sora_test, y_sira_test,x_cali_train, y_cali_train, x_cali_test, y_cali_test
    elif class1=='bomay' and class2=='sira' or class2=='bomay' and class1=='sira':
        return x_bomay_train, y_bomay_train, x_bomay_test, y_bomay_test,x_sira_train, y_sira_train, x_sora_test, y_sira_test
    elif class1=='bomay' and class2=='cali' or class2=='bomay' and class1=='cali':
        return x_bomay_train, y_bomay_train, x_bomay_test, y_bomay_test,x_cali_train, y_cali_train, x_cali_test, y_cali_test


def x_and_y(data):
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return x,y