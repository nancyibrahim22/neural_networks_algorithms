from utils.preprocessing import *
def Backpropagation(hidden_layer_number, neurons_list, eta_entry, epochs_entry, mse_entry, bias_var, activation_var):
    print('write your code here')
    # Calling DataSplit function
    x_train, x_test, y_train, y_test = data_split_back_propagation()
    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)
    return 0
