from utils.preprocessing import *
import numpy as np
import math
from sklearn.preprocessing import StandardScaler,OneHotEncoder
generated_weights=[]


def Backpropagation(hidden_layer_number, neurons_list, eta_entry, epochs_entry, mse_entry, bias_var, activation_var):

    x_train, x_test, y_train, y_test = data_split_back_propagation()
    #scaling data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    #encoding output using one hot encoding
    encoder = OneHotEncoder(sparse=False)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    Y_Train = encoder.fit_transform(y_train)
    Y_Test = encoder.fit_transform(y_test)


    #generating weights for each layer
    for i in range(hidden_layer_number+1):
        if i == 0:
            generated_weights.append(CreateWeights(neurons_list[i], 5,bias_var))
        elif i == hidden_layer_number:
            generated_weights.append(CreateWeights(3, neurons_list[i-1],bias_var))

        else:
            generated_weights.append(CreateWeights(neurons_list[i], neurons_list[i-1],bias_var))

    #looping on training samples
    for epoch in range(epochs_entry):
        for pp in range(len(x_train_scaled)):
            Net_answers = []
            for i in range(hidden_layer_number+1):
                if i ==0:
                    #calculating net values for the input layer
                    Net_answers.append(CalcNet(i,neurons_list[i],x_train_scaled[pp],activation_var,bias_var))
                elif i==hidden_layer_number:
                    # calculating net values for the output layer
                    X=CalcNet(i,3,Net_answers[i-1], activation_var,bias_var)
                    Net_answers.append(X)
                else:
                    # calculating net values for the hidden layers
                    Net_answers.append(CalcNet(i, neurons_list[i], Net_answers[i-1], activation_var,bias_var))
            #making the backward step
            k = backward(activation_var, Y_Train[pp], Net_answers,bias_var)
            #updating weights
            updateWeigths(x_train_scaled[pp],Net_answers, eta_entry, k,bias_var)
    bombay_predbombay_train = 0
    bombay_predcali_train = 0
    bombay_predsira_train = 0
    cali_predbombay_train = 0
    cali_predcali_train = 0
    cali_predsira_train = 0
    sira_predbombay_train = 0
    sira_predcali_train = 0
    sira_predsira_train = 0
    # looping on testing values
    for o in range(len(x_train_scaled)):
        net_answers_train = []

        for i in range(hidden_layer_number + 1):
            if i == 0:
                # calculating net values for the input layer
                net_answers_train.append(CalcNet(i, neurons_list[i], x_train_scaled[o], activation_var, bias_var))
            elif i == hidden_layer_number:
                # calculating net values for the output layer
                X = CalcNet(i, 3, net_answers_train[i - 1], activation_var, bias_var)
                maximum = max(X)
                for i in range(len(X)):
                    if X[i] == maximum:
                        X[i] = 1
                    else:
                        X[i] = 0

                net_answers_train.append(X)
            else:
                # calculating net values for the hidden layers
                net_answers_train.append(CalcNet(i, neurons_list[i], net_answers_train[i - 1], activation_var, bias_var))

        # calculating accuracy of test

        for i in range(3):
            print(X)
            if Y_Train[o][i] == X[i] and X[i] == 1:
                if i == 0:
                    bombay_predbombay_train += 1
                elif i == 1:
                    cali_predcali_train += 1
                elif i == 2:
                    sira_predsira_train += 1
            elif Y_Train[o][i] != X[i] and X[i] == 0 and Y_Train[o][i] == 1:
                if i == 0:
                    if X[i + 1] == 1:
                        bombay_predcali_train += 1
                    if X[i + 2] == 1:
                        bombay_predsira_train += 1
                if i == 1:
                    if X[i - 1] == 1:
                        cali_predbombay_train += 1
                    if X[i + 1] == 1:
                        cali_predsira_train+= 1

                if i == 2:
                    if X[i - 2] == 1:
                        sira_predbombay_train += 1
                    if X[i - 1] == 1:
                        sira_predcali_train += 1
    acc_train = (bombay_predbombay_train + cali_predcali_train + sira_predsira_train) / len(y_train)
    print(acc_train)
    print(bombay_predbombay_train)
    print(bombay_predcali_train)
    print(bombay_predsira_train)
    print(cali_predbombay_train)
    print(cali_predcali_train)
    print(cali_predsira_train)
    print(sira_predbombay_train)
    print(sira_predcali_train)
    print(sira_predsira_train)
    print(bombay_predbombay_train + cali_predcali_train + sira_predsira_train)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    bombay_predbombay = 0
    bombay_predcali = 0
    bombay_predsira = 0
    cali_predbombay = 0
    cali_predcali = 0
    cali_predsira = 0
    sira_predbombay = 0
    sira_predcali = 0
    sira_predsira = 0
    #looping on testing values
    for o in range(len(x_test_scaled)):
        net_answers_test = []

        for i in range(hidden_layer_number + 1):
            if i == 0:
                # calculating net values for the input layer
                net_answers_test.append(CalcNet(i, neurons_list[i], x_test_scaled[o], activation_var,bias_var))
            elif i == hidden_layer_number:
                # calculating net values for the output layer
                X = CalcNet(i, 3, net_answers_test[i - 1], activation_var,bias_var)
                maximum=max(X)
                for i in range(len(X)):
                    if X[i]==maximum:
                        X[i]=1
                    else :
                        X[i]=0

                net_answers_test.append(X)
            else:
                # calculating net values for the hidden layers
                net_answers_test.append(CalcNet(i, neurons_list[i], net_answers_test[i - 1], activation_var,bias_var))



        # calculating accuracy of test

        for i in range(3):
            print(X)
            if Y_Test[o][i] == X[i] and X[i]==1:
                if i==0:
                    bombay_predbombay+=1
                elif i==1:
                    cali_predcali+=1
                elif i==2:
                    sira_predsira+=1
            elif Y_Test[o][i] != X[i] and X[i]==0 and Y_Test[o][i] == 1 :
                if i==0:
                    if X[i+1]==1:
                        bombay_predcali+=1
                    if X[i+2]==1:
                        bombay_predsira+=1
                if i==1:
                    if X[i-1]==1:
                        cali_predbombay+=1
                    if X[i+1]==1:
                        cali_predsira+=1

                if i==2:
                    if X[i-2]==1:
                        sira_predbombay+=1
                    if X[i-1]==1:
                        sira_predcali+=1
    accuracy = (bombay_predbombay + cali_predcali + sira_predsira) / len(y_test)
    print(accuracy)
    print(bombay_predbombay)
    print(bombay_predcali)
    print(bombay_predsira)
    print(cali_predbombay)
    print(cali_predcali)
    print(cali_predsira)
    print(sira_predbombay)
    print(sira_predcali)
    print(sira_predsira)
    print(bombay_predbombay+cali_predcali+sira_predsira)




def CreateWeights(neurons_count, previous_number_of_neurons, bias):

    if bias ==0:
        weights = np.random.uniform(low=0.1, high=0.7, size=(neurons_count, previous_number_of_neurons))
    else:
        weights = np.random.uniform(low=0.1, high=0.7, size=(neurons_count, previous_number_of_neurons+1))

    return weights


def CalcNet(hiden_layer_index, number_of_neurons, previous_layer_neurons_output, activation,bias):

    net_answer = []
    weightsoflayer = generated_weights[hiden_layer_index]
    number_of_PreviousNeurons = len(previous_layer_neurons_output)
    if bias ==0:
        #calculating net values of each neuron
        for neuron in range(number_of_neurons):
            sum1 = 0
            for i in range(number_of_PreviousNeurons):
                sum1 += weightsoflayer[neuron][i]*previous_layer_neurons_output[i]
            if activation == 1:
                sum1 = 1/(1+math.exp(-sum1))
            if activation == 2:
                sum1 = np.tanh(sum1)
            net_answer.append(sum1)
    else:
        for neuron in range(number_of_neurons):
            sum1 = 0
            for i in range(number_of_PreviousNeurons):
                if i==0:
                    sum1+=weightsoflayer[neuron][i]*1
                else:
                    sum1 += weightsoflayer[neuron][i]*previous_layer_neurons_output[i]
            if activation == 1:
                sum1 = 1/(1+math.exp(-sum1))
            if activation == 2:
                sum1 = np.tanh(sum1)
            net_answer.append(sum1)

    return net_answer


def backward(activation, Y, Net_answers,bias):

    result = []
    if activation == 1:

        sigma_output=[]
        #calculating the sigma of each output neuron
        for i in range(3):
            sigma_output.append(((Y[i] - Net_answers[-1][i]) * Net_answers[-1][i]*(1-Net_answers[-1][i])))
        result.insert(0,sigma_output)
        for i in reversed(range(len(Net_answers))):
            if i == len(Net_answers)-1:
                continue
            else:
                sigma_neuron = 0
                sigma_layer = []
                for neuron in range(len(generated_weights[i])):
                    for weight in range(len(generated_weights[i+1])):
                        if bias==0:
                            sigma_neuron += result[0][weight]*generated_weights[i+1][weight][neuron]
                        else:
                            sigma_neuron += result[0][weight] * generated_weights[i + 1][weight][neuron+1]
                    #the result of dot product multiply dserivative of activation function
                    sigma = sigma_neuron * Net_answers[i][neuron]*(1-Net_answers[i][neuron])
                    sigma_layer.append(sigma)
                result.insert(0,sigma_layer)
    elif activation==2:
        sigma_output = []
        # calculating the sigma of each output neuron
        for i in range(3):
            sigma_output.append(((Y[i] - Net_answers[-1][i]) * (1 - (Net_answers[-1][i]**2))))
        result.insert(0, sigma_output)
        for i in reversed(range(len(Net_answers))):
            if i == len(Net_answers) - 1:
                continue
            else:
                sigma_neuron = 0
                sigma_layer = []
                for araf in range(len(generated_weights[i])):
                    for k in range(len(generated_weights[i + 1])):
                        if bias == 0:
                            sigma_neuron += result[0][k] * generated_weights[i + 1][k][araf]
                        else:
                            sigma_neuron += result[0][k] * generated_weights[i + 1][k][araf + 1]
                    # the result of dot product multiply dserivative of activation function
                    sigma = sigma_neuron * (1 - (Net_answers[i][araf]**2))
                    sigma_layer.append(sigma)
                result.insert(0, sigma_layer)

    return result


def updateWeigths(x, net_answers, eta, result_of_backward, bias):
    if bias==0:
        #i=layer j=neuron k=weight
        for i in range(len(generated_weights)):
            for j in range(len(generated_weights[i])):
                for k in range(len(generated_weights[i][j])):
                    if i != 0:

                        generated_weights[i][j][k] += eta * result_of_backward[i][j] * net_answers[i-1][k]
                    else:
                        generated_weights[i][j][k] += eta * result_of_backward[i][j]*x[k]
    else:
        # i=layer j-neuron k=weights
        for i in range(len(generated_weights)):
            for j in range(len(generated_weights[i])):
                for k in range(len(generated_weights[i][j])-1):
                    if i != 0:
                        if k != 0:
                            generated_weights[i][j][k] += eta * result_of_backward[i][j] * net_answers[i - 1][k]
                        else:
                            generated_weights[i][j][k] += eta*result_of_backward[i][j]
                    else:
                        if k != 0:
                            generated_weights[i][j][k] += eta * result_of_backward[i][j] * x[k]
                        else:
                            generated_weights[i][j][k] += eta * result_of_backward[i][j]




