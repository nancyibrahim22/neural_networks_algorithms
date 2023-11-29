from utils.preprocessing import *
import numpy as np
import math
from sklearn.preprocessing import StandardScaler,OneHotEncoder
generated_weights=[]


def Backpropagation(hidden_layer_number, neurons_list, eta_entry, epochs_entry, mse_entry, bias_var, activation_var):

    x_train, x_test, y_train, y_test = data_split_back_propagation()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    x_test_scaled = scaler.fit_transform(x_test)
    encoder = OneHotEncoder(sparse=False)

    y_train= y_train.values.reshape(-1, 1)
    y_test=y_test.values.reshape(-1,1)

    Y_Train = encoder.fit_transform(y_train)
    Y_Test = encoder.fit_transform(y_test)



    for i in range(hidden_layer_number+1):
        if i == 0:
            generated_weights.append(CreateWeights(neurons_list[i], 5,bias_var))
        elif i == hidden_layer_number:
            generated_weights.append(CreateWeights(3, neurons_list[i-1],bias_var))

        else:
            generated_weights.append(CreateWeights(neurons_list[i], neurons_list[i-1],bias_var))

    print(generated_weights)
    for epoch in range(epochs_entry):
        for pp in range(90):
            Net_answers = []
            for i in range(hidden_layer_number+1):
                if i ==0:
                    Net_answers.append(CalcNet(i,neurons_list[i],x_train_scaled[pp],activation_var,bias_var))
                elif i==hidden_layer_number:
                    X=CalcNet(i,3,Net_answers[i-1], activation_var,bias_var)

                    Net_answers.append(X)
                else:
                    Net_answers.append(CalcNet(i, neurons_list[i], Net_answers[i-1], activation_var,bias_var))
            k = backward(activation_var, Y_Train[pp], Net_answers,bias_var)
            updateWeigths(x_train_scaled[pp],Net_answers, eta_entry, k,bias_var)

    # Net_answers = []
    # for i in range(hidden_layer_number+1):
    #     if i ==0:
    #         Net_answers.append(CalcNet(i,neurons_list[i],x_train_scaled[0],activation_var))
    #     elif i==hidden_layer_number:
    #          X=CalcNet(i,3,Net_answers[i-1], activation_var)
    #
    #          Net_answers.append(X)
    #     else:
    #          Net_answers.append(CalcNet(i, neurons_list[i], Net_answers[i-1], activation_var))
    #
    # k = backward(activation_var, Y_Train[0], Net_answers)
    # print(x_train_scaled[0])
    # print(generated_weights)
    # print(Net_answers)
    # print('++++++++++')
    # updateWeigths(x_train_scaled[0],Net_answers, eta_entry, k)
    # print(generated_weights)
    # print('++++++++++')
    #
    # print('++++++++++')
    # print(k)
    # print('++++++++++')

    acc = 0
    for o in range(60):
        Net_answers2 = []

        for i in range(hidden_layer_number + 1):
            if i == 0:
                Net_answers2.append(CalcNet(i, neurons_list[i], x_test_scaled[o], activation_var,bias_var))
            elif i == hidden_layer_number:
                X = CalcNet(i, 3, Net_answers2[i - 1], activation_var,bias_var)
                maximum=max(X)
                for i in range(len(X)):
                    if X[i]==maximum:
                        X[i]=1
                    else :
                        X[i]=0

                Net_answers2.append(X)
            else:
                Net_answers2.append(CalcNet(i, neurons_list[i], Net_answers2[i - 1], activation_var,bias_var))
        faloga=True
        for i in range(3):
            if Y_Test[o][i]==X[i]:
                faloga=True
            else:
                faloga=False
        if faloga:
            acc+=1
        print(X)

    print(acc)

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
        for i in range(3):
            sigma_output.append(((Y[i] - Net_answers[-1][i]) * Net_answers[-1][i]*(1-Net_answers[-1][i])))
        result.insert(0,sigma_output)
        for i in reversed(range(len(Net_answers))):
            if i == len(Net_answers)-1:
                continue
            else:
                sum1 = 0
                listaya = []
                for araf in range(len(generated_weights[i])):
                    for k in range(len(generated_weights[i+1])):
                        if bias==0:
                            sum1 += result[0][k]*generated_weights[i+1][k][araf]
                        else:
                            sum1 += result[0][k] * generated_weights[i + 1][k][araf+1]
                    net = sum1 * Net_answers[i][araf]*(1-Net_answers[i][araf])
                    listaya.append(net)
                result.insert(0,listaya)
    elif activation==2:
        sigma_output = []
        for i in range(3):
            sigma_output.append(((Y[i] - Net_answers[-1][i]) * (1 - (Net_answers[-1][i]**2))))
        result.insert(0, sigma_output)
        for i in reversed(range(len(Net_answers))):
            if i == len(Net_answers) - 1:
                continue
            else:
                sum1 = 0
                listaya = []
                for araf in range(len(generated_weights[i])):
                    for k in range(len(generated_weights[i + 1])):
                        if bias == 0:
                            sum1 += result[0][k] * generated_weights[i + 1][k][araf]
                        else:
                            sum1 += result[0][k] * generated_weights[i + 1][k][araf + 1]
                    net = sum1 * (1 - (Net_answers[i][araf]**2))
                    listaya.append(net)
                result.insert(0, listaya)


    return result


def updateWeigths(X,Net_answers, eta, result_of_backward,bias):
    if bias==0:
        for i in range(len(generated_weights)):
            for j in range(len(generated_weights[i])):
                for k in range(len(generated_weights[i][j])):
                    if i != 0:
                        l=result_of_backward[i][j]

                        q=Net_answers[i-1][k]
                        generated_weights[i][j][k] += eta * l * q
                    else:
                        generated_weights[i][j][k] += eta * result_of_backward[i][j]*X[k]
    else:
        for i in range(len(generated_weights)):
            for j in range(len(generated_weights[i])):
                for k in range(len(generated_weights[i][j])-1):
                    if i != 0:
                        if k!=0:
                            l = result_of_backward[i][j]
                            q = Net_answers[i - 1][k]
                            generated_weights[i][j][k] += eta * l * q
                        else:
                            generated_weights[i][j][k] +=eta*result_of_backward[i][j]
                    else:
                        if k!=0:


                            generated_weights[i][j][k] += eta * result_of_backward[i][j] * X[k]
                        else:
                            generated_weights[i][j][k] += eta * result_of_backward[i][j]




