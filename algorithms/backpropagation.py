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
    # custom_encoding = {'BOMBAY': 0, 'CALI': 1,'SIRA':2}
    y_train= y_train.values.reshape(-1, 1)
    # y_encoded = y_train.map(custom_encoding)
    # Fit and transform the 'Category' column
    Y_Train = encoder.fit_transform(y_train)



    for i in range(hidden_layer_number+1):
        if i==0:
            generated_weights.append(CreateWeights(neurons_list[i],5))
        elif i==hidden_layer_number:
            generated_weights.append(CreateWeights(3, neurons_list[i-1]))

        else:
            generated_weights.append(CreateWeights(neurons_list[i],neurons_list[i-1]))

    Net_answers = []
    for i in range(hidden_layer_number+1):
        if i ==0:
            Net_answers.append(CalcNet(i,neurons_list[i],x_train_scaled[0],activation_var))
        elif i==hidden_layer_number:
            X=CalcNet(i,3,Net_answers[i-1], activation_var)

            Net_answers.append(X)


        else:
            Net_answers.append(CalcNet(i,neurons_list[i],Net_answers[i-1],activation_var))

    print(backward(activation_var,Y_Train[0],Net_answers))
    print(Net_answers)

    print(generated_weights)
    print(generated_weights[0][1])
    for i in reversed(range(len(Net_answers))):
        print(Net_answers[i])
        print('+++++++++++++')
        print(i)

    return 0
def CreateWeights(neurons_count,previous_number_of_neurons):
    weights = np.random.uniform(low=0.1, high=0.7, size=(neurons_count, previous_number_of_neurons))
    return weights


def CalcNet(hiden_layer_index,number_of_neurons,previous_layer_neurons_output,activation):
    net_answer=[]
    weightsoflayer=generated_weights[hiden_layer_index]
    number_of_PreviousNeurons=len(previous_layer_neurons_output)
    for neuron in range(number_of_neurons):
        Sum=0
        for i in range(number_of_PreviousNeurons):
            Sum+= weightsoflayer[neuron][i]*previous_layer_neurons_output[i]
        if activation==1:
            Sum=1/(1+math.exp(-Sum))
        if activation==2:
            Sum=(1-math.exp(-Sum))/(1+math.exp(-Sum))
        net_answer.append(Sum)
    return net_answer
def backward(activation,Y,Net_answers):
    result=[]
    if activation==1:
        sigma_output=[]
        for i in range(3):
          sigma_output.append(((Y[i] - Net_answers[-1][i])* Net_answers[-1][i]*(1-Net_answers[-1][i])))
        for i in reversed(range(len(Net_answers))):
            if i==len(Net_answers)-1:
                continue
            else:
                sum=0
                for k in range(len(sigma_output[i])):
                    sum+=sigma_output[k]*generated_weights[k][i]













    return sigma_output
def updateWeigths():
    return