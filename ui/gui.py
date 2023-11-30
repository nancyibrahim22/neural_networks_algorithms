from tkinter import *
from tkinter import messagebox
from algorithms.adaline import *
from algorithms.perceptron import *
from algorithms.backpropagation import *


def main_GUI():
    # global eta_entry, epochs_entry, mse_entry
    master = Tk()
    master.title("Neural Networks Tasks")
    master.geometry("500x800")
    title_label = Label(master, text="Neural Networks Task 1")
    title_label.config(font=("Arial", 14), fg="red")
    title_label.grid(row=1, column=5)

    features_label = Label(master,text="Choose 2 features")
    features_label.config(font=("Arial", 10), fg="blue")
    features_label.grid(row=2,column=5)

    first_features_var = IntVar()
    Radiobutton(master, text="Area",variable=first_features_var,value=0).grid(row=3,column=4)
    Radiobutton(master, text="Perimeter", variable=first_features_var, value=1).grid(row=4,column=4)
    Radiobutton(master, text="MajorAxisLength", variable=first_features_var, value=2).grid(row=5,column=4)
    Radiobutton(master, text="MinorAxisLength", variable=first_features_var, value=3).grid(row=6,column=4)
    Radiobutton(master, text="roundnes", variable=first_features_var, value=4).grid(row=7,column=4)

    second_features_var = IntVar()
    Radiobutton(master, text="Area", variable=second_features_var, value=0).grid(row=3,column=6)
    Radiobutton(master, text="Perimeter", variable=second_features_var, value=1).grid(row=4,column=6)
    Radiobutton(master, text="MajorAxisLength", variable=second_features_var, value=2).grid(row=5,column=6)
    Radiobutton(master, text="MinorAxisLength", variable=second_features_var, value=3).grid(row=6,column=6)
    Radiobutton(master, text="roundnes", variable=second_features_var, value=4).grid(row=7,column=6)

    classes_label = Label(master, text="Choose 2 classes")
    classes_label.config(font=("Arial", 10), fg="blue")
    classes_label.grid(row=8,column=5)
    classes_var = IntVar()
    Radiobutton(master, text="Bombay, Cali", variable=classes_var, value=1).grid(row=9,column=5)
    Radiobutton(master, text="Bombay, Sira", variable=classes_var, value=2).grid(row=10,column=5)
    Radiobutton(master, text="Cali, Sira", variable=classes_var, value=3).grid(row=11,column=5)

    eta_label = Label(master, text="enter learning rate")
    eta_label.config(font=("Arial", 10), fg="blue")
    eta_label.grid(row=12,column=4)
    eta_entry = Entry(master, width=20, borderwidth=3)
    eta_entry.grid(row=12,column=5)

    epochs_label = Label(master, text="enter number of epochs")
    epochs_label.config(font=("Arial", 10), fg="blue")
    epochs_label.grid(row=13,column=4)
    epochs_entry = Entry(master, width=20, borderwidth=3)
    epochs_entry.grid(row=13,column=5)

    mse_label = Label(master, text="enter MSE threshold")
    mse_label.config(font=("Arial", 10), fg="blue")
    mse_label.grid(row=14,column=4)
    mse_entry = Entry(master, width=20, borderwidth=3)
    mse_entry.grid(row=14,column=5)

    bias_var = IntVar()
    bias_check = Checkbutton(master, text="Bias", variable=bias_var)
    bias_check.grid(row=15,column=5)

    algorithm_label = Label(master, text="Choose the used algorithm")
    algorithm_label.config(font=("Arial", 10), fg="blue")
    algorithm_label.grid(row=16,column=5)
    algorithm_var = IntVar()
    Radiobutton(master, text="Perceptron", variable=algorithm_var, value=1).grid(row=17,column=5)
    Radiobutton(master, text="Adaline", variable=algorithm_var, value=2).grid(row=18,column=5)
    Radiobutton(master, text="Back-Propagation", variable=algorithm_var, value=3).grid(row=19, column=5)

    activation_fun_label = Label(master, text="Choose the used activation function")
    activation_fun_label.config(font=("Arial", 10), fg="blue")
    activation_fun_label.grid(row=20, column=5)
    activation_fun_var = IntVar()
    Radiobutton(master, text="Sigmoid", variable=activation_fun_var, value=1).grid(row=21, column=5)
    Radiobutton(master, text="Hyperbolic", variable=activation_fun_var, value=2).grid(row=22, column=5)


    hidden_layers_label = Label(master, text="Enter number of hidden layers: ")
    hidden_layers_label.grid(row=23,column=5)

    hidden_layers_entry = Entry(master, width=20, borderwidth=3)
    hidden_layers_entry.grid(row=24,column=5)

    hidden_neurons_frame = Frame(master)
    hidden_neurons_frame.grid(row=25,column=5)


    hidden_layers_button = Button(master, text="Enter", command=lambda:create_hidden_neurons_entries(hidden_layers_entry.get(),hidden_neurons_frame,button))
    hidden_layers_button.grid(row=26,column=5)
    # Initially hide the button

    button = Button(master, text="Display",
                    command=lambda: display_fun(master, first_features_var.get(), second_features_var.get(),
                                                classes_var.get(), algorithm_var.get(),
                                                eta_entry.get(), epochs_entry.get(), mse_entry.get(), bias_var.get(),
                                                hidden_layers_entry.get(),activation_fun_var.get()))
    button.config(font=("Arial", 12), fg="white", bg="red")
    button.grid(row=27, column=5)

    master.mainloop()


def create_hidden_neurons_entries(num_hidden_layers, hidden_neurons_frame, button):
    if num_hidden_layers == "":
        messagebox.showerror('Hidden layers Error', 'Error: You must add the numbers of hidden layers')
    else:
        for widget in hidden_neurons_frame.winfo_children():
            widget.destroy()

        global hidden_neurons_entries
        hidden_neurons_entries = []

        for i in range(int(num_hidden_layers)):
            label = Label(hidden_neurons_frame, text=f"Neurons in Layer {i + 1}: ")
            label.grid(row=i, column=0, padx=5, pady=5)

            entry = Entry(hidden_neurons_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            hidden_neurons_entries.append(entry)

        button.grid(row=26, column=5)


def display_fun(TK, first_features_var, second_features_var, classes_var, algorithms_var, eta_entry,
                epochs_entry, mse_entry, bias_var, num_hidden_layers, activation_var):
    first_feature_radio = first_features_var
    second_feature_radio = second_features_var
    class_radio = classes_var
    algorithms_radio = algorithms_var
    if algorithms_radio == 0:
        messagebox.showerror('Algorithm Error', 'Error: You must choose an algorithm')
    elif algorithms_radio == 1 or algorithms_radio == 2:
        if first_features_var == second_features_var:
            messagebox.showerror('Features Error', 'Error: You must choose different features')
        elif classes_var == 0:
            messagebox.showerror('Classes Error', 'Error: You must choose 2 classes')
        elif eta_entry == "":
            messagebox.showerror('Learning rate Error', 'Error: You must add a valid learning rate value')
        elif epochs_entry == "":
            messagebox.showerror('epochs Error', 'Error: You must add a valid epochs value')
        elif mse_entry == "":
            messagebox.showerror('MSE threshold Error', 'Error: You must add a valid MSE threshold value')
        else:
            if class_radio == 1 and algorithms_radio == 1:
                result = Perceptron(first_feature_radio,second_feature_radio,'BOMBAY','CALI',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28,column=5)
            elif class_radio == 2 and algorithms_radio == 1:
                result = Perceptron(first_feature_radio,second_feature_radio,'BOMBAY','SIRA',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28, column=5)
            elif class_radio == 3 and algorithms_radio == 1:
                result = Perceptron(first_feature_radio,second_feature_radio,'CALI','SIRA',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28, column=5)
            elif class_radio == 1 and algorithms_radio == 2:
                result = Adaline(first_feature_radio,second_feature_radio,'BOMBAY','CALI',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28, column=5)
            elif class_radio == 2 and algorithms_radio == 2:
                result = Adaline(first_feature_radio,second_feature_radio,'BOMBAY','SIRA',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28, column=5)
            elif class_radio == 3 and algorithms_radio == 2:
                result = Adaline(first_feature_radio,second_feature_radio,'CALI','SIRA',float(eta_entry),int(epochs_entry),float(mse_entry),int(bias_var))
                accuracy_label = Label(TK, text=f'Accuracy = {result}')
                accuracy_label.config(font=("Arial", 12), fg="blue")
                accuracy_label.grid(row=28, column=5)
    else:
        if num_hidden_layers == "":
            messagebox.showerror('Hidden layers Error', 'Error: You must add the numbers of hidden layers')
        neurons_in_layers = []
        num_hidden_layers = int(num_hidden_layers)
        for i in range(num_hidden_layers):
            neuron_entry = hidden_neurons_entries[i].get()
            if neuron_entry.strip() != '':
                neurons_in_layers.append(int(neuron_entry))
            else:
                # Handle empty entry here, such as skipping or providing a default value
                # For now, let's just append a placeholder value of 0
                neurons_in_layers.append(0)
        if eta_entry == "":
            messagebox.showerror('Learning rate Error', 'Error: You must add a valid learning rate value')
        elif epochs_entry == "":
            messagebox.showerror('epochs Error', 'Error: You must add a valid epochs value')
        elif activation_var == 0:
            messagebox.showerror('Activation function Error', 'Error: You must choose activation function')
        else:
            print("num_hidden_layers", num_hidden_layers)
            print("neurons_in_layers", neurons_in_layers)
            print("eta_entry", eta_entry)
            print("epochs_entry", epochs_entry)
            print("bias_var", bias_var)
            print("activation_var", activation_var)
            result = Backpropagation(num_hidden_layers, neurons_in_layers, float(eta_entry),
                             int(epochs_entry), int(bias_var), activation_var)
            accuracy_label = Label(TK, text=f'Accuracy = {result}')
            accuracy_label.config(font=("Arial", 12), fg="blue")
            accuracy_label.grid(row=27, column=5)

