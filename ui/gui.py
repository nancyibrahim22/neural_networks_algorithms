from tkinter import *
from tkinter import messagebox
from utils.preprocessing import *
from algorithms.adaline import *
from algorithms.perceptron import *


def main_GUI():
    global eta_entry, epochs_entry, mse_entry
    master = Tk()
    master.title("Neural Networks Tasks")
    master.geometry("1000x1000")
    title_label = Label(master, text="Neural Networks Task 1")
    title_label.pack()

    features_label = Label(master,text="Choose 2 features")
    features_label.pack()
    first_features_var = IntVar()
    Radiobutton(master, text="Area",variable=first_features_var,value=0).pack()
    Radiobutton(master, text="Perimeter", variable=first_features_var, value=1).pack()
    Radiobutton(master, text="MajorAxisLength", variable=first_features_var, value=2).pack()
    Radiobutton(master, text="MinorAxisLength", variable=first_features_var, value=3).pack()
    Radiobutton(master, text="roundnes", variable=first_features_var, value=4).pack()

    second_features_var = IntVar()
    Radiobutton(master, text="Area", variable=second_features_var, value=0).pack()
    Radiobutton(master, text="Perimeter", variable=second_features_var, value=1).pack()
    Radiobutton(master, text="MajorAxisLength", variable=second_features_var, value=2).pack()
    Radiobutton(master, text="MinorAxisLength", variable=second_features_var, value=3).pack()
    Radiobutton(master, text="roundnes", variable=second_features_var, value=4).pack()

    classes_label = Label(master, text="choose 2 classes")
    classes_label.pack()
    classes_var = IntVar()
    Radiobutton(master, text="Bombay, Cali", variable=classes_var, value=1).pack()
    Radiobutton(master, text="Bombay, Sira", variable=classes_var, value=2).pack()
    Radiobutton(master, text="Cali, Sira", variable=classes_var, value=3).pack()

    eta_label = Label(master, text="enter learning rate")
    eta_label.pack()
    eta_entry = Entry(master, width=20, borderwidth=3)
    eta_entry.pack()

    epochs_label = Label(master, text="enter number of epochs")
    epochs_label.pack()
    epochs_entry = Entry(master, width=20, borderwidth=3)
    epochs_entry.pack()

    mse_label = Label(master, text="enter MSE threshold")
    mse_label.pack()
    mse_entry = Entry(master, width=20, borderwidth=3)
    mse_entry.pack()

    bias_var = IntVar()
    bias_check = Checkbutton(master, text="Bias", variable=bias_var)
    bias_check.pack()

    algorithm_label = Label(master, text="choose the used algorithm")
    algorithm_label.pack()
    algorithm_var = IntVar()
    Radiobutton(master, text="Perceptron", variable=algorithm_var, value=1).pack()
    Radiobutton(master, text="Adaline", variable=algorithm_var, value=2).pack()

    button = Button(master, text="Display",command=lambda:display_fun(first_features_var.get(),second_features_var.get(), classes_var.get(),algorithm_var.get(),
                                                                      eta_entry.get(),epochs_entry.get(),mse_entry.get(), bias_var.get()))
    button.pack()
    master.mainloop()


def display_fun(first_features_var, second_features_var, classes_var, algorithms_var, eta_entry, epochs_entry, mse_entry, bias_var):
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
    elif algorithms_var == 0:
        messagebox.showerror('Algorithm Error', 'Error: You must choose an algorithm')
    else:
        print(first_features_var)
        print('---------------------------')
        print(second_features_var)
        print('---------------------------')
        print(classes_var)
        print('---------------------------')
        print(algorithms_var)
        print('---------------------------')
        print(eta_entry)
        print('---------------------------')
        print(epochs_entry)
        print('---------------------------')
        print(mse_entry)
        print('---------------------------')
        print(bias_var)
        first_feature_radio = first_features_var
        second_feature_radio = second_features_var
        class_radio = classes_var
        algorithms_radio = algorithms_var
        if class_radio == 1 and algorithms_radio == 1:
            Perceptron(first_feature_radio,second_feature_radio,'BOMBAY','CALI',eta_entry,epochs_entry,mse_entry,bias_var)
        elif class_radio == 2 and algorithms_radio == 1:
            Perceptron(first_feature_radio,second_feature_radio,'BOMBAY','SIRA', eta_entry, epochs_entry, mse_entry,bias_var)
        elif class_radio == 3 and algorithms_radio == 1:
            Perceptron(first_feature_radio,second_feature_radio,'CALI','SIRA', eta_entry, epochs_entry, mse_entry,bias_var)
        elif class_radio == 1 and algorithms_radio == 2:
            Adaline(first_feature_radio,second_feature_radio,'BOMBAY','CALI',eta_entry,epochs_entry,mse_entry,bias_var)
        elif class_radio == 2 and algorithms_radio == 2:
            Adaline(first_feature_radio,second_feature_radio,'BOMBAY','SIRA',eta_entry,epochs_entry,mse_entry,bias_var)
        elif class_radio == 3 and algorithms_radio == 2:
            Adaline(first_feature_radio,second_feature_radio,'CALI','SIRA',eta_entry,epochs_entry,mse_entry,bias_var)
