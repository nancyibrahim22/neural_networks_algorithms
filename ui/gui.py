from tkinter import *
from tkinter import messagebox
from utils.preprocessing import *


def main_GUI():
    global eta_entry, epochs_entry, mse_entry
    master = Tk()
    master.title("Neural Networks Tasks")
    master.geometry("500x500")
    title_label = Label(master, text="Neural Networks Task 1")
    title_label.pack()

    classes_label = Label(master, text="choose 2 classes")
    classes_label.pack()
    classes_var = IntVar()
    Radiobutton(master, text="Bomay, Cali", variable=classes_var, value=1,
                command=lambda: clicked_button(classes_var.get())).pack()
    Radiobutton(master, text="Bomay, Sira", variable=classes_var, value=2,
                command=lambda: clicked_button(classes_var.get())).pack()
    Radiobutton(master, text="Cali, Sira", variable=classes_var, value=3,
                command=lambda: clicked_button(classes_var.get())).pack()

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
    Radiobutton(master, text="Perceptron", variable=algorithm_var, value=1,
                command=lambda: clicked_button(algorithm_var.get())).pack()
    Radiobutton(master, text="Adaline", variable=algorithm_var, value=2,
                command=lambda: clicked_button(algorithm_var.get())).pack()

    button = Button(master, text="Display",command=lambda:display_fun(classes_var.get()))
    button.pack()
    master.mainloop()


def clicked_button(val):
    global num
    num = val

def display_fun(classes_var):
    radio=classes_var
    if radio==1:
      DataSplit('bomay','cali')
    elif radio==2:
        DataSplit('bomay','sira')
    elif radio==3:
        DataSplit('cali','sira')
