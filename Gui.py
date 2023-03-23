import tkinter as tk
import numpy as np

window = tk.Tk()
bias = np.random.random()

window.title('Deep learning model')
window.geometry("600x350")

mylabel = tk.Label(window, text="Choose Features").place(x=15, y=30)
options = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']


def valueRemove(selection):
    options.remove(selection)
    tk.OptionMenu(window, selected2, *options).place(x=450, y=30)


selected1 = tk.StringVar()
selected1.set('None')
selected2 = tk.StringVar()
selected2.set('None')

tk.OptionMenu(window, selected1, *options, command=valueRemove).place(x=150, y=30)

mylabel = tk.Label(window, text="Choose Classes").place(x=15, y=100)
options2 = ['Adelie', 'Gentoo', 'Chinstrap']


def valueRemove2(selection):
    options2.remove(selection)
    tk.OptionMenu(window, selected4, *options2).place(x=450, y=100)


selected3 = tk.StringVar()
selected3.set('None')
selected4 = tk.StringVar()
selected4.set('None')

tk.OptionMenu(window, selected3, *options2, command=valueRemove2).place(x=150, y=100)

# Learning Rate Entry
eta = tk.DoubleVar()
tk.Label(window, text="Enter Learning Rate ").place(x=15, y=175)
eta_entry = tk.Entry(window)
eta_entry.place(x=160, y=175)

# Epoches Entry
epoches = tk.IntVar()
tk.Label(window, text="Enter number of epochs ").place(x=15, y=220)
epoches_entry = tk.Entry(window)
epoches_entry.place(x=160, y=220)

# Bias Checkbox
bias = tk.BooleanVar()


def check_changed():
    print(bias.get())


tk.Checkbutton(window, text='Check for Bias', command=check_changed, variable=bias, onvalue=True, offvalue=False).place(
    x=15, y=265)


def call_back(e1, e2):
    e1.set(eta_entry.get())
    e2.set(epoches_entry.get())
    window.destroy()


tk.Button(window, text='Start', background="green", width=20, command=lambda: call_back(eta, epoches)).place(x=250, y=300)
window.mainloop()
