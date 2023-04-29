import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sns
data=pd.read_csv('C:/Users/STEVE/Downloads/diabetes.csv')
print(data.head())
print(data.tail())
print(data.isnull().sum())
print(data.dtypes)
X = data.drop('Outcome', axis = 1)
y = data['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.20, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
y_prd2 = reg.predict(X_train)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))

#sns.countplot(x = 'Outcome',data = data)
#sns.pairplot(data = data, hue = 'Outcome')
#plt.show()
#sns.heatmap(data.corr(), annot = True)
#plt.show()
#fig, ax = plt.subplots(figsize = (15, 10))
#sns.boxplot(data = data, width = 0.5, ax = ax, fliersize = 3)
#plt.show()
#plt.figure(figsize = (16, 8))
#corr = data.corr()
#mask = np.triu(np.ones_like(corr, dtype = bool))
#sns.heatmap(corr, mask = mask, annot = True, fmt = '.2g', linewidths = 1)
#plt.show()

import tkinter as tk

root= tk.Tk()
#Make a Canvas (i.e, a screen for your project
canvas1 = tk.Canvas(root, width = 500, height = 350)
canvas1.pack()
canvas1.configure(bg='peach puff')
#To see the GUI screen

label = tk.Label(root, text= 'Diabetes Prediction App', bg='Yellow')
canvas1.create_window(250, 50,window=label)

label1 = tk.Label(root, text=' Blood Pressure : ')
canvas1.create_window(100, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)


label2 = tk.Label(root, text='  Pregnancies: ')
canvas1.create_window(100, 120, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)


label3 = tk.Label(root, text=' Glucose: ')
canvas1.create_window(100, 140, window=label3)
entry3 = tk.Entry (root) # create 3nd entry box
canvas1.create_window(270, 140, window=entry3)

label4 = tk.Label(root, text=' Insulin: ')
canvas1.create_window(100, 160, window=label4)
entry4 = tk.Entry (root) # create 4nd entry box
canvas1.create_window(270, 160, window=entry4)

label5 = tk.Label(root, text=' Skin Thickness: ')
canvas1.create_window(100, 180, window=label5)
entry5 = tk.Entry (root) # create 5nd entry box
canvas1.create_window(270, 180, window=entry5)

label6= tk.Label(root, text=' Age: ')
canvas1.create_window(100, 200, window=label6)
entry6 = tk.Entry (root) # create 6nd entry box
canvas1.create_window(270, 200, window=entry6)

label7 = tk.Label(root, text=' BMI: ')
canvas1.create_window(100, 220, window=label7)
entry7 = tk.Entry (root) # create 7nd entry box
canvas1.create_window(270, 220, window=entry7)

label8 = tk.Label(root, text=' DiabetesPedrieeFunction: ')
canvas1.create_window(100, 240, window=label8)
entry8 = tk.Entry (root) # create 8nd entry box
canvas1.create_window(270, 240, window=entry8)


def values():
    BloodPressure = float(entry1.get())
    Pregnancies = float(entry2.get())
    Glucose = float(entry3.get())
    Insulin = float(entry4.get())
    SkinThickness = float(entry5.get())
    Age = float(entry6.get())
    BMI = float(entry7.get())
    DiabetesPedrieeFunction = float(entry8.get())
    Prediction_result = reg.predict([[BloodPressure ,Pregnancies, Glucose, Insulin, SkinThickness, Age, BMI, DiabetesPedrieeFunction]])
    if list(Prediction_result)==0:
        label_Prediction = tk.Label(root, text='You have no Diabetes 😊',bg='pink')
        canvas1.create_window(200, 300, window=label_Prediction)
    else:
        label_Prediction = tk.Label(root, text= 'You have Diabetes 😰',bg='pink')
        canvas1.create_window(200, 300, window=label_Prediction)


# Please add prediction code by yourselves i.e, dtree .predict() line before label_prediction.
label_Prediction = tk.Button(root, text= 'Predict If You have Diabetes or not',command=values, bg='gold')
canvas1.create_window(220, 270,window=label_Prediction)

#Important Links

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#figure3 = plt.Figure(figsize=(5,4), dpi=100)
#ax3 = figure3.add_subplot(111)

fig, axes = plt.subplots(1, 2, figsize=(15, 3))
sns.countplot(x = 'Outcome',data = data,ax=axes[0])
sns.boxplot(data = data, width = 0.5,fliersize = 3,ax=axes[1])

plt.title("Countplot using tkinter and seaborn")
plt.xlabel("Data Values")
plt.ylabel("Count")

canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)



root.mainloop()
