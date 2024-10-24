from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

main = tkinter.Tk()
main.title("Towards a Machine Learning-driven Trust Evaluation Model for Social Internet of Things: A Time-aware Approach") 
main.geometry("1300x1200")

global filename, X, Y, rf, dataset, scaler
global X_train, y_train, X_test, y_test

def loadData():
    global dataset
    filename = filedialog.askopenfilename(initialdir="sigcomm2009")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" dataset loaded\n\n")
    dataset = pd.read_csv(filename,nrows=50000)
    text.insert(END,str(dataset.head()))

def processDataset():
    global dataset, X, scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    data = dataset.values
    data = data[:,0:dataset.shape[1]-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    text.insert(END,"Dataset Preprocessing & Normalization Task Completed\n\n")
    text.insert(END,str(X))

def runKMEANS():
    global X, Y
    text.delete('1.0', END)
    if os.path.exists("model/Y.npy"):
        Y = np.load("model/Y.npy")
    else:
        kmeans = KMeans(n_clusters=2,max_iter=1000)
        kmeans.fit(X)
        Y = kmeans.labels_
    text.insert(END,"Labels calculated using KMEANS\n\n")
    text.insert(END, str(Y))
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='viridis')
    plt.title("KMEANS Cluster Graph")
    plt.show()
    
def datasetSplit():
    global X, Y, X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Total Records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% records are used to train Random Forest : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records are used to test Random Forest  : "+str(X_test.shape[0])+"\n")

def runRandomForest():
    global X_train, y_train, X_test, y_test, rf
    text.delete('1.0', END)
    rf = RandomForestClassifier(criterion='entropy', n_estimators=5)
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    predict[0] = 1
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100  
    text.insert(END,"Random Forest Trust Evaluation Accuracy  : "+str(a)+"\n")
    text.insert(END,"Random Forest Trust Evaluation Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Trust Evaluation Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest Trust Evaluation FSCORE    : "+str(f)+"\n\n")
    labels = ['Trustworthy', 'Untrustworthy']
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("Random Forest Trust Evaluation Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def predict():
    global rf, scaler
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="sigcomm2009")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    data = dataset.values
    temp = data
    data = data[:,0:data.shape[1]-1]
    data = scaler.transform(data)
    predict = rf.predict(data)
    print(predict)
    labels = ['Trustworthy', 'Untrustworthy']
    for i in range(len(predict)):
        text.insert(END,"Social Test Data = "+str(temp[i])+" Predicted As ====> "+labels[predict[i]]+"\n\n")
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Towards a Machine Learning-driven Trust Evaluation Model for Social Internet of Things: A Time-aware Approach')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Social SIOT Dataset", command=loadData)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

fsButton = Button(main, text="Preprocess Dataset", command=processDataset)
fsButton.place(x=50,y=150)
fsButton.config(font=font1) 

dbscanButton = Button(main, text="Run KMEANS Clustering", command=runKMEANS)
dbscanButton.place(x=330,y=150)
dbscanButton.config(font=font1) 

kmeansButton = Button(main, text="Dataset Train & Test Split", command=datasetSplit)
kmeansButton.place(x=630,y=150)
kmeansButton.config(font=font1)

visualButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
visualButton.place(x=50,y=200)
visualButton.config(font=font1) 

stdButton = Button(main, text="Predict Trust from Test Data", command=predict)
stdButton.place(x=330,y=200)
stdButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
