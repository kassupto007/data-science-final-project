# Import Library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def models():
    # Loading the dataset
    df = pd.read_csv("./../../Dataset/cleaned_data.csv", index_col=0)
    
    #Adding a new column for years
    df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"])
    df["year"] = df["CRASH DATE"].dt.year
    df["year"].unique()
    
    #Adding a month column for months
    df["month"] = df["CRASH DATE"].dt.month
    df["month"].unique()
    
    # Setting our training and target variables
    X = df.drop(["CRASH DATE", "CRASH TIME", "BOROUGH", "year", "month"], axis=1)
    y = df["BOROUGH"]
    
    #Splitting our dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    #Logistic Regression Model instantiation
    logreg = LogisticRegression()
    
    #Model fitting
    logreg.fit(X_train, y_train)
    
    #Model Prediction
    pred = logreg.predict(X_test)
    
    #Checking our model accuracy
    print("Logistic Regression Model score is: " + str(accuracy_score(y_test, pred)))
    
    #Neural Network
    #Using sklearn to to train a Neural Network (MLP Classifier) on the training set
    nn = MLPClassifier(hidden_layer_sizes=(12, 6), max_iter=10)
    nn.fit(X_train, y_train)

    #Checking our model accuracy
    print("Neural Network Model score is: " + str(nn.score(X_test, y_test)))
    
    #K-Nearest Neighbors
    # Using sklearn to 'train' a k-Neighbors Classifier
    # Note: KNN is a nonparametric model and technically doesn't require training
    # fit will essentially load the data into the model see link below for more information
    k_neighbors = KNeighborsClassifier(n_neighbors=5)
    k_neighbors.fit(X_train, y_train)

    #Checking our model accuracy
    print("K-Nearest Neighbors Model score is: " + str(k_neighbors.score(X_test, y_test)))
    
    #Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    print(cm)

    fig, ax = plt.subplots(figsize=(15,8))
    plot_confusion_matrix(logreg, X_test, y_test, ax=ax)
    plt.grid(False)
    plt.show()
    
    #Classification report of our model
    print(classification_report(y_test, pred))

models()