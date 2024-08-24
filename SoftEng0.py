import csv
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import missingno

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import recall_score, accuracy_score,roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
import shap 


# Loading the dataset
# heartds0 = pd.read_csv('https://raw.githubusercontent.com/orionmc/SoftEng0/main/heart0.csv')
heartds0 = pd.read_csv(r'C:\Users\user0\Documents\BPP\Software Engineering\Git\0\SoftEng0\SoftEng0\heart.ds.csv')

# For quick overview of the data we can use either tail or head functions
print("\nLast 6 rows\n",heartds0.tail(6),sep=os.linesep)
print("\nFirst 6 rows\n",heartds0.head(6),sep=os.linesep)

print("\nPrint the shape of the dataset - number of rows and columns:",heartds0.shape)

# check for missing values in the dataset
print("\nSum of null values for each column\n",heartds0.isnull().sum())   
# General information about the data set
heartds0.info()

# for better visuall demonstration we can use a bar chart to visuallise
missingno.bar(heartds0, color = "c")
plt.show()

# rectify the missing values in the dataset can be done in multiple ways
# we can simply use dropna() to drop the rows with missing values
# heartds_no_missing = heartds0.dropna() 
# as we can see from the above function's output - we haven't got any missing values

heartds0[heartds0.duplicated()]            # check for duplicate rows in the dataset
heartds0.drop_duplicates(inplace=True)     # drop the duplicate rows in the dataset

# Print basic stat
print("\nBasic Stat\n",heartds0.describe(),sep=os.linesep)
sns.countplot(x='sex', data=heartds0)
plt.show()

# Grouping values by their type
numberic_var=['age','chol','trtbps','thalachh','old_peak','caa']
cat_var=['sex','fbs','exng','output','cp','rest_ecg','slp','thall']


mypal = ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']
#ax = sns.countplot(x=heartds0['output'], palette=mypal[1::4])
ax = sns.countplot(x=heartds0['output'], hue=heartds0['output'], palette=mypal[1::4], legend=False)


def label_encode_cat_features(heartds0, cat_var):
# Given a dataframe and its categorical features, # this function returns label-encoded dataframe
    label_encoder = LabelEncoder()
    heartds0_encoded = heartds0.copy()
    for col in cat_var:
        heartds0_encoded[col] = label_encoder.fit_transform(heartds0[col])
    heartds0 = heartds0_encoded
    return heartds0

def score_summary(names, classifiers):
    # For the list of classiers, this function calculates the accuracy, 
    # ROC_AUC and Recall and returns the values in a dataframe
        
    cols=["Classifier", "Accuracy", "ROC_AUC", "Recall", "Precision", "F1"]
    heartds0_table = pd.DataFrame(columns=cols)
    
    for name, clf in zip(names, classifiers):        
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, pred)

        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)        
        roc_auc = auc(fpr, tpr)
        
        # confusion matric, cm
        cm = confusion_matrix(y_val, pred) 
        
        # recall: TP/(TP+FN)
        recall = cm[1,1]/(cm[1,1] +cm[1,0])
        
        # precision: TP/(TP+FP)
        precision = cm[1,1]/(cm[1,1] +cm[0,1])
        
        # F1 score: TP/(TP+FP)
        f1 = 2*recall*precision/(recall + precision)

        df = pd.DataFrame([[name, accuracy*100, roc_auc, recall, precision, f1]], columns=cols)
        # heartds0_table = heartds0_table.concat(df)     
        heartds0_table = pd.concat([heartds0_table, df])

    return(np.round(heartds0_table.reset_index(drop=True), 2))

def plot_conf_matrix(names, classifiers, nrows, ncols, fig_a, fig_b):
    # Plots matric of confusion matrices
    # Arguments:
    #     names         : list of names of the classifier
    #     classifiers   : list of classification algorithms
    #     nrows, ncols  : number of rows and rows in the plots
    #     fig_a, fig_b  : dimensions of the figure
       
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_a, fig_b))
    
    for i, (clf, ax) in enumerate(zip(classifiers, axes.flatten())):
        
        clf.fit(X_train, y_train)  
        # Compute confusion matrix
        pred = clf.predict(X_val)
        cm = confusion_matrix(y_val, pred)
        # Create ConfusionMatrixDisplay object
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        
        # Plot the confusion matrix on the given axis
        disp.plot(ax=ax)
        ax.title.set_text(names[i]) 
      
    plt.tight_layout() 
    plt.show()
def roc_auc_curve(names, classifiers):
    
    # ROC curves according to the list of classifiers
    
    plt.figure(figsize=(12, 8))   
        
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        
        pred_proba = clf.predict_proba(X_val)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_val, pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=3, label= name +' ROC curve (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], color='b', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) curves', fontsize=18)
        plt.legend(loc="lower right")        

# split the data into train and test sets

cat_variables = cat_var
heartds0 = label_encode_cat_features(heartds0, cat_var)

seed = 0
test_size = 0.25

features = heartds0.columns[:-1]

X = heartds0[features]
y = heartds0['output']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state=seed)

names = [
    'Naive Bayes',
    'Logistic Regression',
    'Nearest Neighbors',
    'Support Vectors',
    'Nu SVC',
    'Decision Tree',
    'Random Forest',
    'AdaBoost',
    'Gradient Boosting',
    'Linear DA',
    'Quadratic DA',
    "Neural Net"
]

classifiers = [
    GaussianNB(),
    LogisticRegression(solver="liblinear", random_state=seed),
    KNeighborsClassifier(2),
    SVC(probability=True, random_state=seed),
    NuSVC(probability=True, random_state=seed),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    GradientBoostingClassifier(random_state=seed),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(random_state=seed),
]


score_summary = score_summary(names, classifiers).sort_values(by='Accuracy', ascending = False)\
.style.background_gradient(cmap='coolwarm')\
.bar(subset=["ROC_AUC",], color='#6495ED')\
.bar(subset=["Recall"], color='#ff355d')\
.bar(subset=["Precision"], color='lightseagreen')\
.bar(subset=["F1"], color='gold')

display(score_summary)

roc_auc_curve(names, classifiers)
plot_conf_matrix(names, classifiers, nrows=4, ncols=3, fig_a=12, fig_b=12)



