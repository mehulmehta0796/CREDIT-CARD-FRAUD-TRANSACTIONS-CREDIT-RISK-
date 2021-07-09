"""
@author: Mehul Mehta
Project: Credit Risk (Detection of Credit Card Fraud Transaction)
"""

#Importing all the important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib.pyplot import figure
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#Reading the dataset
data = pd.read_csv("creditcard.csv")

#Converting the data into 24 hrs format.
data['Time'] = data['Time']/3600 #Converted second into hours.
bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48] 
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6','6-7', '7-8', '8-9','9-10', '10-11', '11-12', '12-13', '13-14', '14-15','15-16', '16-17', '17-18','18-19','19-20','20-21', '21-22', '22-23', '23-24', '24-25','25-26', '26-27', '27-28','28-29','29-30','30-31', '31-32', '32-33', '33-34', '34-35','35-36', '36-37', '37-38','38-39','39-40','40-41', '41-42', '42-43', '43-44', '44-45','45-46', '46-47', '47-48']
data['Time'] = pd.cut(data['Time'], bins, labels = labels,include_lowest = True)
convert_time = {"Time": {"47-48": "23-24","46-47": "22-23","45-46": "21-22","44-45": "20-21","43-44": "19-20","42-43": "18-19","41-42": "17-18","40-41": "16-17","39-40": "15-16","38-39": "14-15","37-38": "13-14","36-37": "12-13","35-36": "11-12","34-35": "10-11","33-34": "9-10","32-33": "8-9","31-32": "7-8","30-31": "6-7","29-30": "5-6","28-29": "4-5","27-28": "3-4","26-27": "2-3","25-26": "1-2","24-25": "0-1"}}
data = data.replace(convert_time)
print(data)

"Exploratory Data Analysis"
# Generate descriptive statistics that summarize the dataset
print(data.describe())

# Shape of the dataset
print(data.shape)

# Extracting the dataset information
print(data.info)

# Checking the Null Values in the dataset
print(data.isnull().sum())

# Checking the Column Names in the dataset
print(data.columns)

# Time when Fraud happend.
data_Fraud = data[data['Class']==1]
sns.countplot(data_Fraud['Time'])
plt.xticks(rotation=90)

#Amount of Fraud happened at each hour of day.
sns.scatterplot(data_Fraud['Time'],data_Fraud['Amount'])
plt.xticks(rotation=90)

#Correlation Matrix
correlation = data.corr()
sns.heatmap(correlation, square = True, cmap = 'YlOrRd', annot = True, vmax = 0.9, fmt = '.2f')
plt.show() #Thus, None of the atrributes are correalted to each other.

#The classes are imbalanced and we need to solve this issue later.
Fraud = data['Class'].value_counts()[0]
Non_Fraud = data['Class'].value_counts()[1]

# Total Percentage of Fraud and Non Fraud Transactions.
print("No Frauds",data['Class'].value_counts()[0]/len(data) * 100,"%")
print("Frauds",data['Class'].value_counts()[1]/len(data) * 100,"%")

#Analysing the Class Column: It seems that most of the transactions are non-fraud.
sns.countplot('Class', data=data)
plt.xlabel("Class")
plt.ylabel("Number of Observations")
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)')

# #Dropping Time Column since it is not giving any insights.
data = data.drop('Time',axis=1)
print(data)

#Sum of Fraud Transaction
Fraud_Transaction = data[data['Class'] == 1]['Amount'].sum()
print("The sum of Fraud Transaction is:",Fraud_Transaction)

#Selecting X and Y parameters to train the Model.
X = data.drop(columns = 'Class', axis=1)
y = data['Class']

#Detecting Outliers in the Dataset
i=0  
feature_name = data.columns.values
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))
for feature in feature_name:
    i += 1
    plt.subplot(8,4,i)
    sns.boxplot(data[feature])
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();    


# Function to remove Outliers from the Dataset
def remove_outlier(data_in, col_name):
    # calculate interquartile range
    q1 = data_in[col_name].quantile(0.25)
    q3 = data_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q1, q3, iqr))
    low  = q1-1.5*iqr
    high = q3+1.5*iqr
    df_out = data_in.loc[(data_in[col_name] > low) & (data_in[col_name] < high)]
    outliers = data_in.loc[(data_in[col_name] < low) | (data_in[col_name] > high)]
    return df_out


# Apply SelectKBest class to extract top 10 best features
X_norm = MinMaxScaler().fit_transform(X)
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X_norm,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns) 
featureScores = pd.concat([dfcolumns,dfscores],axis=1) #concat two dataframes for better visualization
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #This will print 10 best features


# Kernel density estimation (KDE) plots: This should give us a clear idea of how fraudulent and non fraudulent transactions are distributed along each variable.
i=0
feature_name = data.columns.values
C0 = data.loc[data['Class'] == 0]
C1 = data.loc[data['Class'] == 1]
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))
for feature in feature_name:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(C0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(C1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();

#Observarion:
# 1. The distribution of the fraudulent transaction takes a shape that very close to a Normal Distribution. 
# 2. The distribution of the non fraudulent transaction takes a shape that very close to the Standard Normal Distribution.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#Before applying SMOTE.
print("Xtrain.shape : ", X_train.shape)
print("Xtest.shape  : ", X_test.shape)
print("Ytrain.shape : ", y_train.shape)
print("Ytest.shape  : ", y_test.shape)

#Sampling the data so that we have equal number of "Class 0" and "Class 1"
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state= 42)
X_train_sample, y_train_sample = sm.fit_resample(X_train,y_train)


#After applying SMOTE.
print("Xtrain.shape : ", X_train_sample.shape)
print("Xtest.shape  : ", X_test.shape)
print("Ytrain.shape : ", y_train_sample.shape)
print("Ytest.shape  : ", y_test.shape)

#Dataset has now balanced i.e., it has equal number of "0" and "1".
print(y_train_sample.value_counts())
#Training the Model

# 1. Logistic Regression
model = LogisticRegression()
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Logistic Regression Accuracy is',accuracy_score(y_pred,y_test))
print('/n')


# 2. # K-Nearest Neighbours
model = KNeighborsClassifier(metric='manhattan', n_neighbors=2, weights='uniform')
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('K-Nearest Neighbour Accuracy is',accuracy_score(y_pred,y_test))
print('/n')


# # Tuning the K-Nearest Neighbours Model using Grid SearchCV
# model = KNeighborsClassifier()
# n_neighbors = range(1, 15, 1)
# weights = ['uniform', 'distance']
# metric = ['euclidean', 'manhattan', 'minkowski']
# # define grid search
# grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train_sample, y_train_sample)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


#3. # Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Decision Tree Accuracy is',accuracy_score(y_pred,y_test))
print('/n')


#4. Naive Bayes

model = GaussianNB()
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Naive Bayes Accuracy is',accuracy_score(y_pred,y_test))
print('/n')

#5. Random Forest
model =  RandomForestClassifier(n_estimators=100, max_features=3)
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Random Forest Accuracy is',accuracy_score(y_pred,y_test))
print('/n')

#6. Support Vector Machine
model = SVC(kernel='rbf', C=1)
model.fit(X_train_sample, y_train_sample)
y_pred = model.predict(X_test)

# # Tuning the Support Vector Machine Model.
# model = SVC()
# kernel = ['poly', 'rbf', 'sigmoid']
# C = [50, 10, 1.0, 0.1, 0.01]
# gamma = ['scale']
# # define grid search
# grid = dict(kernel=kernel,C=C,gamma=gamma)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train_sample, y_train_sample)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Support Vector Machine Accuracy is',accuracy_score(y_pred,y_test))
print('/n')


# End..............................
# Thank You........................
