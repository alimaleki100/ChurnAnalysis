# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:11:03 2020

@author: alimaleki100
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


df = pd.read_csv('churn.txt', sep=",")

print(df.head(10))

#################################################################################
#################################################################################
#################################EDA################################


### Coreelation
df.corr().plot(kind='hexbin',title ="VMail Plan",figsize=(10,10))
plt.figure(figsize=(15,8))
sns.heatmap(dfcorr, cmap="YlGnBu", annot=True)
plt.savefig('f:/Churn Project/images/corr.png')



df.groupby(["Int'l Plan"]).size().plot(kind='pie',title ="Int'l Plan")
plt.savefig('f:/Churn Project/images/IntlPie.png')



df.groupby(["VMail Plan"]).size().plot(kind='pie',title ="VMail Plan")
plt.savefig('f:/Churn Project/images/VMailPie.png')


dfstate=df.groupby(["State"])
dfstate.plot(kind='bar',color='red',figsize=(15,5),title ="Number of Customers by State")
plt.savefig('f:/Churn Project/images/StateBar.png')

df.plot(kind='box',x='Eve Mins',y='Eve Calls',title ="VMail Plan",figsize=(10,10))
plt.savefig('f:/Churn Project/images/VMailPie.png')


ax=sns.heatmap()



plt.figure(figsize=(15,8))
ax = sns.scatterplot(x='Day Mins', y='Night Mins',data=df)



#################################################################################
#################################################################################
#################################Predicting Churn################################
#Dropping irrelevant Data
df=df.drop(['Area Code','Phone'], axis=1)

#Detect columns contain any null
df.isnull().any()


print(df.dtypes)


yes_no_cols = ["Int'l Plan","VMail Plan"]



#OneHotEncoding using GetDummies
df=pd.get_dummies(df,columns=yes_no_cols)




y = df["Churn?"].values
X = df.drop(labels = ["Churn?","State"],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


###################Logistic Regression
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
LRaccuracy=metrics.accuracy_score(y_test, y_pred)
print (LRaccuracy)




########################KNN
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNNclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = KNNclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
KNNcm = confusion_matrix(y_test, y_pred)

from sklearn import metrics
# training metrics
KNNaccuracy=metrics.accuracy_score(y_test, y_pred)

print("KNNAccuracy: {0:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Confusion Matrix")
print(KNNcm)

########################SVM

# Fitting SVM to the Training set
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'linear', random_state = 0)
svmclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = svmclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
svmcm = confusion_matrix(y_test, y_pred)
svmaccuracy=metrics.accuracy_score(y_test, y_pred)


########################Naive Bayes

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()
nbclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nbclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
nbcm = confusion_matrix(y_test, y_pred)
nbaccuracy=metrics.accuracy_score(y_test, y_pred)

########################ِDecision Tree

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dtclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dtclassifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
dtcm = confusion_matrix(y_test, y_pred)
dtaccuracy=metrics.accuracy_score(y_test, y_pred)


########################ِRandom Forest

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rfclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
rfcm = confusion_matrix(y_test, y_pred)
rfaccuracy=metrics.accuracy_score(y_test, y_pred)




########Print Results
#LR
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Results$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("%%%%%%%%%%%%%%%%%%%%%%Logistic Regression:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(cm)
print (LRaccuracy)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%   KNN:     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(KNNcm)
print(KNNaccuracy)


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%   SVM:     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(svmcm)
print(svmaccuracy)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Naive Bayes:     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(nbcm)
print(nbaccuracy)



print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Decision Tree:     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(dtcm)
print(dtaccuracy)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Random Forest:     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5")
print(rfcm)
print(rfaccuracy)




# To get the weights of all the variables linear models
weights = pd.Series(classifier.coef_[0],index=X.columns.values)
weights.sort_values(ascending = False)
print(weights)

# To get the weights of all the variables nonlinear models
importnace = pd.Series(rfclassifier.feature_importances_,index=X.columns.values)
importnace.sort_values(ascending = False)
print(importnace)



