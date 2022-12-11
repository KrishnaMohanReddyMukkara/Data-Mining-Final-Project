# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:33:18 2020

@author: HOME
"""

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
#from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
np.random.seed(123)

df = pd.read_csv('cancer1.csv')
df=df.iloc[:,1:-1]
#df.replace('?',99999,inplace=True)
df.dropna(axis=1)
print('shape of cells')
print(df.shape)

df.info()

z=df.diagnosis
B, M = z.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
#df['diagnosis'].value_counts()
 
#visualization
sns.countplot(df['diagnosis'],label="Count")

#Encoding categorical data values (
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
df.iloc[:,0]= labelencoder_y.fit_transform(df.iloc[:,0].values).astype('float64')
print(labelencoder_y.fit_transform(df.iloc[:,0].values))

#print(df.describe())

# X =df.iloc[:,2:31]  #independent columns
#y =df.iloc[:,0]    #target column i.e classes

#apply SelectKBest class to extract top 8 best features
#bestfeatures = SelectKBest(score_func=chi2, k=9)
#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(df.iloc[:,2:31].columns)
#concat two dataframes for better visualization 
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(9,'Score'))  #print 10 best features

#X=pd.DataFrame(featureScores.columns)
#pairplot
#sns.pairplot(df.iloc[:,1:4],hue='diagnosis')

#new dataset
print(df.head(5))

#Get the correlation of the columns
#df2=df.iloc[:,2:32]

#generating correlation matrix 
corr = df.corr()
#df.corr()
plt.figure(figsize=(8,8))  
sns.heatmap(df.corr())


#Next, we compare the correlation between features and remove one of two features that have a correlation higher than 0.9
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]

#Next we will be selecting the columns based on how they affect the p-value.
selected_columns = selected_columns[1:].values
#import statsmodels.formula.api as sm
import statsmodels.api as sm
def backwardElimination(X, y, sl, columns):
    numVars = len(X[0])
    for i in range(0, numVars): 
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X = np.delete(X, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return X, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(df.iloc[:,1:].values, df.iloc[:,0].values, SL, selected_columns)

#the result to a new Dataframe.
result = pd.DataFrame()
result['diagnosis'] = df.iloc[:,0]

#Creating a Dataframe with the columns selected using the p-value and correlation
data = pd.DataFrame(data = data_modeled, columns = selected_columns)
print(data.columns)

#visualize the data
fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(df[i][result['diagnosis']==0], color='g', label = 'benign')
    sns.distplot(df[i][result['diagnosis']==1], color='r', label = 'malignant')
    plt.legend(loc='best')
fig.suptitle('Breast Cance Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(data.values, result.values, test_size = 0.3)


print('xtrain data')
print(X_train )

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
#principle component analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance=pca.explained_variance_ratio_

print(X_test.shape)
#print(X_test)



# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = []
for i in range(1,21):
            
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model=classifier.fit(X_train,y_train)
    trained_model.fit(X_train,y_train )
    
    # Predicting the Test set results
    
    y_pred = classifier.predict(X_test)
   # print(y_pred)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix   
    
    cm_KNN = confusion_matrix(y_test, y_pred)
    print(cm_KNN)
    print("Accuracy score of train KNN")
    print(accuracy_score(y_train, trained_model.predict(X_train))*100)
    
    print("Accuracy score of test KNN")
    print(accuracy_score(y_test, y_pred)*100)
    
    knn.append(accuracy_score(y_test, y_pred)*100)
    
       
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 21),knn, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
plt.title('Accuracy for different  K Value')  
plt.xlabel('K Value')  
plt.ylabel('Accuracy') 


# Fitting SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )


# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy score of train SVM")
print(accuracy_score(y_train, trained_model.predict(X_train))*100)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, y_pred)*100)




# CODE FOR PREDICTING new data
df2 = pd.read_csv('breast-cancer-wisconsin.csv')
df2=df2.iloc[:,1:-1]
#df.replace('?',99999,inplace=True)
df2.dropna(axis=1)
labelencoder_y = LabelEncoder()
df2.iloc[:,0]= labelencoder_y.fit_transform(df2.iloc[:,0].values).astype('float64')
print(labelencoder_y.fit_transform(df2.iloc[:,0].values))


Y_test=df2.iloc[:,1:-1]
print(Y_test.head(5))

Y_test=sc.fit_transform(Y_test)
pca=PCA(n_components=1)
Y_test=pca.fit_transform(Y_test)
#print(Y_test)

#SVM PREDICTION FOR NEW DATA
Y_pred=classifier.predict(Y_test)
# Making the Confusion Matrix

cm_SVM = confusion_matrix(y_test, Y_pred)
print(cm_SVM)


print("Accuracy score of test SVM")
print(accuracy_score(y_test, Y_pred)*100)



#KNN PREDICTION FOR NEW DATA
knn = []
i=13
classifier = KNeighborsClassifier(n_neighbors=i)
trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )   
classifier = KNeighborsClassifier(n_neighbors=i)
trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )
         

    # Predicting the Test set results
    
Y_pred=classifier.predict(Y_test)
print('shape of predicted data')
print(Y_pred.shape)
print('shape of test data')
print(y_test.shape)
print(y_test.shape)
cm_KNN = confusion_matrix(y_test, Y_pred)
print(cm_KNN)

print("Accuracy score of test KNN")
print(accuracy_score(y_test, Y_pred)*100)
knn.append(accuracy_score(y_test, Y_pred)*100)
#print(cm_KNN)


#print(Y_test)

    