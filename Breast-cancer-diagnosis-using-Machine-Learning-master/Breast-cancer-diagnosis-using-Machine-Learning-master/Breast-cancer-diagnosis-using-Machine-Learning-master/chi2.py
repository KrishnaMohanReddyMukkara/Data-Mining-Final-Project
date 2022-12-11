# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:31:39 2020

@author: HOME
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
#from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df = pd.read_csv('cancer1.csv')
print(df.head(5))
df.replace('?',9999,inplace=True)
df.drop(['id'],1,inplace=True)
print(df.head(5))

X =df.iloc[:,1:-1]  #independent columns
y =df.iloc[:,0]    #target column i.e classes
labelencoder_y = LabelEncoder()
df.iloc[:,0]= labelencoder_y.fit_transform(df.iloc[:,0].values).astype('float64')
print(labelencoder_y.fit_transform(df.iloc[:,0].values))
#sns.pairplot(df.iloc[:,1:7],hue='diagnosis')

#apply SelectKBest class to extract top 8 best features
bestfeatures = SelectKBest(score_func=chi2, k=13)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(13,'Score'))  #print 8 best features



#pairplot
#sns.pairplot(df.iloc[:,1:10],hue='classes')

#correlation
df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(df.corr())
# Importing the dataset

#df = pd.read_csv('cancer.csv')
#df.replace('?',-99999,inplace=True)
#df.drop(['id'],1,inplace=True)

#X=np.array(df.drop(['classes'],1))
#y=np.array(df['classes'])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)
#print(X_train)
select_feature = SelectKBest(chi2, k=13).fit(X_train, y_train)
X_train2 = select_feature.transform(X_train)
X_test2 = select_feature.transform(X_test)

print('X_train2')
print(X_train2)

print('X_test2')
print(X_test2)

print('y_train')
print(y_train)

#print(np.array(X_train2))
#clf_rf_2 = RandomForestClassifier()      
#clr_rf_2 = clf_rf_2.fit(X_train2,y_train)
#print(clr_rf_2)
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)
 
#principle component analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train2 = pca.fit_transform(X_train2)
X_test2 = pca.fit_transform(X_test2)
explained_variance=pca.explained_variance_ratio_





# Fitting KNN to the Training set

from sklearn.neighbors import KNeighborsClassifier
knn = []
for i in range(1,21):
            
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model=classifier.fit(X_train2,y_train)
    trained_model.fit(X_train2,y_train )
    
    # Predicting the Test set results
    
    y_pred = classifier.predict(X_test2)
    
    # Making the Confusion Matrix
    
    
    from sklearn.metrics import confusion_matrix
    
    cm_KNN = confusion_matrix(y_test, y_pred)
    print(cm_KNN)
    print("Accuracy score of train KNN")
    print(accuracy_score(y_train, trained_model.predict(X_train2))*100)
    
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

trained_model=classifier.fit(X_train2,y_train)
trained_model.fit(X_train2,y_train )


# Predicting the Test set results
print(X_test2.shape)
y_pred = classifier.predict(X_test2)
#print(y_test) 
#print(y_pred)# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy score of train SVM")
print(accuracy_score(y_train, trained_model.predict(X_train2))*100)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, y_pred)*100)



#new data prediction
df2 = pd.read_csv('wisc_data.csv')
print(df2.head(5))
df2.replace('?',9999,inplace=True)
df2.drop(['id'],1,inplace=True)
print(df2.head(5))

X =df2.iloc[:,1:-1]  #independent columns
y =df2.iloc[:,0]    #target column i.e classes
labelencoder_y = LabelEncoder()
df2.iloc[:,0]= labelencoder_y.fit_transform(df2.iloc[:,0].values).astype('float64')
print(labelencoder_y.fit_transform(df2.iloc[:,0].values))


Y_test=df2.iloc[:,1:-1]
print(Y_test.head(5))

Y_test=sc.fit_transform(Y_test)   #feature scaling
pca=PCA(n_components=2)
Y_test=pca.fit_transform(Y_test)
#print(Y_test.shape)
#print(Y_test)

#SVM PREDICTION FOR NEW DATA
Y_pred=classifier.predict(Y_test)
# Making the Confusion Matrix
print('Y_test.shape')
print(Y_test.shape)
#print(X_test2.shape)
print('Y_pred')
print(Y_pred)
cm_SVM = confusion_matrix(y_test, Y_pred)
print(cm_SVM)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, Y_pred)*100)
print(cm_SVM)
print("Accuracy score of test KNN")
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
# print(y_test.shape)
cm_KNN = confusion_matrix(y_test, Y_pred)
print(cm_KNN)

print("Accuracy score of test KNN")
print(accuracy_score(y_test, Y_pred)*100)
knn.append(accuracy_score(y_test, Y_pred)*100)
#print(cm_KNN)


#print(Y_test)


