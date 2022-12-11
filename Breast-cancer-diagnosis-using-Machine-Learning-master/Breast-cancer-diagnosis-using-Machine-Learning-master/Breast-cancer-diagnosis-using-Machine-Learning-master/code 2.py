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
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('cancer.csv')
#print(df)
df.replace('?',9999,inplace=True)
df.drop(['id'],1,inplace=True)
#print(df)

X =df.iloc[:,2:30]  #independent columns
y =df.iloc[:,-2]    #target column i.e classes

print (X.describe())
print (y.describe())

#pairplot
sns.pairplot(df.iloc[:,1:10],hue='classes')

#correlation
df.corr()
plt.figure(figsize=(10,10))  
sns.heatmap(df.corr())
#pandas.cut(x, bins, right: bool = True, labels=None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise')[source]¶
#apply SelectKBest class to extract top 8 best features
bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(5,'Score'))  #print 8 best features



# Importing the dataset

#df = pd.read_csv('cancer.csv')
#df.replace('?',-99999,inplace=True)
#df.drop(['id'],1,inplace=True)

#X=np.array(df.drop(['classes'],1))
#y=np.array(df['classes'])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 42)

select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)
X_train2 = select_feature.transform(X_train)
X_test2 = select_feature.transform(X_test)
#print(np.array(X_train2))
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(X_train2,y_train)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)
 
#principle component analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
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
    print("Accuracy score of train fuzzy KNN")
    print(accuracy_score(y_train, trained_model.predict(X_train2))*100)
    
    print("Accuracy score of test fuzzy KNN")
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

y_pred = classifier.predict(X_test2)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy score of train SVM")
print(accuracy_score(y_train, trained_model.predict(X_train2))*100)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, y_pred)*100)





