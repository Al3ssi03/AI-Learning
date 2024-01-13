# %%
import numpy as np 
import pandas as pd
from scipy import stats

#READ CSV AND INSERT IN DATAFRAME
ParkisonDataSet = pd.read_csv('input\Data.csv',sep=',',header=0)

#CLEANING DATA
ParkisonDataSet = ParkisonDataSet.dropna()
ParkisonDataSet = ParkisonDataSet.drop_duplicates()

#SUMMARY DATA SET INDICATING MEAND,STD, MIN
ParkisonDataSet.describe().T

# %%
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.model_selection import  GridSearchCV,  train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X = ParkisonDataSet
Y = ParkisonDataSet['RPDE']

#Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real),2)))

def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)


X = ParkisonDataSet
#round the all value of columns RPDE and multiply It for 100 for example 0,745 --> 75
y = np.round(ParkisonDataSet['RPDE']*100)
#replace infinti value or NaN value to 0 value 
X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)

scaler = MinMaxScaler()
scaler.fit(X)
#Assign at X the column scaler in 0 and 1 
X = scaler.transform(X) 

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)

Algo = [
        #'ElasticNet',
         'KNN',
         'DecisionTree',
         'RandomForestClassifier',
         #'GridSearchCV',
         'HuberRegressor',
         'Ridge',
         'Lasso',
         'LassoCV',
         'Lars',
         #'BayesianRidge',
         'SGDClassifier',
         'RidgeClassifier',
         'LogisticRegression',
         'OrthogonalMatchingPursuit',
         #'RANSACRegressor',
]
classifiers = [
    KNeighborsClassifier(n_neighbors = 1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators = 200),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),
    Ridge(fit_intercept=True, alpha=0.0, random_state=0),
    Lasso(alpha=0.05),
    LassoCV(),
    Lars(n_nonzero_coefs=10),
    #BayesianRidge(),
    SGDClassifier(),
    RidgeClassifier(),
    LogisticRegression(),
    OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
correction = [0,0,0,0,0,0,0,0,0,0,0,0]
temp=zip(Algo,classifiers,correction)

for name, clf,correct in temp:
    regr=clf.fit(X,Y)
    print(name,'%1 error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))
''' from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    # Confusion Matrix
    print(name,'Confusion Matrix')
    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )
    print('--'*40)

    # Classification Report
    print('Classification Report')
    print(classification_report(Y,np.round( regr.predict(X) ) ))

    # Accuracy
    print('--'*40)
    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)
    print('Accuracy', logreg_accuracy,'%')
'''



