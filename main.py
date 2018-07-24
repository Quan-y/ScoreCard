import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chisquare
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import ensemble,cross_validation, metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import random
import datetime
import time
import operator
import numbers
import math
import numbers

# read the data: customer details and dictionary of fields in customer details dataset
# read the internal data and external data
# 'path' is the path for students' folder which contains the lectures of xiaoxiang
bankChurn = pd.read_csv('C:/Users/QUAN/Desktop/bank_1/2/bankChurn.csv',header=0)
externalData = pd.read_csv('C:/Users/QUAN/Desktop/bank_1/2/ExternalData.csv',header = 0)
# merge two dataframes
AllData = pd.merge(bankChurn,externalData,on='CUST_ID')

# step 1: check the type for each column and descripe the basic profile
columns = set(list(AllData.columns))
columns.remove('CHURN_CUST_IND')  #the target variable is not our object
# we differentiate the numerical type and catigorical type of the columns
numericCols = []
stringCols = []

for var in columns:
    x = list(set(AllData[var]))
    x = [i for i in x if i==i]   # we need to eliminate the noise, which is nan type
    if isinstance(x[0],numbers.Real):
        numericCols.append(var) 
    elif isinstance(x[0], str):
        stringCols.append(var)
    else:
        print 'The type of ',var,' cannot be determined'
        
# Part 1: Single factor analysis for independent variables
# we check the distribution of each numerical variable, separated by churn/not churn
filepath = 'C:/Users/QUAN/Desktop/bank_1/2/pictures_1'
for var in numericCols:
    NumVarPerf(AllData,var, 'CHURN_CUST_IND',filepath)

# need to do some truncation for outliers
filepath = 'C:/Users/QUAN/Desktop/bank_1/2/pictures_2'
for val in numericCols:
    NumVarPerf(AllData,val, 'CHURN_CUST_IND',filepath,True)
# anova test
anova_results = anova_lm(ols('ASSET_MON_AVG_BAL~CHURN_CUST_IND',AllData).fit())
# single factor analysis for categorical analysis
filepath = 'C:/Users/QUAN/Desktop/bank_1/2/pictures_3'
for val in stringCols:
    print val
    CharVarPerf(AllData,val,'CHURN_CUST_IND',filepath)

# chisquare test
chisqDf = AllData[['GENDER_CD','CHURN_CUST_IND']]
grouped = chisqDf['CHURN_CUST_IND'].groupby(chisqDf['GENDER_CD'])
count = list(grouped.count())
churn = list(grouped.sum())
chisqTable = pd.DataFrame({'total':count,'churn':churn})
chisqTable['expected'] = chisqTable['total'].map(lambda x: round(x*0.101))
chisqValList = chisqTable[['churn','expected']].apply(lambda x: (x[0]-x[1])**2/x[1], axis=1)
chisqVal = sum(chisqValList) 

# Part 2: Multi factor analysis for independent variables
# use short name to replace the raw name, since the raw names are too long to be shown
col_to_index = {numericCols[i]:'var'+str(i) for i in range(len(numericCols))}
# sample from the list of columns, since too many columns cannot be displayed in the single plot
corrCols = random.sample(numericCols,15)
sampleDf = AllData[corrCols]
for col in corrCols:
    sampleDf.rename(columns = {col:col_to_index[col]}, inplace = True)
scatter_matrix(sampleDf, alpha=0.2, figsize=(6, 6), diagonal='kde')

# ------------------------------------------------------------------------------------------------------------------------------ #
modelData = AllData.copy()
# convert date to days, using minimum date 1999/1/1 as the base to calculate the gap
modelData['days_from_open'] = Date2Days(modelData, 'open_date','1999/1/1')
del modelData['open_date']
indepCols = list(modelData.columns)
indepCols.remove('CHURN_CUST_IND')
indepCols.remove('CUST_ID')
indepCols.remove('educ1')
except_var = []

for var in indepCols:
    try:
        x0 = list(set(modelData[var]))
        if var == 'forgntvl':  
            x00 = [nan]
            [x00.append(i) for i in x0 if i not in x00 and i==i]
            x0 = x00
        if len(x0) == 1:
            print 'Remove the constant column {}'.format(var)
            indepCols.remove(var)
            continue
        x = [i for i in x0 if i==i]   #we need to eliminate the noise, which is nan type
        if isinstance(x[0],numbers.Real) and len(x)>4:
            if nan in x0:
                print 'nan is found in column {}, so we need to make up the missing value'.format(var)
                modelData[var] = MakeupMissing(modelData,var,'Contiunous','Random')
        else:
            #for categorical variable, at this moment we do not makeup the missing value. Instead we think the missing as a special type
            #if nan in x0:
                #print 'nan is found in column {}, so we need to make up the missing value'.format(var)
                #modelData[var] = MakeupMissing(modelData, var, 'Categorical', 'Random')
            print 'Encode {} using numerical representative'.format(var)
            modelData[var] = Encoder(modelData, var, 'CHURN_CUST_IND')
            
    except:
        print "something is wrong with {}".format(var)
        except_var.append(var)
        continue

modelData['AVG_LOCAL_CUR_TRANS_TX_AMT'] = ColumnDivide(modelData, 'LOCAL_CUR_TRANS_TX_AMT','LOCAL_CUR_TRANS_TX_NUM')
modelData['AVG_LOCAL_CUR_LASTSAV_TX_AMT'] = ColumnDivide(modelData, 'LOCAL_CUR_LASTSAV_TX_AMT','LOCAL_CUR_LASTSAV_TX_NUM')


# 1: creating features : max of all
maxValueFeatures = ['LOCAL_CUR_SAV_SLOPE','LOCAL_BELONEYR_FF_SLOPE','LOCAL_OVEONEYR_FF_SLOPE','LOCAL_SAV_SLOPE','SAV_SLOPE']
modelData['volatilityMax']= modelData[maxValueFeatures].apply(max, axis =1)

# 2: deleting features : some features are coupling so we need to delete the redundant
del modelData['LOCAL_CUR_MON_AVG_BAL_PROP']

# 3: sum up features: some features can be summed up to work out a total number
sumupCols0 = ['LOCAL_CUR_MON_AVG_BAL','LOCAL_FIX_MON_AVG_BAL']
sumupCols1 = ['LOCAL_CUR_WITHDRAW_TX_NUM','LOCAL_FIX_WITHDRAW_TX_NUM']
sumupCols2 = ['LOCAL_CUR_WITHDRAW_TX_AMT','LOCAL_FIX_WITHDRAW_TX_AMT']
sumupCols3 = ['COUNTER_NOT_ACCT_TX_NUM','COUNTER_ACCT_TX_NUM']
sumupCols4 = ['ATM_ALL_TX_NUM','COUNTER_ALL_TX_NUM']
sumupCols5 = ['ATM_ACCT_TX_NUM','COUNTER_ACCT_TX_NUM']
sumupCols6 = ['ATM_ACCT_TX_AMT','COUNTER_ACCT_TX_AMT']
sumupCols7 = ['ATM_NOT_ACCT_TX_NUM','COUNTER_NOT_ACCT_TX_NUM']

modelData['TOTAL_LOCAL_MON_AVG_BAL'] = modelData[sumupCols0].apply(sum, axis = 1)
modelData['TOTAL_WITHDRAW_TX_NUM'] = modelData[sumupCols1].apply(sum, axis = 1)
modelData['TOTAL_WITHDRAW_TX_AMT'] = modelData[sumupCols2].apply(sum, axis = 1)
modelData['TOTAL_COUNTER_TX_NUM'] = modelData[sumupCols3].apply(sum, axis = 1)
modelData['TOTAL_ALL_TX_NUM'] = modelData[sumupCols4].apply(sum, axis = 1)
modelData['TOTAL_ACCT_TX_NUM'] = modelData[sumupCols5].apply(sum, axis = 1)
modelData['TOTAL_ACCT_TX_AMT'] = modelData[sumupCols6].apply(sum, axis = 1)
modelData['TOTAL_NOT_ACCT_TX_NUM'] = modelData[sumupCols7].apply(sum, axis = 1)


# 4：creating features 3: ratio
numeratorCols = ['LOCAL_SAV_CUR_ALL_BAL','SAV_CUR_ALL_BAL','ASSET_CUR_ALL_BAL',
                 'LOCAL_CUR_WITHDRAW_TX_NUM','LOCAL_CUR_WITHDRAW_TX_AMT',
                 'COUNTER_NOT_ACCT_TX_NUM','ATM_ALL_TX_NUM','ATM_ACCT_TX_AMT',
                 'ATM_NOT_ACCT_TX_NUM']
denominatorCols = ['LOCAL_SAV_MON_AVG_BAL','SAV_MON_AVG_BAL','ASSET_MON_AVG_BAL',
                   'TOTAL_WITHDRAW_TX_NUM','TOTAL_WITHDRAW_TX_AMT','TOTAL_COUNTER_TX_NUM',
                   'TOTAL_ACCT_TX_NUM','TOTAL_ACCT_TX_AMT','TOTAL_NOT_ACCT_TX_NUM']

newColName = ["RATIO_"+str(i) for i in range(len(numeratorCols))]
for i in range(len(numeratorCols)):
    modelData[newColName[i]] = ColumnDivide(modelData, numeratorCols[i], denominatorCols[i])
    
# ------------------------------------------------------------------------------------------------------------------------------ #
allFeatures = list(modelData.columns)
# remove the class label and cust id from features
allFeatures.remove('CUST_ID')
allFeatures.remove('CHURN_CUST_IND')

# split the modeling dataset into trainning set and testing set
X_train, X_test, y_train, y_test = train_test_split(modelData[allFeatures],modelData['CHURN_CUST_IND'], test_size=0.5,random_state=9)
y_train.value_counts()

# try 1: using default parameter
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X_train,y_train)
y_pred = gbm0.predict(X_test)
y_predprob = gbm0.predict_proba(X_test)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred)
print "AUC Score (Testing): %f" % metrics.roc_auc_score(y_test, y_predprob)

y_pred2 = gbm0.predict(X_train)
y_predprob2 = gbm0.predict_proba(X_train)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred2)
print "AUC Score (Testing): %f" % metrics.roc_auc_score(y_train, y_predprob2)


# tunning the number of estimators
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_
gsearch1.best_params_
gsearch1.best_score_

# tunning the parameters of simple trees
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}

gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, min_samples_leaf=20,
      max_features='sqrt', subsample=0.8, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# tunning the parameters of min_samples_leaf
param_test3 = {'min_samples_split':range(400,1001,100), 'min_samples_leaf':range(60,101,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=9,
                                     max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# use the tunned parameters to train the model again
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=9, min_samples_leaf =70,
               min_samples_split =500, max_features='sqrt', subsample=0.8, random_state=10)
gbm1.fit(X_train,y_train)
y_pred1 = gbm1.predict(X_train)
y_predprob1= gbm1.predict_proba(X_train)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred1)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob1)

y_pred2 = gbm1.predict(X_test)
y_predprob2= gbm1.predict_proba(X_test)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred2)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob2)

# tunning max_features
param_test4 = {'max_features':range(5,31,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=9, min_samples_leaf =70,
               min_samples_split =500, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# tunning subsample
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70,max_depth=9, min_samples_leaf =70,
               min_samples_split =500, max_features=28, random_state=10),
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# tunning the learning rate and
gbm2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, n_estimators=70,max_depth=9, min_samples_leaf =70,
               min_samples_split =1000, max_features=28, random_state=10,subsample=0.8),
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gbm2.fit(X_train,y_train)

y_pred1 = gbm2.predict(X_train)
y_predprob1= gbm2.predict_proba(X_train)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred1)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob1)

y_pred2 = gbm2.predict(X_test)
y_predprob2= gbm2.predict_proba(X_test)[:,1]
print "Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred2)
print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob2)

clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=70,max_depth=9, min_samples_leaf =70,
               min_samples_split =1000, max_features=28, random_state=10,subsample=0.8)
clf.fit(X_train, y_train)
importances = clf.feature_importances_

# sort the features by importance in descending order. by default argsort returing asceding order
features_sorted = argsort(-importances)
import_feautres = [allFeatures[i] for i in features_sorted]