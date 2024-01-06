#!/usr/bin/env python
# coding: utf-8

# In[2747]:


#First data which is about transport Istanbul/Turkiye . This data has provided how many people had taken transport for several years . Especially 2019 years( the year before corona ) train transport datas are looked for to compare Ireland train transport datas .
#Data has been read.

import pandas as pd

dataist = pd.read_csv('ıstanbulverı1.csv', encoding='iso-8859-9')  #encoding has been added due to different language . Components of data are Turkish . 
print(dataist)


# In[2748]:


dataist #data has been shown .


# In[2749]:


#Columns of the data have been assigned due to be clear . They naturally were written being Turkish and they have been changed .

new_column_names = {
    'Yıl': 'Year',
    'Yolcu Sayısı (Kişi/Gün)': 'Passenger Count (Person/Day)',
    'Yolculuk Türü': 'Transport Type'
}

dataist = dataist.rename(columns=new_column_names)

print(dataist)


# In[2750]:


dataist #Action has been controlled , names seem properly .


# In[2751]:


#Other Turkish datas have been converted being English .

dataist['Transport Type'] = dataist['Transport Type'].replace('Raylı Sistemler', 'Train')
dataist['Transport Type'] = dataist['Transport Type'].replace('Deniz Ulaşımı', 'Sea')
dataist['Transport Type'] = dataist['Transport Type'].replace('Karayolu', 'Road')


# In[2752]:


dataist #Control has been done .


# In[2753]:


dataist.isnull().sum() #Data has been controlled obtain 'Nan' values or not . As long as the results seem '0' , data is ready to evaluate .


# In[2754]:


#The object was approaching 2019 train transport datas . Filtre command is used to shrink actual data .

filtre = (dataist['Year'] == 2019) & (dataist['Transport Type'] == 'Train') 
dataist_train_2019= dataist[filtre]


# In[2755]:


dataist_train_2019 #data has been controlled to be sure . The year before corona which had been targetted .


# In[2756]:


dataist_train_2019.info() #data is examined . Types of components are known being important if there is request for managing data .


# In[2757]:


#Passenger Count values were observed object and with this code which is settled below Passenger Count values are converted being integer .

dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)'].str.replace(',', '').astype(int)


# In[2758]:


dataist_train_2019.info() #Data types have been controlled after using code .


# In[2759]:


#For later , Passenger Count (Person/Year) has been calculated .

import pandas as pd

# 'Person/Day' data is converted being 'Person/Year' 
dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)']
dataist_train_2019['Passenger Count (Person/Year)'] = dataist_train_2019['Passenger Count (Person/Day)'] * 365  

# Unnecessary datas are wiped out
dataist_train_2019y= dataist_train_2019.drop(columns=['Passenger Count (Person/Day)'])

print(dataist_train_2019y)


# In[2760]:


dataist_train_2019y #data has been displayed


# In[2761]:


dataist_train_2019y.info() #data had been examined 


# In[2762]:


dataist_train_2019y.describe(include=object) #data is examined


# In[2763]:


dataist_train_2019.describe() #data is examined


# In[2764]:


dataist_train_2019 #data which will be used is displayed .


# In[2765]:


#Data is ready for using with 2019 Train Transport values . 
#Another data which obtains Ireland Transport datas . It has been read below .


# In[2766]:


import pandas as pd
import requests

# JSON data is read
url = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/THA25/JSON-stat/1.0/en"
response = requests.get(url)

# JSON data is converted Dataframe
if response.status_code == 200:
    data = response.json()
    print(data)  # JSON data is showed
else:
    print("No data. Error:", response.status_code)


# In[2767]:


data #data has been shown .


# In[2768]:


#data has been converted being DataFrame . 

import pandas as pd

# data is created 
data = {
    'STATISTIC': ['TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01'],
    'Statistic Label': ['Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys'],
    'TLIST(W1)' : ['2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019'] ,
    'Year': ['2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019'],
    'Luas Line ' : ['All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines' ],
    'Weeks of the year' : ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53'],
    'VALUE' : ['549533', '839022', '819359', '860745', '912795', '912612', '940476', '952291', '929391', '941919', '924998', '868205', '1003871', '933575', '945662', '842186', '801296', '919255', '817933', '904983', '902415', '973025', '826269', '925516', '883208', '905636', '982288', '919158', '903958', '926491', '832452', '807393', '868677', '862939', '897355', '933362', '969818', '990123', '1031937', '986159', '1054749', '1014017', '1028522', '924586', '1019705', '1038825', '1062275', '1113668', '1080791', '1151098', '1173473', '538511','NaN']
}


# DataFrame is created .
df = pd.DataFrame(data)
df['Year'] = df['Year'].astype(int)

# DataFrame is shown
print(df)


# In[2769]:


df #DataFrame had been displayed


# In[2770]:


df.shape


# In[2771]:


df.describe()


# In[2772]:


df.info()


# In[2773]:


df.isnull().sum() #Data has been controlled obtain 'Nan' value .


# In[2774]:


df[df.isna().any(axis=1)] #Data has been controlled obtain 'Nan' value .
print(df[df.isna().any(axis=1)])


# In[2775]:


df.columns #For amputing names of columns have been taken .


# In[2776]:


df = df.drop(['STATISTIC', 'Statistic Label', 'TLIST(W1)', 'Luas Line '], axis=1) #Negligible columns have been dropped . Another data which contains Istanbul values is known that these columns do not exist there .


# In[2777]:


df


# Nan value has been targeted to fill up it . 
# 
# Option 1 ) Fill with the Mean or the Median
# If the data were missing completely at random, then mean /median imputation might be suitable. You might also want to capture if the data was originally missing or not by creating a “missing indicator” variable.
# Both methods are extremely straight forward to implement.
# 
# If a variable is normally distributed, the mean, median, and mode, are approximately the same. Therefore, replacing missing values by the mean and the median are almost equivalent.
# 
# Replacing missing data by the mode is not appropriate for numerical variables.
# 
# If the variable is skewed, the mean is biased by the values at the far end of the distribution.
# 
# Therefore, the median is a better representation of the majority of the values in the variable.
# 
# Having said that, you should avoid filling with mean, if you observe and increasing or decreasing trend in your data, in which case you might want to consider ‘interpolation’ and [predicting the missing value using ML approach].
# 
# Option 2 ) Fill with Machine Learning models as some of them have been counted below
# Decision Tree
# Random Forest
# Linear Regression
# Ridge Regression
# Lasso Regression

# In[2778]:


#Mode and Median methods have been used .

import pandas as pd

# Sample dataframe has been created 
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN value has been filled up with median .
median_value = datacreate['values'].median()
datacreate['values'].fillna(median_value, inplace=True)

print(datacreate)


# In[2779]:


import pandas as pd

# Sample dataframe has been created 
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN value has been filled up with mode .
mode_value = datacreate['values'].mode()[0]  
datacreate['values'].fillna(mode_value, inplace=True)

print(datacreate)


# In[2780]:


import pandas as pd

# Sample dataframe has been created 
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN value has been filled up with mode .
datacreate['values'].fillna(datacreate['values'].mean(), inplace=True)

print(datacreate)


# In[2781]:


#Mean , Mod and Median values have been taken . Several Machine Learning Models will have used and the conclusion will be given after to compare all .


# In[2782]:


#Libraries have been uploaded .

import numpy as np 
import pandas as pd 
import seaborn as sns
import re
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

from scipy.stats.mstats import winsorize
import scipy.stats as stats
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve)

from statsmodels.tools.eval_measures import mse, rmse
from sklearn import preprocessing


# In[2783]:


dftest=df[ :52] #Until 52.rows have been selected and created test dataframe .


# In[2784]:


dftest #test dataframe has been displayed 


# In[2785]:


dftest.info() #Data has been examined getting knowledge . Type of value and type of Weeks of the year have to be changed being integer .


# In[2786]:


dftest['VALUE'] = dftest['VALUE'].astype(int) #Converting has been completed .


# In[2787]:


dftest['Weeks of the year'] = dftest['Weeks of the year'].astype(int) #Converting has been completed .


# In[2788]:


#X and y values have been derivated .

y = dftest['VALUE'] 
X = dftest.drop(['VALUE'],axis=1)


# In[2789]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2 , random_state= 4) #Train and Test values have been assigned . %80 train values , %20 test values have been taken . 


# In[2790]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[2791]:


## Applying ML Models


# In[2792]:


# Linear regression


# In[2793]:


lrm = LinearRegression()
lrm.fit(X_train, Y_train) #fit an OLS model


# In[2794]:


y_preds_train = lrm.predict(X_train)
y_preds_test = lrm.predict(X_test)  #making predictions


# In[2795]:


print("R-squared of the model in training set is: {}".format(lrm.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lrm.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test) / Y_test)) * 100))


# In[2796]:


## Ridge Regression


# In[2797]:


# Using GridSearch for parameter optimization
ridgeregr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

ridgeregr.fit(X_train, Y_train)

ridge = ridgeregr.best_estimator_


# In[2798]:


# Predictions have been made
y_preds_train = ridge.predict(X_train)
y_preds_test_ridge = ridge.predict(X_test)

print("R-squared of the model in training set is: {}".format(ridge.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(ridge.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_preds_test_ridge)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_ridge) / Y_test)) * 100))


# In[2799]:


### Lasso Regression


# In[2800]:


# using GridSearch for parameter optimization
lassoregr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

lassoregr.fit(X_train, Y_train)

lasso = lassoregr.best_estimator_


# In[2801]:


# Predictions have been made
y_preds_train = lasso.predict(X_train)
y_preds_test_lasso = lasso.predict(X_test)

print("R-squared of the model in training set is: {}".format(lasso.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lasso.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test_lasso)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_lasso) / Y_test)) * 100))


# In[2802]:


#### Decision Tree Regressor


# In[2803]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[2804]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[2805]:


y_preds_train = DTregressor.predict(X_train)
y_preds_train


# In[2806]:


y_pred_DT = DTregressor.predict(X_test)


# In[2807]:


y_pred_DT


# In[2808]:


print("R-squared of the model in training set is: {}".format(DTregressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(DTregressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_DT)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_DT) / Y_test)) * 100))


# In[2809]:


##### Random Forest


# In[2810]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state= 100)


# In[2811]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)


# In[2812]:


y_pred_random = regressor.predict(X_test)


# In[2813]:


y_pred_random


# In[2814]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_random)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_random) / Y_test)) * 100))


# In[2815]:


#Machine Learning Methods which were used , they are all have been shown together by bar . This showing has provided clear sight to see what is distinguish between of them .


# In[2816]:


import matplotlib.pyplot as plt
r_squared_values = [0.221021374150552, 0.221021371544315, 0.221021371544315 , 1.0 , 0.9327453294294139] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Training Values from Models')
plt.show()


# In[2817]:


import matplotlib.pyplot as plt
r_squared_test_values = [-0.193743303365938, -0.193673935964648, -0.193742554697871 , 0.942035242036975 , -0.8878587040645121] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_test_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Test Values from Models')
plt.show()


# In[2818]:


#Comparasion of several method Decision Tree seems greater way . But Decision Tree will not be taken to find Nan value because of overfitting . R-squared of the model in training set represents great value but R-squared of the model in test set represents low value . What does it mean ? That indicates there is something wrong here as overfitting . As long as test and train values closer each other , that way can be rational . Decision Tree model has not been taken because of dimention of train&test values .  


# In[2819]:


#Datacreate which had been created by mode . That value has been accepted instead of NaNn value . For remaining it has been displayed below .


# In[2820]:


import pandas as pd

# Sample dataframe has been created 
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN value has been filled up with mode .
mode_value = datacreate['values'].mode()[0]  
datacreate['values'].fillna(mode_value, inplace=True)

datacreate


# In[2821]:


#Values which are coming from Ireland and Istanbul will be merged to compare them . Datas which contains Istanbul values has to be converted as datacreate .


# In[2822]:


datacreate.shape #shape has been controlled


# In[2823]:


dataist_train_2019.shape #shape has been controlled


# In[2824]:


#Dataist_train_2019 which contains Istanbul train transport values should be changed as datacreate which contains Ireland train transport values . Dataist_train_2019 provides transport count (person/day) however solution will be brought than transport count (person/week) . Dataist_train_2019 will be converted below .


# In[2825]:


#2019 Calendar has been brought to arrange transport count (person/week)
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

# Image url has been added
image_url = "https://www.pngitem.com/pimgs/m/40-405470_2019-calendar-png-transparent-hd-photo-calendar-2019.png"

# Image has been shown
Image(url=image_url)


# In[2826]:


#data has been created according days . 3401993 is person/day . It has been used for calculation .
data = {
    'Year': [2019] * 53,
    'Transport Type': ['Train'] * 53,
    'week of the year': list(range(1, 54)),
    'values' : [7*3401993]*53 
}

dataistm = pd.DataFrame(data)
dataistm


# In[2827]:


#2 values which had been assigned before randomly however they need be fixed and it is occured below .

dataistm.iloc[0, dataistm.columns.get_loc('values')] = 4*3401993
dataistm.iloc[52, dataistm.columns.get_loc('values')] = 3*3401993


# In[2828]:


dataistm


# In[2829]:


datacreate #data has been checked .


# In[2830]:


datacreate['values'] = datacreate['values'].astype(int) #Converting has been completed .


# In[2831]:


inner_merge=pd.merge(datacreate, dataistm,on=["week of the year"]) #2 different datas have been merged by ' week of the year'


# In[2832]:


inner_merge.shape #Shape has been controlled .


# In[2833]:


inner_merge #data has been controlled 


# In[2834]:


#Data has been created values which have come from Istanbul and Ireland . Individually or together values of datas which have been taken Istanbul and Ireland have been shown by plot , bar , scatter . Different libraries have been used .


# In[2835]:


#Ireland train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x'] / 1000 # Values have been divided by 1000

plt.figure(figsize=(20, 10)) # Size has been adjusted
plt.plot(weeks, values, marker='o', linestyle='-') # Plot showroom has been selected
for i, txt in enumerate(values):
    plt.annotate(f'{int(txt*1):,}', (weeks[i], values[i]), textcoords="offset points" , xytext=(0,10), ha='center') #Details have been created .

plt.title('Transport Values According Weeks In Ireland 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values/1000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()



# In[2836]:


#The max and the min value among Ireland values have been shown . Apperantly which week has max value and which week has min value have been displayed .


# In[2837]:


#Istanbul train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 10)) # Size has been adjusted
plt.plot(weeks, values, marker='o', linestyle='-') # Plot showroom has been selected
for i, txt in enumerate(values):
    plt.annotate(f'{int(txt*1):,}', (weeks[i], values[i]), textcoords="offset points" , xytext=(0,10), ha='center') #Details have been created .

plt.title('Train Transport Values According Weeks In Istanbul 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values(Million)')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()



# In[2838]:


#The max and the min value among Istanbul values have been shown . Apperantly which weeks have max value and which week has min value have been displayed .


# In[2839]:


#Ireland train transport values have been showed according weeks by scatter .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x']  

plt.figure(figsize=(20, 5)) # Size has been adjusted
plt.scatter(weeks, values) # Plot showroom has been selected

plt.title('Train Transport Values According Weeks In Ireland 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values(Million)')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()


# In[2840]:


#Istanbul train transport values have been shown according weeks by scatter .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 5)) # Size has been adjusted
plt.scatter(weeks, values) # Plot showroom has been selected

plt.title('Train Transport Values According Weeks In Istanbul 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values(Million)')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()


# In[2841]:


#Ireland train transport values have been displayed according weeks by bar .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 5))
plt.bar(weeks, values, width=0.5, align='center', alpha=0.7)  # Bar graph has been occured 


plt.title('Train Transport Values According Weeks In Ireland 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values(Million)')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()


# In[2842]:


#Istanbul train transport values have been displayed according weeks by bar .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 5))
plt.bar(weeks, values, width=0.5, align='center', alpha=0.7)  # Bar graph has been occured 


plt.title('Train Transport Values According Weeks In Istanbul 2019 ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values(Million)')
plt.grid(True)
plt.xticks(weeks) #All weeks have been shown
plt.legend()
plt.show()


# In[2843]:


#Values of Istanbul and values of Ireland have been shown by same bar&plot


# In[2844]:


import seaborn as sns #Library has been uploaded
import matplotlib.pyplot as plt #Library has been uploaded

plt.figure(figsize=(15, 6)) #Size has been fixed

sns.lineplot(data=inner_merge, x='week of the year', y='values_x', label='Ireland') #First line has been assigned
sns.lineplot(data=inner_merge, x='week of the year', y='values_y', label='Istanbul') #Second line has been assigned

plt.title('Istanbul and Ireland Train Transport Over Weeks 2019') #Plot has been named
plt.xlabel('Weeks')
plt.ylabel('Values (Million)')
plt.legend()
plt.xticks(weeks) #All weeks have been shown
plt.grid(True) #Grid has been created
plt.show()


# In[2845]:


#According the plot , Istanbul train transport values are greater than Ireland train transport values for every week . Min distance between of them has been shown 


# In[2846]:


import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'].unique()
ireland = inner_merge.groupby('week of the year')['values_x'].mean() #New data has been named
istanbul = inner_merge.groupby('week of the year')['values_y'].mean() #New data has been named

plt.figure(figsize=(18, 6))
plt.bar(weeks, ireland, label='Ireland', alpha=0.4)
plt.bar(weeks, istanbul, label='Istanbul', alpha=0.9, bottom=ireland)
plt.xlabel("Weeks", fontsize=13)
plt.ylabel("Value", fontsize=13)
plt.title("Ireland and Istanbul Train Transport by Week")
plt.xticks(weeks)
plt.legend()
plt.show()


# In[2847]:


#According the bar , Istanbul train transport values are greater than Ireland train transport values for every week . Min distance or max distance between of datas have been shown that can track here .  


# In[2848]:


inner_merge #data has been displayed again


# In[2849]:


import altair as alt #library has been uploaded 
import pandas as pd #library has been uploaded 


# In[2850]:


data = inner_merge.rename(columns={'values_y': 'Istanbul_transport', 'values_x': 'Ireland_transport'}) #A few columns have been named


# In[2851]:


data #Data has been shown


# In[2852]:


#Ireland and Istanbul datas have been shown together by altair library .

alt.Chart(data).mark_point(filled=True).encode(
    alt.X('week of the year:Q', scale=alt.Scale(zero=False)), #Horizontal X has been assigned
    alt.Y('Ireland_transport:Q', scale=alt.Scale(zero=False)), #Vertical Y has been assigned
    alt.Size('Year:N'), #Size has been selected 
    alt.Color('Istanbul_transport:Q'), 
    alt.OpacityValue(0.35),
    alt.Order('Year:Q', sort='descending'),
    tooltip = [alt.Tooltip('Istanbul_transport:N'), #While mouse indicates points , what wanna be displayed has been arranged 
               alt.Tooltip('week of the year:Q'),
               alt.Tooltip('Ireland_transport:Q')
              ]
)


# In[2853]:


#According 2019 Train Transport values , there is huge difference between of Istanbul and Ireland . What can be reason ? Istanbul is known that there are a lot of people live there . Ireland population is known being less due to island or other reasons . Istanbul and Ireland population values will have uploaded and values are gonna be read with population informations .


# In[2854]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2855]:


data=pd.read_csv('pivot.csv') #data is read 


# In[2856]:


data


# In[2857]:


#data .xsl imagine has been shared

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")  #Deskopt has been choosen

image_path = os.path.join(desktop_path, "Istanbul.imagine.png")  #File has been choosen 

img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')  #Grids have been closed
plt.show()  # Görseli görüntüle


# In[2858]:


#According imagine and file Istanbul population has been created .

import pandas as pd

# Sample dataframe has been created 
data = {
    'Year': list(range(2013, 2023)),
    'values': [14160467, 14377018, 14657434, 14804116, 15029231, 15067724, 15519267, 15462452, 15840900, 15907951],
    'Location' : ['Istanbul']*10
}

istanbul_population = pd.DataFrame(data)
istanbul_population


# In[2859]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ireland_population=pd.read_csv('PEA15.20240105T060158.csv') #data is read 
ireland_population


# In[2860]:


#2019 year has been targetted and choosen for comprasion values which had already been read .


# In[2861]:


filtre = (istanbul_population['Year'] == 2019) 
istanbul_population_2019= istanbul_population[filtre]
istanbul_population_2019


# In[2862]:


filtre = (ireland_population['Year'] == 2019)

ireland_population_2019=ireland_population[filtre]
ireland_population_2019


# In[2863]:


ireland_population_2019['VALUE'].replace(4958.5, 4958500, inplace=True)
ireland_population_2019['VALUE'] = ireland_population_2019['VALUE'].astype(int)


# In[2864]:


ireland_population_2019 #data has been created


# In[2865]:


population_merge=pd.merge(ireland_population_2019, istanbul_population_2019,on=["Year"]) #2 different datas have been merged by ' year'


# In[2866]:


population_merge #after mercing data has been controlled


# In[2867]:


population_merge=population_merge.drop(['STATISTIC Label','Component','Location'],axis=1) #Several columns have been wiped out .


# In[2868]:


population_merge #Data has been controlled


# In[2869]:


population_merge = population_merge.rename(columns={'VALUE': 'Ireland Population', 'values': 'Istanbul Population'}) #Some names have been renamed .


# In[2870]:


population_merge #data has been controlled


# In[2871]:


inner_merge #another data has been brought to merge


# In[2872]:


data=pd.merge(inner_merge, population_merge,on=["Year"]) #2 different datas have been merged by ' year'


# In[2873]:


data #Data has been controlled


# In[2874]:


data = data.rename(columns={'values_x': 'Ireland Transport', 'values_y': 'Istanbul Transport'}) #Some columns have been renamed


# In[2875]:


data=data.drop(['UNIT'],axis=1) #Unıt column has been deleted


# In[2876]:


data


# In[2877]:


#Columns have been arrayed .

fixing_columns = ['week of the year','Year', 'Transport Type', 'Ireland Transport', 'Istanbul Transport', 'Ireland Population', 'Istanbul Population']
data=data[fixing_columns]
data


# In[2878]:


#Percentages columns have been created .

data['Percentage of transportation Ireland'] = (data['Ireland Transport'] * 100) / data['Ireland Population']
data['Percentage of transportation Istanbul'] = (data['Istanbul Transport'] * 100) / data['Istanbul Population']


# In[2879]:


data


# In[2880]:


#Percentage of transportation Ireland and Percentage of transportation Istanbul have been displayed by bar .

import matplotlib.pyplot as plt

weeks = data['week of the year'].unique()
ireland = data.groupby('week of the year')['Percentage of transportation Ireland'].mean() #New data has been named
istanbul = data.groupby('week of the year')['Percentage of transportation Istanbul'].mean() #New data has been named

plt.figure(figsize=(18, 6))
plt.bar(weeks, ireland, label='Percentages of Ireland', alpha=0.4)
plt.bar(weeks, istanbul, label='Percentages of Istanbul', alpha=0.9, bottom=ireland)
plt.xlabel("Weeks", fontsize=13)
plt.ylabel("Value", fontsize=13)
plt.title("Percentages of Ireland and Percentages of Istanbul Train Transport by Week")
plt.xticks(weeks)
plt.legend()
plt.show()


# In[2881]:


#No doubt , all values have shown that percentages of transportation Istanbul are greater than percentages of transportation Ireland . Beside that , as has occured for a lot of weeks , transport of Istanbul is greater than population of Istanbul . 


# In[2882]:


#Comparation between Istanbul Transport and Istanbul Population by Week has been displayed by bar .

import matplotlib.pyplot as plt

grouped_transport = data.groupby('week of the year')['Istanbul Transport'].sum()
grouped_population = data.groupby('week of the year')['Istanbul Population'].sum()

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.bar(grouped_transport.index, grouped_transport, color='g', label='Istanbul Transport')
ax1.set_xlabel('week of the year')
ax1.set_ylabel('Istanbul Transport', color='y')
ax1.tick_params(axis='y', labelcolor='y')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(grouped_population.index, grouped_population, color='w', marker='o', label='Istanbul Population')
ax2.set_ylabel('Istanbul Population', color='y')
ax2.tick_params(axis='y', labelcolor='y')
ax2.legend(loc='upper right')

plt.title('Istanbul Transport and Population by Week')
plt.show()


# In[2883]:


#Comparation between Ireland Transport and Ireland Population by Week has been displayed by bar .

import matplotlib.pyplot as plt

grouped_transport = data.groupby('week of the year')['Ireland Transport'].sum()
grouped_population = data.groupby('week of the year')['Ireland Population'].sum()

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.bar(grouped_transport.index, grouped_transport, color='r', label='Ireland Transport')
ax1.set_xlabel('week of the year')
ax1.set_ylabel('Ireland Transport', color='y')
ax1.tick_params(axis='y', labelcolor='y')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(grouped_population.index, grouped_population, color='w', marker='o', label='Ireland Population')
ax2.set_ylabel('Ireland Population', color='y')
ax2.tick_params(axis='y', labelcolor='y')
ax2.legend(loc='upper right')

plt.title('Ireland Transport and Population by Week')
plt.show()


# In[2884]:


data['Percentage of transportation Istanbul']


# In[2885]:


data['Percentage of transportation Ireland']


# In[2886]:


from scipy import stats
t_statistic, p_value = stats.ttest_ind(data['Percentage of transportation Ireland'], data['Percentage of transportation Istanbul'], equal_var=False)
print(f"T statictic: {t_statistic}")
print(f"P value: {p_value}")


# In[2887]:


#The p-value determines whether the obtained correlation is random or not. If the p-value is lower than the alpha level (less than 0.05), then it is accepted  . According the result , p-value has come smaller than 0.05 . That's meaning there is huge difference between percentage of transportation Ireland and Istanbul . As long as P value is closer '0' , ıt means that is not coming coincidentally , ıt has occured statistically .  


# In[2888]:


from scipy.stats import f_oneway

# ANOVA has been exacuted .
statistic, p_value = f_oneway(data['Percentage of transportation Ireland'], data['Percentage of transportation Istanbul'])

# Results have been marked
alpha = 0.05
if p_value < alpha:
    print("Statistically significant difference is found between the groups")
else:
    print("Statistically significant difference is not found between the groups.")


# In[2889]:


#This indicates a significant difference between the transportation usage percentages of the two locations.This result suggests that there is a difference between Ireland and Istanbul in terms of train transportation usage rates, or at least statistically indicates a difference .


# In[2890]:


import pandas as pd
from scipy.stats import spearmanr

corr, p_value = spearmanr(data['Percentage of transportation Ireland'], data['Percentage of transportation Istanbul'])
print(f"Spearman Correlation Coefficient: {corr}")
print(f"P value: {p_value}")


# In[2891]:


#The Spearman correlation coefficient provides information about the direction and strength of the relationship between two variables. A positive Spearman correlation coefficient indicates that the two variables tend to increase together. In other words, when one variable's value increases, the other variable's value oftenly increases.The result is positive and it can be said that there is a moderate positive relationship between the weekly train usage percentages of Ireland and Istanbul


# In[2892]:


import pandas as pd
from scipy.stats import kruskal

# Kruskal-Wallis Test  has been executed .
statistic, p_value = kruskal(data['Percentage of transportation Ireland'], data['Percentage of transportation Istanbul'])

print("Kruskal-Wallis Test Statistic:", statistic)
print("P value:", p_value)


# In[2893]:


#When have a look P-value and Kruskal-Wallis value there are something be occured as before as other test , there is statistically meaningfully difference between of Istanbul and Ireland transport datas .P-value is smaller than 0.05 and also has been supported by Kruskal-Wallis value . 


# In[2894]:


import pandas as pd
from scipy.stats import mannwhitneyu

# Mann-Whitney Test has been executed .
stat, p = mannwhitneyu(data['Percentage of transportation Ireland'], data['Percentage of transportation Istanbul'])
print(f"Mann-Whitney U Test Statistic : {stat}")
print(f"P value: {p}")


# In[2895]:


#According to these results, it can be stated that there is a statistically significant difference between the percentage of weekly train transportation usage based on the population of Ireland and that of Istanbul.the Mann-Whitney U Test indicates that there is a difference in train usage percentages between Ireland and Istanbul.


# In[2896]:


#Correlation of percentages of Istanbul and Ireland has been examined . Several methods have been used for data . Apperantly that result has been taken . What is that , there is strongly relationship between population and transport at Istanbul and Ireland . 


# In[2897]:


#According this values , 2023 Transport values can be predictable . Using Machine Learnin model the objective will have approached . 


# In[2898]:


istanbul_population


# In[2899]:


y = istanbul_population['values']
X = istanbul_population.drop(['values','Location'],axis=1)


# In[2900]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state= 0)


# In[2901]:


## Applying ML Models
## Linear regression


# In[2902]:


lrm = LinearRegression()
lrm.fit(X_train, Y_train) #fit an OLS model


# In[2903]:


y_preds_train = lrm.predict(X_train)
y_preds_test = lrm.predict(X_test)  #making predictions


# In[2904]:


print("R-squared of the model in training set is: {}".format(lrm.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lrm.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test) / Y_test)) * 100))


# In[2905]:


## Ridge Regression


# In[2906]:


y = istanbul_population['values']
X = istanbul_population.drop(['values','Location'],axis=1)


# In[2907]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 2)


# In[2908]:


# Using GridSearch for parameter optimization
ridgeregr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

ridgeregr.fit(X_train, Y_train)

ridge = ridgeregr.best_estimator_


# In[2909]:


# Making predictions here
y_preds_train = ridge.predict(X_train)
y_preds_test_ridge = ridge.predict(X_test)

print("R-squared of the model in training set is: {}".format(ridge.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(ridge.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_preds_test_ridge)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_ridge) / Y_test)) * 100))


# In[2910]:


y_preds_train


# In[2911]:


y_preds_test_ridge 


# In[2912]:


## Lasso Regression


# In[2913]:


y = istanbul_population['values']
X = istanbul_population.drop(['values','Location'],axis=1)


# In[2914]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state= 1)


# In[2915]:


# using GridSearch for parameter optimization
lassoregr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

lassoregr.fit(X_train, Y_train)

lasso = lassoregr.best_estimator_


# In[2916]:


# We are making predictions here
y_preds_train = lasso.predict(X_train)
y_preds_test_lasso = lasso.predict(X_test)

print("R-squared of the model in training set is: {}".format(lasso.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lasso.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test_lasso)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_lasso) / Y_test)) * 100))


# In[2917]:


## Decision Tree Regressor


# In[2918]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[2919]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[2920]:


y_pred_DT = DTregressor.predict(X_test)


# In[2921]:


print("R-squared of the model in training set is: {}".format(DTregressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(DTregressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_DT)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_DT) / Y_test)) * 100))


# In[2922]:


## Random Forest


# In[2923]:


y = istanbul_population['values']
X = istanbul_population.drop(['values','Location'],axis=1)


# In[2924]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 2)


# In[2925]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)


# In[2926]:


y_pred_random = regressor.predict(X_test)


# In[2927]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_random)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_random) / Y_test)) * 100))


# In[2928]:


import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import numpy as np


# In[2929]:


y = istanbul_population['values']
X = istanbul_population.drop(['values','Location'],axis=1)


# In[2930]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[2931]:


parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svr = SVR()
regressor = GridSearchCV(svr, parameters)
regressor.fit(X_train, Y_train)


# In[2932]:


y_pred = regressor.predict(X_test)


# In[2933]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred) / Y_test)) * 100))


# In[2934]:


import matplotlib.pyplot as plt
r_squared_values = [0.9812504569718, 0.97737451618657, 0.984838610138315 , 1.0 , 0.974143489362347] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Training Values from Models')
plt.show()


# In[2935]:


import matplotlib.pyplot as plt
r_squared_test_values = [0.967968814116784,0.987507166338252, 0.956426556206693 , 0.822564988389076 , 0.867559751354687] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_test_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Test Values from Models')
plt.show()


# According results , R-squared of Decision Tree model in training is greater than other . However , R-squared of Decision Tree model in training is not stronger . In fact , R-squared of entire models in training are greater . Hence , max R-squared of model in training which is Decision Tree has not been chosen . Ridge Regression has provided high values for training and test . They are not highest one's but Ridge Regression has lowest root mean squared error of the prediction . What does it mean ? It has said that there is no huge distance between prediction and reality . If the result is low , the result has provided high accurancy . As well as mean absolute percentage error of the prediction is low . That also supports why Ridge Regression have been choosen .
# 
# R-squared of the model in training set is: 0.9773745161865709
# -----Test set statistics-----
# R-squared of the model in test set is: 0.9875071663382521 
# Root mean squared error of the prediction is: 36449.36226213802
# Mean absolute percentage error of the prediction is: 0.23388592770090758

# In[2936]:


# 2023 Population Prediction
year_2023 = [[2023]]

# Pprediction has been completed by Ridge model 
value_2023 = ridge.predict(year_2023)

print(f"2023 prediction population: {value_2023[0]}")


# In[2937]:


istanbul_population


# In[2938]:


new_data = {
    'Year': [2023],
    'values': [16149146],
    'Location': ['Istanbul']
}

new_row = pd.DataFrame(new_data)
istanbul_population = istanbul_population.append(new_row, ignore_index=True)


# In[2939]:


istanbul_population


# In[2940]:


ireland_population


# In[2941]:


dataist_train_2019y


# In[2942]:


import pandas as pd

data = {
    'week of the year': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'values': [549533.0, 839022.0, 819359.0, 860745.0, 912795.0, 912612.0, 940476.0, 952291.0, 929391.0, 941919.0, 924998.0, 868205.0, 1003871.0, 933575.0, 945662.0, 842186.0, 801296.0, 919255.0, 817933.0, 904983.0, 902415.0, 973025.0, 826269.0, 925516.0, 883208.0, 905636.0, 982288.0, 919158.0, 903958.0, 926491.0, 832452.0, 807393.0, 868677.0, 862939.0, 897355.0, 933362.0, 969818.0, 990123.0, 1031937.0, 986159.0, 1054749.0, 1014017.0, 1028522.0, 924586.0, 1019705.0, 1038825.0, 1062275.0, 1113668.0, 1080791.0, 1151098.0, 1173473.0, 538511.0, 538511.0]
}

df = pd.DataFrame(data)

# 'values' sütununun toplamını hesaplama
total = df['values'].sum()

print("Total values:", total)


# In[2943]:


import pandas as pd

data = {
    'Year' : [2019],
    'Transport Type' : ['Train'],
    'Passenger Count (Person/Year)' : [48687017]
}

datairl_train_2019y = pd.DataFrame(data)

datairl_train_2019y


# In[2944]:


population_merge


# In[2945]:


final_merge=pd.merge(population_merge, datairl_train_2019y,on=['Year']) #2 different datas have been merged by 'Year'


# In[2946]:


final_merge=pd.merge(final_merge, dataist_train_2019y,on=['Year']) #2 different datas have been merged by ' Year'


# In[2947]:


final_merge


# In[ ]:




