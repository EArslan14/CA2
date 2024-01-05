#!/usr/bin/env python
# coding: utf-8

# In[851]:


#First data which is about transport Istanbul/Turkiye . This data has provided how many people had taken transport for several years . Especially 2019 years( the year before corona ) train transport datas are looked for to compare Ireland train transport datas .
#Data has been read.

import pandas as pd

dataist = pd.read_csv('ıstanbulverı1.csv', encoding='iso-8859-9')  #encoding has been added due to different language . Components of data are Turkish . 
print(dataist)


# In[852]:


dataist #data has been showed .


# In[853]:


#Columns of the data have been assigned due to be clear . They naturally were written being Turkish and they have been changed .

new_column_names = {
    'Yıl': 'Year',
    'Yolcu Sayısı (Kişi/Gün)': 'Passenger Count (Person/Day)',
    'Yolculuk Türü': 'Transport Type'
}

dataist = dataist.rename(columns=new_column_names)

print(dataist)


# In[854]:


dataist #Action has been controlled , names seem properly .


# In[855]:


#Other Turkish datas have been converted being English .

dataist['Transport Type'] = dataist['Transport Type'].replace('Raylı Sistemler', 'Train')
dataist['Transport Type'] = dataist['Transport Type'].replace('Deniz Ulaşımı', 'Sea')
dataist['Transport Type'] = dataist['Transport Type'].replace('Karayolu', 'Road')


# In[856]:


dataist #Control has been done .


# In[857]:


dataist.isnull().sum() #Data has been controlled obtain 'Nan' values or not . As long as the results seem '0' , data is ready to evaluate .


# In[858]:


#The object was approaching 2019 train transport datas . Filtre command is used to shrink actual data .

filtre = (dataist['Year'] == 2019) & (dataist['Transport Type'] == 'Train') 
dataist_train_2019= dataist[filtre]


# In[859]:


dataist_train_2019 #data has been controlled to be sure . The year before corona which had been targetted .


# In[860]:


dataist_train_2019.info() #data is examined . Types of components are known being important if there is request for managing data .


# In[861]:


#Passenger Count values were observed object and with this code which is settled below Passenger Count values are converted being integer .

dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)'].str.replace(',', '').astype(int)


# In[862]:


dataist_train_2019.info() #Data types have been controlled after using code .


# In[863]:


#For later , Passenger Count (Person/Year) has been calculated .

import pandas as pd

# 'Person/Day' data is converted being 'Person/Year' 
dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)']
dataist_train_2019['Passenger Count (Person/Year)'] = dataist_train_2019['Passenger Count (Person/Day)'] * 365  

# Unnecessary datas are wiped out
dataist_train_2019y= dataist_train_2019.drop(columns=['Passenger Count (Person/Day)'])

print(dataist_train_2019y)


# In[864]:


dataist_train_2019y #data has been displayed


# In[865]:


dataist_train_2019y.info() #data had been examined 


# In[866]:


dataist_train_2019y.describe(include=object) #data is examined


# In[867]:


dataist_train_2019.describe() #data is examined


# In[868]:


dataist_train_2019 #data which will be used is displayed .


# In[869]:


#Data is ready for using with 2019 Train Transport values . 
#Another data which obtains Ireland Transport datas . It has been read below .


# In[870]:


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


# In[871]:


data #data has been showed .


# In[872]:


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

# DataFrame is showed
print(df)


# In[873]:


df #DataFrame had been displayed


# In[874]:


df.shape


# In[875]:


df.describe()


# In[876]:


df.info()


# In[877]:


df.isnull().sum() #Data has been controlled obtain 'Nan' value .


# In[878]:


df[df.isna().any(axis=1)] #Data has been controlled obtain 'Nan' value .
print(df[df.isna().any(axis=1)])


# In[879]:


df.columns #For amputing names of columns have been taken .


# In[880]:


df = df.drop(['STATISTIC', 'Statistic Label', 'TLIST(W1)', 'Luas Line '], axis=1) #Negligible columns have been dropped . Another data which contains Istanbul values is known that these columns do not exist there .


# In[881]:


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

# In[882]:


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


# In[883]:


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


# In[884]:


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


# In[885]:


#Mean , Mod and Median values have been taken . Several Machine Learning Models will have used and the conclusion will be given after to compare all .


# In[886]:


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


# In[887]:


dftest=df[ :52] #Until 52.rows have been selected and created test dataframe .


# In[888]:


dftest #test dataframe has been displayed 


# In[889]:


dftest.info() #Data has been examined getting knowledge . Type of value and type of Weeks of the year have to be changed being integer .


# In[890]:


dftest['VALUE'] = dftest['VALUE'].astype(int) #Converting has been completed .


# In[891]:


dftest['Weeks of the year'] = dftest['Weeks of the year'].astype(int) #Converting has been completed .


# In[892]:


#X and y values have been derivated .

y = dftest['VALUE'] 
X = dftest.drop(['VALUE'],axis=1)


# In[893]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2 , random_state= 4) #Train and Test values have been assigned . %80 train values , %20 test values have been taken . 


# In[894]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[895]:


## Applying ML Models


# In[896]:


# Linear regression


# In[897]:


lrm = LinearRegression()
lrm.fit(X_train, Y_train) #fit an OLS model


# In[898]:


y_preds_train = lrm.predict(X_train)
y_preds_test = lrm.predict(X_test)  #making predictions


# In[899]:


print("R-squared of the model in training set is: {}".format(lrm.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lrm.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test) / Y_test)) * 100))


# In[900]:


## Ridge Regression


# In[901]:


# Using GridSearch for parameter optimization
ridgeregr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

ridgeregr.fit(X_train, Y_train)

ridge = ridgeregr.best_estimator_


# In[902]:


# Predictions have been made
y_preds_train = ridge.predict(X_train)
y_preds_test_ridge = ridge.predict(X_test)

print("R-squared of the model in training set is: {}".format(ridge.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(ridge.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_preds_test_ridge)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_ridge) / Y_test)) * 100))


# In[903]:


### Lasso Regression


# In[904]:


# using GridSearch for parameter optimization
lassoregr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

lassoregr.fit(X_train, Y_train)

lasso = lassoregr.best_estimator_


# In[905]:


# Predictions have been made
y_preds_train = lasso.predict(X_train)
y_preds_test_lasso = lasso.predict(X_test)

print("R-squared of the model in training set is: {}".format(lasso.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lasso.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test_lasso)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_lasso) / Y_test)) * 100))


# In[906]:


#### Decision Tree Regressor


# In[907]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[908]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[909]:


y_preds_train = DTregressor.predict(X_train)
y_preds_train


# In[910]:


y_pred_DT = DTregressor.predict(X_test)


# In[911]:


y_pred_DT


# In[912]:


print("R-squared of the model in training set is: {}".format(DTregressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(DTregressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_DT)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_DT) / Y_test)) * 100))


# In[913]:


##### Random Forest


# In[914]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state= 100)


# In[915]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)


# In[916]:


y_pred_random = regressor.predict(X_test)


# In[917]:


y_pred_random


# In[918]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_random)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_random) / Y_test)) * 100))


# In[919]:


import matplotlib.pyplot as plt
r_squared_values = [0.221021374150552, 0.221021371544315, 0.221021371544315 , 1.0 , 0.9327453294294139] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Training Values from Models')
plt.show()


# In[920]:


import matplotlib.pyplot as plt
r_squared_test_values = [-0.193743303365938, -0.193673935964648, -0.193742554697871 , 0.942035242036975 , -0.8878587040645121] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_test_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Test Values from Models')
plt.show()


# In[921]:


#Comparasion of several method Decision Tree seems greater way . But Decision Tree will not be taken to find Nan value because of overfitting . R-squared of the model in training set represents great value but R-squared of the model in test set represents low value . What does it mean ? That indicates there is something wrong here as overfitting . As long as test and train values closer each other , that way can be rational . Decision Tree model has not been taken because of dimention of train&test values .  


# In[922]:


#Datacreate which had been created by mode . That value has been accepted instead of NaNn value . For remaining it has been displayed below .


# In[923]:


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


# In[924]:


#Values which are coming from Ireland and Istanbul will be merged to compare them . Datas which contains Istanbul values has to be converted as datacreate .


# In[925]:


datacreate.shape #shape has been controlled


# In[926]:


dataist_train_2019.shape #shape has been controlled


# In[ ]:


#Dataist_train_2019 which contains Istanbul train transport values should be changed as datacreate which contains Ireland train transport values . Dataist_train_2019 provides transport count (person/day) however solution will be brought than transport count (person/week) . Dataist_train_2019 will be converted below .


# In[947]:


#2019 Calendar has been brougt to arrange transport count (person/week)
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

# Image url has been added
image_url = "https://www.pngitem.com/pimgs/m/40-405470_2019-calendar-png-transparent-hd-photo-calendar-2019.png"

# Image has been showed
Image(url=image_url)


# In[948]:


#data has been created according days . 3401993 is person/day . It has been used for calculation .
data = {
    'Year': [2019] * 53,
    'Transport Type': ['Train'] * 53,
    'week of the year': list(range(1, 54)),
    'values' : [7*3401993]*53 
}

dataistm = pd.DataFrame(data)
dataistm


# In[949]:


#2 values which had been assigned before randomly however they need be fixed and it is occured below .

dataistm.iloc[0, dataistm.columns.get_loc('values')] = 4*3401993
dataistm.iloc[52, dataistm.columns.get_loc('values')] = 3*3401993


# In[950]:


dataistm


# In[951]:


datacreate


# In[952]:


datacreate['values'] = datacreate['values'].astype(int) #Converting has been completed .


# In[956]:


inner_merge=pd.merge(datacreate, dataistm,on=["week of the year"]) #2 different datas have been merged by ' week of the year'


# In[957]:


inner_merge.shape #Shape has been controlled .


# In[958]:


inner_merge


# In[959]:


#Ireland train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x'] / 1000  # Values have been divided by 1000

plt.figure(figsize=(20, 10)) # Size has been adjusted
plt.plot(weeks, values, marker='o', linestyle='-') # Plot showroom has been selected
for i, txt in enumerate(values):
    plt.annotate(f'{int(txt*1):,}', (weeks[i], values[i]), textcoords="offset points" , xytext=(0,10), ha='center') #Details have been created .

plt.title('Transport Values According Weeks In Ireland ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values/1000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()



# In[960]:


#Istanbul train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 10)) # Size has been adjusted
plt.plot(weeks, values, marker='o', linestyle='-') # Plot showroom has been selected
for i, txt in enumerate(values):
    plt.annotate(f'{int(txt*1):,}', (weeks[i], values[i]), textcoords="offset points" , xytext=(0,10), ha='center') #Details have been created .

plt.title('Transport Values According Weeks In Istanbul ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values/1000000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()



# In[961]:


#Ireland train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x'] / 1000  # Values have been divided by 1000

plt.figure(figsize=(20, 5)) # Size has been adjusted
plt.scatter(weeks, values) # Plot showroom has been selected

plt.title('Transport Values According Weeks In Ireland ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values/1000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()


# In[962]:


#Istanbul train transport values have been plotted according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000000  # Values have been divided by 1000000

plt.figure(figsize=(20, 5)) # Size has been adjusted
plt.scatter(weeks, values) # Plot showroom has been selected

plt.title('Transport Values According Weeks In Istanbul ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values/1000000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()


# In[963]:


#Ireland train transport values have been displayed bar according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_x'] / 1000  # Values have been divided by 1000

plt.figure(figsize=(20, 5))
plt.bar(weeks, values, width=0.5, align='center', alpha=0.7)  # Bar graph has been occured 


plt.title('Transport Values According Weeks In Ireland ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Ireland Train Transport Values/1000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()


# In[964]:


#Istanbul train transport values have been displayed bar according weeks .

import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'] # Values have been created
values = inner_merge['values_y'] / 1000  # Values have been divided by 1000

plt.figure(figsize=(20, 5))
plt.bar(weeks, values, width=0.5, align='center', alpha=0.7)  # Bar graph has been occured 


plt.title('Transport Values According Weeks In Istanbul ') #Title has been named
plt.xlabel('Weeks')
plt.ylabel('Istanbul Train Transport Values/1000')
plt.grid(True)
plt.xticks(weeks) #All weeks have been showed
plt.legend()
plt.show()


# In[965]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

sns.lineplot(data=inner_merge, x='week of the year', y='values_x', label='Ireland')
sns.lineplot(data=inner_merge, x='week of the year', y='values_y', label='Istanbul')

plt.title('Istanbul and Ireland Train Transport Over Weeks')
plt.xlabel('Weeks')
plt.ylabel('Values / 1000000')
plt.legend()
plt.xticks(weeks) #All weeks have been showed
plt.grid(True)
plt.show()


# In[966]:


import matplotlib.pyplot as plt

weeks = inner_merge['week of the year'].unique()
ireland = inner_merge.groupby('week of the year')['values_x'].mean()
istanbul = inner_merge.groupby('week of the year')['values_y'].mean()

plt.figure(figsize=(18, 6))
plt.bar(weeks, ireland, label='Ireland', alpha=0.4)
plt.bar(weeks, istanbul, label='Istanbul', alpha=0.9, bottom=ireland)
plt.xlabel("Weeks", fontsize=13)
plt.ylabel("Value", fontsize=13)
plt.title("Ireland and Istanbul Train Transport by Week")
plt.xticks(weeks)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




