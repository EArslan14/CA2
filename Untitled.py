#!/usr/bin/env python
# coding: utf-8

# In[569]:


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


# In[570]:


data


# In[571]:


import pandas as pd

# Örnek veri
data = {
    'STATISTIC': ['TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01', 'TII01C01', 'TII01C01','TII01C01', 'TII01C01', 'TII01C01'],
    'Statistic Label': ['Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys','Passenger Journeys', 'Passenger Journeys', 'Passenger Journeys'],
    'TLIST(W1)' : ['2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019'] ,
    'Year': ['2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019', '2019', '2019','2019', '2019', '2019'],
    'Luas Line ' : ['All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines', 'All Luas lines','All Luas lines','All Luas lines', 'All Luas lines', 'All Luas lines' ],
    'Weeks of the year' : ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53'],
    'VALUE' : ['549533', '839022', '819359', '860745', '912795', '912612', '940476', '952291', '929391', '941919', '924998', '868205', '1003871', '933575', '945662', '842186', '801296', '919255', '817933', '904983', '902415', '973025', '826269', '925516', '883208', '905636', '982288', '919158', '903958', '926491', '832452', '807393', '868677', '862939', '897355', '933362', '969818', '990123', '1031937', '986159', '1054749', '1014017', '1028522', '924586', '1019705', '1038825', '1062275', '1113668', '1080791', '1151098', '1173473', '538511','NaN']
}


# DataFrame oluşturma
data = pd.DataFrame(data)

# DataFrame'i gösterme
print(data)


# In[572]:


data


# In[573]:


df = data.drop(['STATISTIC', 'Statistic Label', 'TLIST(W1)'], axis=1)


# In[574]:


df


# In[575]:


df.info()


# In[576]:


df.describe()


# In[577]:


df.isnull().sum()


# In[578]:


import pandas as pd

# Veri çerçevesini oluşturun
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN değerlerini sütunun medyan değeri ile doldurun
median_value = datacreate['values'].median()
datacreate['values'].fillna(median_value, inplace=True)

print(datacreate)


# In[579]:


import pandas as pd

# Veri çerçevesini oluşturun
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

datacreate = pd.DataFrame(data)

# NaN değerlerini sütunun mod değeri ile doldurun
mode_value = datacreate['values'].mode()[0]  # Mode değeri, value_counts() sonucundan alınır
datacreate['values'].fillna(mode_value, inplace=True)

print(datacreate)


# In[580]:


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


# In[581]:


dftest=df[ :52]


# In[582]:


dftest


# In[583]:


dftest.info()


# In[584]:


dftest['VALUE'] = dftest['VALUE'].astype(int)


# In[585]:


dftest['Weeks of the year'] = dftest['Weeks of the year'].astype(int)


# In[586]:


y = dftest['VALUE']
X = dftest.drop(['VALUE','Luas Line '],axis=1)


# In[587]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2 , random_state= 4)


# In[588]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[589]:


## Applying ML Models
## Linear regression


# In[590]:


lrm = LinearRegression()
lrm.fit(X_train, Y_train) #fit an OLS model


# In[591]:


y_preds_train = lrm.predict(X_train)
y_preds_test = lrm.predict(X_test)  #making predictions


# In[592]:


print("R-squared of the model in training set is: {}".format(lrm.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lrm.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test) / Y_test)) * 100))


# In[593]:


## Ridge Regression


# In[594]:


# Using GridSearch for parameter optimization
ridgeregr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

ridgeregr.fit(X_train, Y_train)

ridge = ridgeregr.best_estimator_


# In[595]:


# Making predictions here
y_preds_train = ridge.predict(X_train)
y_preds_test_ridge = ridge.predict(X_test)

print("R-squared of the model in training set is: {}".format(ridge.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(ridge.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_preds_test_ridge)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_ridge) / Y_test)) * 100))


# In[596]:


## Lasso Regression


# In[597]:


# using GridSearch for parameter optimization
lassoregr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

lassoregr.fit(X_train, Y_train)

lasso = lassoregr.best_estimator_


# In[598]:


# We are making predictions here
y_preds_train = lasso.predict(X_train)
y_preds_test_lasso = lasso.predict(X_test)

print("R-squared of the model in training set is: {}".format(lasso.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lasso.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test_lasso)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_lasso) / Y_test)) * 100))


# In[599]:


## Decision Tree Regressor


# In[600]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[601]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[602]:


y_preds_train = DTregressor.predict(X_train)
y_preds_train


# In[603]:


y_pred_DT = DTregressor.predict(X_test)


# In[604]:


y_pred_DT


# In[605]:


print("R-squared of the model in training set is: {}".format(DTregressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(DTregressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_DT)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_DT) / Y_test)) * 100))


# In[606]:


## Random Forest


# In[607]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state= 100)


# In[608]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)


# In[609]:


y_pred_random = regressor.predict(X_test)


# In[610]:


y_pred_random


# In[611]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_random)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_random) / Y_test)) * 100))


# In[612]:


import matplotlib.pyplot as plt
r_squared_values = [0.221021374150552, 0.221021371544315, 0.221021371544315 , 1.0 , 0.9327453294294139] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Training Values from Models')
plt.show()


# In[613]:


import matplotlib.pyplot as plt
r_squared_test_values = [-0.193743303365938, -0.193673935964648, -0.193742554697871 , 0.942035242036975 , -0.8878587040645121] #R-squared of the model in training sets values
model_names = ['Linear regression', 'Ridge Regression', 'Lasso Regression ' , 'Decision Tree' ,' Random Forest ']
plt.figure(figsize=(10, 10))
plt.bar(model_names, r_squared_test_values, color=['blue', 'green', 'red', 'black' , 'grey'])
plt.xlabel('Models' , fontsize=13)
plt.ylabel('R-squared Values', fontsize=13)
plt.title('R-squared Test Values from Models')
plt.show()


# In[614]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)
y_preds_train = DTregressor.predict(X_train)
y_preds_train


# In[615]:


y_pred_DT = DTregressor.predict(X_test)
y_pred_DT


# In[616]:


dftest


# In[617]:


from sklearn.tree import DecisionTreeRegressor

# Veri setini hazırlama
X_train = dftest['Weeks of the year'].values.reshape(-1, 1)  # Eğitim verisi, 'week of the year' kolonu
y_train = dftest['VALUE'].values  # Etiket, 'values' kolonu

# Karar ağacı regresyon modeli oluşturma
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X_train, y_train)

# 53. hafta tahminini yapma
predicted_value = tree_model.predict([[53]])  # 53. hafta için tahmin yapma

print(f"53. hafta tahmini değer: {predicted_value[0]}")


# In[623]:


datacreate


# In[ ]:




