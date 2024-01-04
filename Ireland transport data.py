#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


data


# In[39]:


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
df = pd.DataFrame(data)

# DataFrame'i gösterme
print(df)


# In[40]:


df


# In[41]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

# Görselin URL'si
image_url = "https://www.pngitem.com/pimgs/m/40-405470_2019-calendar-png-transparent-hd-photo-calendar-2019.png"

# Görseli Jupyter Notebook'a ekleme
Image(url=image_url)


# In[ ]:





# In[42]:


import pandas as pd

# Veri çerçevesini oluşturun
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

df = pd.DataFrame(data)

# NaN değerlerini sütunun ortalama değeri ile doldurun
mean_value = df['values'].mean()
df['values'].fillna(mean_value, inplace=True)

print(df)


# In[43]:


df


# In[44]:


import pandas as pd

# Veri çerçevesini oluşturun
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

df = pd.DataFrame(data)

# NaN değerlerini sütunun medyan değeri ile doldurun
median_value = df['values'].median()
df['values'].fillna(median_value, inplace=True)

print(df)


# In[45]:


# Veri çerçevesini oluşturun (yukarıdaki aynı veri)
df = pd.DataFrame(data)

# NaN değerlerini sütunun mod değeri ile doldurun
mode_value = df['values'].mode()[0]  # Mode değeri, value_counts() sonucundan alınır
df['values'].fillna(mode_value, inplace=True)

print(df)


# In[46]:


df


# In[47]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veri seti
data = {
    'week of the year': list(range(1, 54)),
    'values': [549533, 839022, 819359, 860745, 912795, 912612, 940476, 952291, 929391, 941919, 924998, 868205, 1003871, 933575, 945662, 842186, 801296, 919255, 817933, 904983, 902415, 973025, 826269, 925516, 883208, 905636, 982288, 919158, 903958, 926491, 832452, 807393, 868677, 862939, 897355, 933362, 969818, 990123, 1031937, 986159, 1054749, 1014017, 1028522, 924586, 1019705, 1038825, 1062275, 1113668, 1080791, 1151098, 1173473, 538511, None]
}

df1 = pd.DataFrame(data)

# Eksik değerleri tahmin etmek için eksik olmayan değerleri eğitim veri seti olarak alalım
train_data = df1.dropna()

# 'week of the year' değerlerini bağımsız değişken olarak alalım
X_train = train_data[['week of the year']]
y_train = train_data['values']

# Eksik değerleri tahmin etmek için model oluşturalım
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Tüm veri setindeki eksik değerleri tahmin edelim
X_pred = df1[['week of the year']]
df1['values_predicted'] = regressor.predict(X_pred)

print(df1)


# In[48]:


df


# In[50]:


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


# In[53]:


scaler = StandardScaler()

df_scale = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)


# In[54]:


df_scale.head(10)


# In[55]:


y = df['values']
X = df_scale.drop(['values'],axis=1)


# In[155]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2 , random_state= 4)


# In[156]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[157]:


## Applying ML Models
## Linear regression


# In[158]:


lrm = LinearRegression()
lrm.fit(X_train, Y_train) #fit an OLS model


# In[159]:


y_preds_train = lrm.predict(X_train)
y_preds_test = lrm.predict(X_test)  #making predictions


# In[160]:


print("R-squared of the model in training set is: {}".format(lrm.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lrm.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test) / Y_test)) * 100))


# In[161]:


y_preds_train


# In[162]:


y_preds_test 


# In[163]:


# Using GridSearch for parameter optimization
ridgeregr = GridSearchCV(Ridge(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

ridgeregr.fit(X_train, Y_train)

ridge = ridgeregr.best_estimator_


# In[164]:


# Making predictions here
y_preds_train = ridge.predict(X_train)
y_preds_test_ridge = ridge.predict(X_test)

print("R-squared of the model in training set is: {}".format(ridge.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(ridge.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_preds_test_ridge)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_ridge) / Y_test)) * 100))


# In[165]:


# using GridSearch for parameter optimization
lassoregr = GridSearchCV(Lasso(),
                    param_grid={
                        'alpha': [0.01, 0.1, 1]
                    }, verbose=1)

lassoregr.fit(X_train, Y_train)

lasso = lassoregr.best_estimator_


# In[166]:


# We are making predictions here
y_preds_train = lasso.predict(X_train)
y_preds_test_lasso = lasso.predict(X_test)

print("R-squared of the model in training set is: {}".format(lasso.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(lasso.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(rmse(Y_test, y_preds_test_lasso)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_preds_test_lasso) / Y_test)) * 100))


# In[167]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[168]:


from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[169]:


y_preds_train = DTregressor.predict(X_train)
y_preds_train


# In[170]:


y_pred_DT = DTregressor.predict(X_test)


# In[171]:


y_pred_DT


# In[172]:


print("R-squared of the model in training set is: {}".format(DTregressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(DTregressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_DT)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_DT) / Y_test)) * 100))


# In[173]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)


# In[174]:


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, Y_train)


# In[175]:


y_pred_random = regressor.predict(X_test)


# In[176]:


y_pred_random


# In[177]:


print("R-squared of the model in training set is: {}".format(regressor.score(X_train, Y_train)))
print("-----Test set statistics-----")
print("R-squared of the model in test set is: {}".format(regressor.score(X_test, Y_test)))
print("Root mean squared error of the prediction is: {}".format(mse(Y_test, y_pred_random)**(1/2)))
print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((Y_test - y_pred_random) / Y_test)) * 100))


# In[ ]:




