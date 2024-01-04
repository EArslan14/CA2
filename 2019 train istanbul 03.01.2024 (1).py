#!/usr/bin/env python
# coding: utf-8

# In[296]:


import pandas as pd

dataist = pd.read_csv('ıstanbulverı1.csv', encoding='iso-8859-9')
print(dataist)


# In[297]:


dataist


# In[298]:


new_column_names = {
    'Yıl': 'Year',
    'Yolcu Sayısı (Kişi/Gün)': 'Passenger Count (Person/Day)',
    'Yolculuk Türü': 'Transport Type'
}

dataist = dataist.rename(columns=new_column_names)

print(dataist)


# In[299]:


dataist


# In[300]:


dataist['Transport Type'] = dataist['Transport Type'].replace('Raylı Sistemler', 'Train')
dataist['Transport Type'] = dataist['Transport Type'].replace('Deniz Ulaşımı', 'Sea')
dataist['Transport Type'] = dataist['Transport Type'].replace('Karayolu', 'Road')


# In[301]:


dataist


# In[302]:


dataist.isnull().sum()


# In[303]:


filtre = (dataist['Year'] == 2019) & (dataist['Transport Type'] == 'Train') 
dataist_train_2019= dataist[filtre]


# In[304]:


dataist_train_2019


# In[305]:


dataist_train_2019.info()


# In[306]:


dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)'].str.replace(',', '').astype(int)


# In[307]:


dataist_train_2019.info()


# In[308]:


import pandas as pd

# 'Person/Day' data is converted being 'Person/Year' 
dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)']
dataist_train_2019['Passenger Count (Person/Year)'] = dataist_train_2019['Passenger Count (Person/Day)'] * 365  # veya artık yıllarda 366

# Unnecessary datas are wiped out
dataist_train_2019y= dataist_train_2019.drop(columns=['Passenger Count (Person/Day)'])

print(dataist_train_2019y)


# In[309]:


dataist_train_2019y


# In[310]:


import pandas as pd

# 'Person/Day' datas are converted being 'Person/Month' 
dataist_train_2019['Passenger Count (Person/Day)'] = dataist_train_2019['Passenger Count (Person/Day)']
dataist_train_2019['Passenger Count (Person/Month)'] = dataist_train_2019['Passenger Count (Person/Day)'] * 365 / 12  # veya artık yıllarda 366

# Gereksiz sütunu kaldırma
dataist_train_2019m = dataist_train_2019.drop(columns=['Passenger Count (Person/Day)'])

print(dataist_train_2019m)


# In[311]:


dataist_train_2019m


# In[312]:


dataist_train_2019m.info()


# In[313]:


dataist_train_2019m['Passenger Count (Person/Month)'] = dataist_train_2019m['Passenger Count (Person/Month)'].astype(int)
#Data is converted integer version

print(dataist_train_2019m)


# In[314]:


dataist_train_2019m


# In[315]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

# Görselin URL'si
image_url = "https://www.pngitem.com/pimgs/m/40-405470_2019-calendar-png-transparent-hd-photo-calendar-2019.png"

# Görseli Jupyter Notebook'a ekleme
Image(url=image_url)


# In[317]:


data = {
    'Year': [2019] * 53,
    'Transport Type': ['Train'] * 53,
    'week of the year': list(range(1, 54)),
    'values' : [7*3401993]*53 
}

dataistm = pd.DataFrame(data)
dataistm


# In[318]:


dataistm.iloc[0, dataistm.columns.get_loc('values')] = 4*3401993
dataistm.iloc[52, dataistm.columns.get_loc('values')] = 3*3401993


# In[319]:


dataistm


# In[ ]:





# In[124]:


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


# In[125]:


data


# In[126]:


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
df['Year'] = df['Year'].astype(int)

# DataFrame'i gösterme
print(df)


# In[127]:


df


# In[128]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Image

# Görselin URL'si
image_url = "https://www.pngitem.com/pimgs/m/40-405470_2019-calendar-png-transparent-hd-photo-calendar-2019.png"

# Görseli Jupyter Notebook'a ekleme
Image(url=image_url)


# In[129]:


train_rows_2019m


# In[130]:


df


# In[131]:


df.shape


# In[132]:


datamonths.shape


# In[133]:


df.info()


# In[134]:


datamonths.info()


# In[135]:


inner_merge=pd.merge(df, datamonths)


# In[136]:


inner_merge.shape


# In[137]:


inner_merge


# In[138]:


inner_merge_total=pd.merge(df, datamonths, on=["Year"])


# In[139]:


inner_merge_total.shape


# In[140]:


inner_merge_total


# In[141]:


inner_merge_total = inner_merge_total.drop(['STATISTIC', 'Statistic Label','TLIST(W1)',], axis=1)


# In[142]:


inner_merge_total


# In[ ]:




