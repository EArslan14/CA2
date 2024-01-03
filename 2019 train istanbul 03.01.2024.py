#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd

df = pd.read_csv('ıstanbulverı1.csv', encoding='iso-8859-9')
print(df)


# In[30]:


df


# In[34]:


new_column_names = {
    'Yıl ': 'Year',
    'Yolcu Sayısı (Kişi/Gün)': 'Passenger Count (Person/Day)',
    'Yolculuk Türü': 'Transport Type'
}

df = df.rename(columns=new_column_names)

print(df)


# In[35]:


df


# In[36]:


df['Transport Type'] = df['Transport Type'].replace('Raylı Sistemler', 'Train')
df['Transport Type'] = df['Transport Type'].replace('Deniz Ulaşımı', 'Sea')
df['Transport Type'] = df['Transport Type'].replace('Karayolu', 'Road')


# In[37]:


df


# In[38]:


import pandas as pd

# Dataframe columns and values are created
data = {
    'Year': [2004, 2009, 2014, 2015, 2016, 2017, 2018, 2019, 2004, 2009, 2014, 2015, 2016, 2017, 2018, 2019, 2004, 2009, 2014, 2015, 2016, 2017, 2018, 2019],
    'Passenger Count (Person/Day)': ['5,437,650', '8,416,000', '9,916,583', '9,918,601', '10,243,738', '10,793,061', '11,717,979', '12,401,120', '532,000', '881,152', '2,040,489', '2,299,312', '2,299,040', '2,446,028', '2,709,914', '3,401,993', '230,350', '277,809', '552,429', '663,387', '625,513', '507,140', '565,472', '808,278'],
    'Transport Type': ['Road', 'Road', 'Road', 'Road', 'Road', 'Road', 'Road', 'Road', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea', 'Sea']
}

# DataFrame is created
df = pd.DataFrame(data)

# 'Train' datas are chosen
train_rows = df[df['Transport Type'] == 'Train']

print(train_rows)


# In[39]:


train_rows


# In[40]:


import pandas as pd

# Dataframe columns and values are created
data = {
    'Year': [2004, 2009, 2014, 2015, 2016, 2017, 2018, 2019],
    'Passenger Count (Person/Day)': ['532,000', '881,152', '2,040,489', '2,299,312', '2,299,040', '2,446,028', '2,709,914', '3,401,993'],
    'Transport Type': ['Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train', 'Train']
}

# DataFrame is created
df = pd.DataFrame(data)

# '2019 Year' datas are chosen
train_rows_2019 = df[(df['Transport Type'] == 'Train') & (df['Year'] == 2019)]

print(train_rows_2019)


# In[44]:


df


# In[45]:


train_rows_2019


# In[46]:


import pandas as pd

# Dataframe columns and values are created
data = {
    'Year': [2019],
    'Passenger Count (Person/Day)': ['3,401,993'],
    'Transport Type': ['Train']
}

# DataFrame is created
df = pd.DataFrame(data)

# 'Person/Day' data is converted being 'Person/Year' 
df['Passenger Count (Person/Day)'] = df['Passenger Count (Person/Day)'].str.replace(',', '').astype(int)
df['Passenger Count (Person/Year)'] = df['Passenger Count (Person/Day)'] * 365  # veya artık yıllarda 366

# Unnecessary datas are wiped out
train_rows_2019y= df.drop(columns=['Passenger Count (Person/Day)'])

print(train_rows_2019y)


# In[47]:


train_rows_2019y


# In[48]:


import pandas as pd

# Dataframe columns and values are created
data = {
    'Year': [2019],
    'Passenger Count (Person/Day)': ['3,401,993'],
    'Transport Type': ['Train']
}

# DataFrame is created
df = pd.DataFrame(data)

# 'Person/Day' datas are converted being 'Person/Month' 
df['Passenger Count (Person/Day)'] = df['Passenger Count (Person/Day)'].str.replace(',', '').astype(int)
df['Passenger Count (Person/Month)'] = df['Passenger Count (Person/Day)'] * 365 / 12  # veya artık yıllarda 366

# Gereksiz sütunu kaldırma
train_rows_2019m = df.drop(columns=['Passenger Count (Person/Day)'])

print(train_rows_2019m)


# In[49]:


train_rows_2019m['Passenger Count (Person/Month)'] = train_rows_2019m['Passenger Count (Person/Month)'].astype(int)
#Data is converted integer version

print(train_rows_2019m)


# In[50]:


train_rows_2019m


# In[51]:


df


# In[53]:


import pandas as pd

# Dataframe columns and values are created
data = {
    'Year': [2019],
    'Transport Type': ['Train'],
    'January' : [103477287],
    'February': [103477287],
    'March' : [103477287],
    'April': [103477287],
    'May' : [103477287],
    'June': [103477287],
    'July' : [103477287],
    'August': [103477287],
    'September' : [103477287],
    'October': [103477287],
    'November' : [103477287],
    'December': [103477287],
}

datamonths=pd.DataFrame(data)

datamonths


# In[ ]:




