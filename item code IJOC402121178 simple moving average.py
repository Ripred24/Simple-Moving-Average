#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Just importing our necessary packages here 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Preview (only using invoice and SalesNK) of our data for Item Code: IJOC402121178

col_list = ['SoIdNK','InvoiceDateFK']
df = pd.read_csv(r"C:\Users\agautam\Desktop\IJOC402121178.csv",usecols=col_list)
df.head()


# In[3]:


#Change the type of our invoice data to datetime for analysis

print(df.dtypes)
df['InvoiceDateFK'] = pd.to_datetime(df['InvoiceDateFK'])
print(df.dtypes)


# In[5]:


#Setting the invoice date as the index column, summing all the sales on the same date, then plotting our data

#ONLY RUN BELOW LINE ONCE
#df = df.set_index("InvoiceDateFK")
df.groupby(df.index.strftime('%Y-%m-%d')).sum()
#df = df.sort_values(by = "date")
df.head()
df['SoIdNK'].plot(figsize = (14,8))


# In[6]:


#Plotting our data with a rolling average with a window of 10

df['SoIdNK'].plot(figsize = (14,8))
df['SoIdNK'].rolling(window = 10).mean().plot()


# In[7]:


#df['movingAvg'] = df['SoIdNK'].rolling(window = 10).mean()
#df.head(15)


# In[7]:


#Boxplot

import seaborn as sns
fig = plt.subplots(figsize=(20,5))
ax = sns.boxplot(x=df['SoIdNK'],whis = 1.5)


# In[8]:


#Histogram

fig = df.SoIdNK.hist(figsize = (20,5))


# In[19]:


from pylab import rcParams
import statsmodels.api as sm


# In[9]:


#Splitting our data in the train and test sets

train_len = 120
train = df[0:train_len]
test = df[train_len:]


# In[15]:


#creating our forecast column for the plot

y_hat = df.copy()
ma_window = 12
y_hat['forecast'] = df['SoIdNK'].rolling(ma_window).mean()
y_hat['forecast'][train_len:] = y_hat['forecast'][train_len - 1]


# In[27]:


import pylab
pylab.show


# In[39]:


#Plotting our Train, Test, and Forecast data on the same graph

plt.figure(figsize=(20,5))
plt.grid()
plt.plot(train['SoIdNK'], label = 'Train')
plt.plot(test['SoIdNK'], label = 'Test')
plt.plot(y_hat['forecast'], label = 'Forecast')
plt.legend(loc = 'best')
plt.title('Simple Moving Average')
plt.show()


# In[41]:


#Calculating RMSE and MAPE for forecast accuracy analysis. 

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['SoIdNK'], y_hat['forecast'][train_len:])).round(2)
mape = np.round(np.mean(np.abs(tes['SoIdNK']-y_hat['forecast'][train_len:])/test['SoIdNK'])*100,2)

results = pd.DataFrame({'Method':['Simple Moving Average Forecast'], 'MAPE': [mape], 'RMSE': [rmse]})
results = results[['Method', 'RMSE', 'MAPE']]
results


# In[ ]:




