#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import seaborn as sns 
import os 
from datetime import datetime 

import warnings 
warnings.filterwarnings("ignore") 


# In[2]:


data = pd.read_csv(r'C:\Users\monik\Downloads\trial\all_stocks_5yr - Copy (2).csv') 
print(data.shape) 
print(data.sample(7)) 


# In[3]:


data.Name.unique()


# In[4]:


data.info() 


# In[5]:


data['date'] = pd.to_datetime(data['date']) 
data.info() 


# In[6]:


data['date'] = pd.to_datetime(data['date']) 
# date vs open 
# date vs close 
companies = ['AAL', 'AAPL', 'AAP', 'ABBV', 'ABC', 'ABT', 'ACN']

plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
    plt.subplot(3, 3, index) 
    c = data[data['Name'] == company] 
    plt.plot(c['date'], c['close'], c="r", label="close", marker="+") 
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^") 
    plt.title(company) 
    plt.legend() 
    plt.tight_layout() 


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
companies = ['AAL', 'AAPL', 'AAP', 'ABBV', 'ABC', 'ABT', 'ACN']

df = pd.read_csv(r'C:\Users\monik\Downloads\trial\all_stocks_5yr.csv')
fig = plt.figure()

plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
    plt.subplot(3, 3, index) 
    c = df[df['Name'] == company] 
    plt.plot(c['date'], c['volume'], c='purple', marker='*') 
    plt.title(f"{company} Volume") 
    plt.tight_layout() 


# In[8]:


apple = data[data['Name'] == 'AAPL'] 
prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1)) 
& (apple['date']<datetime(2018,1,1))] 
plt.plot(apple['date'],apple['close']) 
plt.xlabel("Date") 
plt.ylabel("Close") 
plt.title("Apple Stock Prices") 
plt.show()


# In[9]:


close_data = apple.filter(['close']) 
dataset = close_data.values 
training = int(np.ceil(len(dataset) * .95)) 
print(training) 


# In[10]:


from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset) 

train_data = scaled_data[0:int(training), :] 
# prepare feature and labels 
x_train = [] 
y_train = [] 

for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0]) 

x_train, y_train = np.array(x_train), np.array(y_train) 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #reshaping


# In[11]:


model = keras.models.Sequential() 
model.add(keras.layers.LSTM(units=64, 
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1))) 
model.add(keras.layers.LSTM(units=64)) 
model.add(keras.layers.Dense(32)) 
model.add(keras.layers.Dropout(0.5)) 
model.add(keras.layers.Dense(1)) 
model.summary 


# In[12]:


model.compile(optimizer='adam', 
            loss='mean_squared_error') 
history = model.fit(x_train, 
                    y_train, 
                    epochs=10) 


# In[13]:


test_data = scaled_data[training - 60:, :] 
x_test = [] 
y_test = dataset[training:, :] 
for i in range(60, len(test_data)): 
    x_test.append(test_data[i-60:i, 0]) 

x_test = np.array(x_test) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 

# predict the testing data 
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions) 

# evaluation metrics 
mse = np.mean(((predictions - y_test) ** 2)) 
print("MSE", mse) 
print("RMSE", np.sqrt(mse)) 


# In[14]:


import pandas as pd
#data = pd.read_csv(r'C:\Users\monik\Downloads\trial\all_stocks_5yr - Copy (2).csv') 
apple = data[data['Name'] == 'AAPL'] 
train = apple[:training] 
test = apple[training:] 
test['Predictions'] = predictions 

plt.figure(figsize=(10, 8)) 
plt.plot(train['date'], train['close']) 
plt.plot(test['date'], test[['close', 'Predictions']]) 
plt.title('Apple Stock Close Price') 
plt.xlabel('Date') 
plt.ylabel("Close") 
plt.legend(['Train', 'Test', 'Predictions']) 


# In[ ]:
