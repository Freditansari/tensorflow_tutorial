#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing;
from sklearn.model_selection import cross_validate as cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %matplotlib inline



# In[ ]:


import datetime
from datetime import timedelta
forecast_out = 3650 #Days from now

stock_ticker ='AMZN'

end = datetime.datetime.now()#-timedelta(7) #remove timedetla -1 to go to productions mode
start = datetime.datetime.now()-timedelta(3650)

# benchmark = web.DataReader(stock_ticker, 'yahoo', start, end )
stock = web.DataReader(stock_ticker, 'yahoo', start, end)


# In[ ]:


test_size=20
test_index = len(stock['Adj Close'])- test_size


# In[ ]:


train = stock['Adj Close'].iloc[:test_index]
test= stock['Adj Close'].iloc[test_index:]


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler=MinMaxScaler()


# In[ ]:


scaler.fit(train.to_frame())


# In[ ]:


scaled_train = scaler.transform(train.to_frame())


# In[ ]:


scaled_test = scaler.transform(test.to_frame())


# In[ ]:





# In[ ]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


len(test)


# In[ ]:


length=19
generator = TimeseriesGenerator(scaled_train,scaled_train, length=length, batch_size=1)


# In[ ]:


from tensorflow.keras.models import Sequential


# In[ ]:


from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[ ]:


n_features =1


# In[ ]:


model= Sequential()
model.add(LSTM(500, activation='relu', input_shape=(length, n_features)))
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
# model.add(CuDNNLSTM(100, activation='relu', input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[17]:


model.summary()


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping


# In[19]:


early_stop = EarlyStopping(monitor='val_loss', patience=1)


# In[20]:


validation_generator= TimeseriesGenerator(scaled_test, scaled_test,
                                          length=length, batch_size =1)


# In[21]:


model.fit_generator(generator, epochs=20,
                   validation_data=validation_generator, callbacks=[early_stop])


# In[22]:


import tensorflow as tf
tf.keras.backend.clear_session()


# In[23]:


losses= pd.DataFrame(model.history.history)
ax =losses.plot()
# plt.show()
plt.savefig('losses_chart.png')


# In[ ]:


test_predictions =[]
first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))
for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)


# In[ ]:


true_predictions =scaler.inverse_transform(test_predictions)


# In[ ]:


predictions = pd.DataFrame(test)
# predictions['Test set'] = test.to_frame()


# In[ ]:


# test['Predictions'] = true_predictions
predictions['Predictions'] = true_predictions


# In[ ]:


ax=predictions.plot(figsize=(12,8))
# plt.show()
plt.savefig('predictions_chart.png')


# In[ ]:




