#!/usr/bin/env python
# coding: utf-8

# In[1]:


#installing & set up


# In[2]:


pip install tensorflow


# In[1]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[2]:


print(tf.__version__)


# In[3]:


#importing libraries


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#loading the dataset


# In[6]:


df = pd.read_csv("C:/Users/MADHUSUDAN/Downloads/fraud.csv")
df


# In[7]:


df.head()


# In[8]:


#data_preprocessing


# In[9]:


df.shape


# In[10]:


#null_values


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[14]:


df['Class'].value_counts()


# In[15]:


#balancing_the_dataset


# In[16]:


frauds = df[df['Class'] == 1]
no_fraud = df[df['Class'] == 0]


# In[17]:


frauds.shape, no_fraud.shape


# In[18]:


#random_selection_of_samples


# In[19]:


fraud = frauds.sample(n = 492)


# In[20]:


fraud.shape


# In[21]:


non_fraud = no_fraud.sample(n = 492)


# In[22]:


non_fraud.shape


# In[23]:


#merging_the_dataset


# In[24]:


dataset = pd.concat([fraud,non_fraud], ignore_index = True)


# In[25]:


dataset


# In[27]:


dataset['Class'].value_counts()


# In[28]:


#defining_feature_matrix


# In[29]:


x = dataset.drop(labels = ['Class'], axis = 1)


# In[30]:


#dependent_variable


# In[31]:


y = dataset['Class']


# In[32]:


x.shape, y.shape


# In[34]:


x.head()


# In[35]:


#splitting_dataset


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[37]:


x_train.shape, x_test.shape


# In[38]:


#rnn_model


# In[39]:


model = Sequential()


# In[40]:


from keras.layers import LSTM, InputLayer
model.add(InputLayer(batch_input_shape=(None, 5, 1)))


# In[41]:


model.add(LSTM(1, return_sequences = True))
model.add(LSTM(1, return_sequences = True))
model.add(LSTM(1, return_sequences = True))
model.add(LSTM(1, return_sequences = False))


# In[42]:


model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['accuracy','precision','recall'])


# In[43]:


model.summary()


# In[44]:


#fitting_the_model


# In[45]:


history = model.fit(x_train, y_train, epochs = 25, validation_data = (x_test, y_test))


# In[46]:


results = model.predict(x_test)


# In[47]:


plt.scatter(range(len(results)), results, c = 'r', label = 'Results')
plt.scatter(range(len(y_test)), y_test, c = 'g', label = 'Test Data')
plt.legend()
plt.show()


# In[48]:


#model_predictions


# In[49]:


predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)


# In[50]:


plt.plot(history.history['loss'])
plt.show()


# In[51]:


plt.plot(history.history['accuracy'])
plt.show()


# In[52]:


#calculations


# In[53]:


cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Confusion Matrix : ")
print(cm)
print("Accuracy Score : ",accuracy)
print("Precision Score : ",precision)
print("Recall Score : ",recall)


# In[54]:


#learning_curve


# In[55]:


def learning_curve(history, epoch):    
    epoch_range = range(1, epoch+1)    
# Plotting Model Accuracy
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()    
# Plotting Model Loss
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
# Plotting Precision
    plt.plot(epoch_range, history.history['precision'])
    plt.plot(epoch_range, history.history['val_precision'])
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
# Plotting Recall
    plt.plot(epoch_range, history.history['recall'])
    plt.plot(epoch_range, history.history['val_recall'])
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# In[57]:


learning_curve(history, 25)


# In[ ]:




