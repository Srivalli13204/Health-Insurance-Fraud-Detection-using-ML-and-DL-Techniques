#!/usr/bin/env python
# coding: utf-8

# In[1]:


#installing & set up


# In[2]:


pip install tensorflow


# In[3]:


import tensorflow as tf


# In[4]:


print(tf.__version__)


# In[5]:


#importing libraries


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# In[7]:


#loading the dataset


# In[8]:


df = pd.read_csv("C:/Users/MADHUSUDAN/Downloads/fraud.csv")
df


# In[9]:


df.head()


# In[10]:


#data_preprocessing


# In[11]:


df.shape


# In[12]:


#null_values


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


df['Class'].value_counts()


# In[16]:


#balancing_the_dataset


# In[17]:


frauds = df[df['Class'] == 1]
no_fraud = df[df['Class'] == 0]


# In[18]:


frauds.shape, no_fraud.shape


# In[19]:


#random_selection_of_samples


# In[20]:


non_fraud = no_fraud.sample(n = 492)


# In[21]:


non_fraud.shape


# In[22]:


#merging_the_dataset


# In[23]:


dataset = pd.concat([frauds,non_fraud], ignore_index = True)


# In[24]:


dataset


# In[25]:


dataset['Class'].value_counts()


# In[26]:


#defining_feature_matrix


# In[27]:


x = dataset.drop(labels = ['Class'], axis = 1)


# In[28]:


#dependent_variable


# In[29]:


y = dataset['Class']


# In[30]:


x.shape, y.shape


# In[31]:


x.head()


# In[32]:


#splitting_dataset


# In[33]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[34]:


x_train.shape, x_test.shape


# In[35]:


#feature_scaling


# In[36]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[37]:


x_train


# In[38]:


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# In[39]:


x_train.shape, x_test.shape


# In[40]:


#reshaping


# In[41]:


x_train = x_train.reshape(787, 30, 1)
x_test = x_test.reshape(197, 30, 1)


# In[42]:


x_train.shape, x_test.shape


# In[43]:


#defining_an_object = model


# In[44]:


model = tf.keras.models.Sequential()


# In[45]:


#1st_cnn_layer


# In[46]:


model.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (30, 1)))

#batch_normalization
model.add(tf.keras.layers.BatchNormalization())

#max_pool_layer
model.add(tf.keras.layers.MaxPool1D(pool_size = 2))

#dropout_layer
model.add(tf.keras.layers.Dropout(0.2))


# In[47]:


#2nd_cnn_layer


# In[48]:


model.add(tf.keras.layers.Conv1D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))

#batch_normalization
model.add(tf.keras.layers.BatchNormalization())

#max_pool_layer
model.add(tf.keras.layers.MaxPool1D(pool_size = 2))

#dropout_layer
model.add(tf.keras.layers.Dropout(0.3))


# In[49]:


#flatten_layer


# In[50]:


model.add(tf.keras.layers.Flatten())


# In[51]:


#1st_dense_layer


# In[52]:


model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))

#dropout_layer
model.add(tf.keras.layers.Dropout(0.3))


# In[53]:


#2nd_dense_layer = output_layer


# In[54]:


model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[55]:


model.summary()


# In[56]:


#compiling_the_model


# In[57]:


opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)


# In[58]:


model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy','precision','recall'])


# In[59]:


#training_the_model


# In[60]:


history = model.fit(x_train, y_train, epochs = 25, validation_data = (x_test, y_test))


# In[61]:


#model_predictions


# In[62]:


predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)


# In[63]:


plt.plot(history.history['val_loss'])
plt.show()


# In[64]:


plt.plot(history.history['val_accuracy'])
plt.show()


# In[65]:


#calculations


# In[66]:


cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print("Confusion Matrix : ")
print(cm)
print("Accuracy Score : ",accuracy)
print("Precision Score : ",precision)
print("Recall Score : ",recall)


# In[67]:


#learning_curve


# In[68]:


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


# In[70]:


learning_curve(history, 25)


# In[ ]:




