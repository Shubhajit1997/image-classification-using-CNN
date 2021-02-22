#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()


# In[6]:


X_test.shape


# In[7]:


y_train = y_train.reshape(-1)
y_train[:5]


# In[8]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[45]:


def plot_sample(X,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[11]:


X_train = X_train / 255
X_test = X_test / 255


# In[47]:


cnn = models.Sequential([
    
        #cnn
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
    
        #dense
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
])


# In[48]:


cnn.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


# In[49]:


cnn.fit(X_train,y_train, epochs=20)


# In[57]:


cnn.evaluate(X_test,y_test)


# In[58]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[59]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[60]:


y_test=y_test.reshape(-1,)
y_test[:5]


# In[61]:


plot_sample(X_test,y_test,5)


# In[62]:


classes[y_classes[5]]


# In[ ]:




