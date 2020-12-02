#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
# Handle table-like data and matrices
import pandas as pd
import numpy as np


# In[2]:


# Load the training data
data = pd.read_csv("C:\\Users\\Waleed Asad\\Downloads\\Compressed\\Attached Files\\Data\\train.csv")


# In[3]:


# drop null value of data
data = data.dropna()


# In[4]:


# reset index number of data frame
data.reset_index(inplace=True)


# In[5]:


# Just cleaning the text over here. Removing stop words, removing unwanted characters like . ; : 
def cleanText(text):
    from bs4 import BeautifulSoup
    import re
    import string
    text = BeautifulSoup(text, "lxml").text
    text = text.replace('\n','')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


# In[6]:


full_cleaned = [cleanText(i) for i in data['title']]


# In[7]:


# Converting the clean text with Count Vectorizer and transformer. This is how the text is converted to TFIDF
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_features = 100)#We are extracting only 100 features for our text.
X_train_counts = count_vect.fit_transform(full_cleaned)


# In[8]:


# These are the standard steps for getting TFIDF features of a text.
# As we have used number of features as 100 in the previous step.
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape[1]


# In[9]:


# Converting the list to an array, for use in Keras.
X_train_tfidf = X_train_tfidf.toarray()
# Converting the label list to an array, for use in Keras.
Y_train = np.array(data['label'])


# In[10]:


import tensorflow as tf
# Define Attention structure
def applyAttention (wordVectorRowsInSentence):   # [*, N_s, N_f]
    N_f = wordVectorRowsInSentence.shape[-1]
    uiVectorRowsInSentence = tf.keras.layers.Dense(units=100, activation='tanh')(wordVectorRowsInSentence) # [*, N_s, N_a]
    vVectorColumnMatrix = tf.keras.layers.Dense(units=1, activation='tanh')(uiVectorRowsInSentence) # [*, N_s, 1]
    vVector = tf.keras.layers.Flatten()(vVectorColumnMatrix)    # [*, N_s]
    attentionWeightsVector = tf.keras.layers.Activation('softmax', name='attention_vector_layer')(vVector) # [*,N_s]
    attentionWeightsMatrix = tf.keras.layers.RepeatVector(N_f)(attentionWeightsVector)   # [*,N_f, N_s]
    attentionWeightRowsInSentence = tf.keras.layers.Permute([2, 1])(attentionWeightsMatrix)  # [*,N_s, N_f]
    attentionWeightedSequenceVectors = tf.keras.layers.Multiply()([wordVectorRowsInSentence, attentionWeightRowsInSentence])  # [*,N_s, N_f]
    attentionWeightedSentenceVector = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(attentionWeightedSequenceVectors)    # [*,N_f]
    return attentionWeightedSentenceVector


# In[11]:


# Converting the list to an array, for use in Keras.
X_train_tfidf = np.array(applyAttention (X_train_tfidf))


# In[12]:


# here we are imporitng important libraries for building model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Attention


# In[13]:


#Creating model
data_dim =100
model = Sequential()
model.add(LSTM(100, input_shape=(None, data_dim),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='RMSProp',metrics=['accuracy'])
model.summary()


# In[14]:


# here we are splitting the data for training and testing the model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train_tfidf[:,None],Y_train , test_size=0.05, random_state=42)


# In[15]:


# callback for earlystopping Monitors the model’s val_loss and ModelCheckpoint Monitors the model’s val_acc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                           verbose=1, patience= 5)

checkpoint = ModelCheckpoint('model_Rnn.h5', monitor='val_acc', 
                            verbose=1, save_best_only=True, mode='max')


# In[16]:


# Fitting the model
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,callbacks = [early_stop,checkpoint])


# In[57]:


# Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[117]:


data_test = pd.read_csv("C:\\Users\\Waleed Asad\\Downloads\\Compressed\\Attached Files\\Data\\test.csv")
data_y_test = pd.read_csv("C:\\Users\\Waleed Asad\\Downloads\\Compressed\\Attached Files\\Data\\submit.csv")


# In[118]:


data_test = pd.concat([data_test, data_y_test], axis=1, sort=False)


# In[119]:


data_test = data_test.dropna()


# In[120]:


full_cleaned_test = [cleanText(i) for i in data_test['title']]


# In[121]:


model.predict(x_test[1,None])


# In[122]:


from sklearn.feature_extraction.text import CountVectorizer

count_vect_test = CountVectorizer(max_features = 100)#We are extracting only 100 features for our text.
#You can change this number according to your own dataset.
X_test_counts = count_vect.fit_transform(full_cleaned_test)


# In[123]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer_test = TfidfTransformer()
X_test_tfidf = tfidf_transformer_test.fit_transform(X_test_counts)


# In[124]:


X_test_tfidf = X_test_tfidf.toarray()
Y_test = np.array(data_test['label'])


# In[125]:


X_test_tfidf = np.array(applyAttention (X_test_tfidf))


# In[126]:


model.evaluate(X_test_tfidf[:,None],Y_test,batch_size=20)


# In[127]:


model_ans = model.predict_classes(X_test_tfidf[:,None])


# In[128]:


df = np.array([model_ans, Y_test])
df = pd.DataFrame(df)


# In[129]:


df


# In[ ]:




