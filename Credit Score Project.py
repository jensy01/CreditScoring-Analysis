#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# In[10]:


dataset = pd.read_excel("path/a_Dataset_CreditScoring.xlsx")


# In[11]:


# shows count of rows and columns
dataset.shape


# In[12]:


#shows first few rows of the code
dataset.head()


# In[13]:


#dropping customer ID column from the dataset
dataset=dataset.drop('ID',axis=1)
dataset.shape


# In[14]:


# explore missing values
dataset.isna().sum()


# In[15]:


# filling missing values with mean
dataset=dataset.fillna(dataset.mean())


# In[16]:


# explore missing values post missing value fix
dataset.isna().sum()


# In[18]:


y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values


# In[19]:


# splitting dataset into training and test (in ratio 80:20)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)


# In[20]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[21]:


classifier =  LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[22]:


print(confusion_matrix(y_test,y_pred))


# In[23]:


print(accuracy_score(y_test, y_pred))


# In[24]:


predictions = classifier.predict_proba(X_test)
predictions


# In[27]:


# writing model output file

df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

dfx.to_csv("path/c1_Model_Prediction.csv", sep=',', encoding='UTF-8')

dfx.head()


# In[ ]:




