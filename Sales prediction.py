#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import pickle
from word2number import w2n


# In[2]:


sales = pd.read_csv("sales.csv")


# In[3]:


sales.head(6)


# In[4]:


sales.info()


# In[5]:


sales.isnull().sum()


# In[6]:


#sales.iloc[2:,:0].apply(w2n.word_to_num)


# In[7]:


sales.head()


# In[8]:


sales['rate'].fillna((sales['rate'].mode()), inplace=True)


# In[9]:


sales['rate'] = sales['rate'].apply(w2n.word_to_num)


# In[10]:


sales.head()


# In[11]:


x = sales.iloc[:,:-1]


# In[12]:


y = sales['sales_in_third_month']


# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


regression = LinearRegression()


# In[15]:


regression.fit(x, y)


# In[16]:


# Saving model to disk
pickle.dump(regression, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




