#!/usr/bin/env python
# coding: utf-8

# # Concrete Slump Test Regression

# The concrete slump test measures the consistency of fresh concrete before it sets. It is performed to check the workability of freshly made concrete, and therefore the ease with which concrete flows. It can also be used as an indicator of an improperly mixed batch.
# 
# <img src="https://i0.wp.com/civiconcepts.com/wp-content/uploads/2019/08/Slump-Cone-test-of-concrete.jpg?fit=977%2C488&ssl=1">
# 
# Our data set consists of various cement properties and the resulting slump test metrics in cm. Later on the set concrete is tested for its compressive strength 28 days later.
# 
# Input variables (9):
# 
# (component kg in one M^3 concrete)(7):
# * Cement
# * Slag
# * Fly ash
# * Water
# * SP
# * Coarse Aggr.
# * Fine Aggr.
# 
# (Measurements)(2)
# * SLUMP (cm)
# * FLOW (cm)
# 
# Target variable (1):
# * **28-day Compressive Strength (Mpa)**
# 
# Data Source: https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test
# 
# *Credit: Yeh, I-Cheng, "Modeling slump flow of concrete using second-order regressions and artificial neural networks," Cement and Concrete Composites, Vol.29, No. 6, 474-480, 2007.*

# # Importing dependencies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# # Loading data

# In[2]:


df = pd.read_csv("cement_slump.csv")


# # EDA and Graphical analysis

# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.columns


# In[10]:


df["Cement"].value_counts()


# In[11]:


sns.pairplot(df)


# In[12]:


df.corr()


# In[13]:


sns.heatmap(df.corr(), annot= True)


# In[14]:


pip install scikit-learn


# In[15]:


df2=df.copy()


# In[16]:


df2.head()


# 

# In[17]:


X=df2.drop(columns= "Compressive Strength (28-day)(Mpa)")


# In[19]:


X.head(1)


# In[20]:


y=df2["Compressive Strength (28-day)(Mpa)"]


# In[22]:


y.head()


# 

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test,y_train, y_test = train_test_split (X,y, test_size = 0.3, random_state=60)


# In[26]:


df2.sample(15)


# In[27]:


print("Train feautures shape : ", X_train.shape)
print("Train target shape : " , y_train.shape)
print("Test feautures shape : ", X_test.shape)
print("Test target shape : " , y_test.shape)


# In[28]:


X_train


# In[29]:


X_test


# In[30]:


y_train


# In[31]:


y_test


# In[ ]:





# In[ ]:





# In[ ]:





# # Data Preprocessing 

# ### Features and target variable

# In[ ]:


# Hizimi alamayip yukarida yapmisim, kusura bakmayin.. Vakit yetisirse düzeltecegim


# ### Splitting data into training and testing

# In[ ]:


# Hizimi alamayip yukarida yapmisim, kusura bakmayin.. Vakit yetisirse düzeltecegim


# ## Scaling

# In[ ]:


# Bu konu bende eksik, bayramda bakmaya firsat olmadi, ilkbakista düzeltecegim ins..


# ##  1. Model Building (Linear Regression)

# In[32]:


from sklearn.linear_model import LinearRegression


# ### 1.1 Interpret the model

# In[33]:


model= LinearRegression()


# ### 1.2 Model Evaluation

# In[34]:


model.fit(X_train, y_train)


# In[42]:


y_pred = model.predict(X_test)
y_pred    # This is actually y_test


# In[40]:


X_test.head()


# In[48]:


y_pred2= model.predict(X_test.head(1))
y_pred2


# In[36]:


model.coef_


# In[37]:


model.intercept_


# In[ ]:


# y_pred = a**9 * feauture1 + b**8 * feature2 + .... + c


# In[44]:


sum(X_test.loc[52] * model.coef_) + model.intercept_


# In[45]:


my_dict = {"Actual" : y_test, "Pred" : y_pred, "Residual" : y_test-y_pred}


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 2. Regularization

# ## 2.1 Ridge (Apply and evaluate)

# In[ ]:





# ## 2.2 Lasso (Apply and evalute)

# In[ ]:





# ## 2.3 Elastic-Net (Apply and evaluate )
# * Use Gridsearch for hyperparameter tuning instead of ElacticnetCV

# In[ ]:




