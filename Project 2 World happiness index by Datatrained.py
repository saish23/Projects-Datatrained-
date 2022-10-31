#!/usr/bin/env python
# coding: utf-8

# In[27]:


# Imp libs:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standardizing and Normalizing data:
from sklearn.preprocessing import StandardScaler

# Spliting data
from sklearn.model_selection import train_test_split,GridSearchCV

# Standardizing data
from sklearn.decomposition import PCA

# Model instantiating
import xgboost as xgb

# Importing metrics
from sklearn.metrics import r2_score,mean_squared_error

# Removing warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# DATA SET :

training_dataset = 'https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv'
df = pd.read_csv(training_dataset)


# In[ ]:


"""
EDA: In this project, we have used XGBoost to predict label ie Happiness score,

By data exploration we can observe that: 

1. (158, 12) are the no. of rows,columns respectively in the dataset.
2. From 12 columns Region and Country are categorical in nature , Happiness rank is integer type and rest are float type.
3. There are no null values in dataset.
4. By data description , data seems well distributed but some outliers can be observed , removing them using zscore stats.

"""


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[12]:


# Filtering numerical data and categorical data:

numericals = ['int8','int16','int32','int64','float16','float32','float64',]
categorical_columns = []
numerical_columns = []
features = df.columns.values.tolist()

for col in features:
    if df[col].dtype in numericals:
        numerical_columns.append(col)
    else:
        categorical_columns.append(col)


# In[16]:


# Checking data distribution:

plt.figure(figsize=(20,15),facecolor='Yellow')
plotnumber = 1
for column in df.columns:
    if plotnumber <= 10:
        ax = plt.subplot(3,4,plotnumber)
        sns.distplot(df[numerical_columns])
        plt.xlabel(column,fontsize=20)
    plotnumber += 1
plt.show()


# In[19]:


# Using Z Statistics to remove outliers:

from scipy.stats import zscore

z_score = zscore(df[numerical_columns])

abs_z_score = np.abs(z_score)

filtering_entry = (abs_z_score < 3).all(axis=1) # values lying in 3 times std will be removed

df_upd = df[filtering_entry]

df_upd.describe()


# In[20]:


# Checking data distribution:

plt.figure(figsize=(20,15),facecolor='Yellow')
plotnumber = 1
for column in df.columns:
    if plotnumber <= 10:
        ax = plt.subplot(3,4,plotnumber)
        sns.distplot(df_upd[numerical_columns])
        plt.xlabel(column,fontsize=20)
    plotnumber += 1
plt.show()


# In[15]:


# Encoding categorical columns:

df_dummies = pd.get_dummies(df[categorical_columns],drop_first=True)
df_dummies.head()


# In[42]:


df_upd = df_upd.join(df_dummies)


# In[43]:


df_upd.head()


# In[44]:


df_upd.drop(columns=categorical_columns,axis=1,inplace=True)


# In[45]:


df_upd.shape


# In[46]:


# Splitting data

X = df_upd.drop(columns = ['Happiness Score'],axis=1)
y = df_upd['Happiness Score']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25 , random_state=10)


# In[47]:


# Using XGBoost:

xgb_clf = xgb.XGBRegressor()
xgb_clf.fit(x_train,y_train)


# In[48]:


y_pred = xgb_clf.predict(x_test)


# In[49]:


r2_score(y_test,y_pred)


# In[50]:


params = {"learning_rate"    : [0.05, 0.10] ,
         "max_depth"        : [ 3, 5, 8, 12]}


# In[51]:


grd = GridSearchCV(xgb_clf,param_grid=params)

grd.fit(x_train,y_train)


# In[52]:


y_gpred = grd.predict(x_test)


# In[53]:


r2_score(y_test,y_gpred)*100


# In[54]:


# Pipelining: 

from sklearn.pipeline import Pipeline

pipe = Pipeline([('grd',GridSearchCV(xgb_clf,param_grid=params))])

pipe.fit(x_train,y_train)


# In[55]:


y_ppred = pipe.predict(x_test)


# In[56]:


r2_score(y_test,y_ppred)*100


# In[57]:


# Saving model to pickle string

import pickle 
saved_model = pickle.dumps(pipe) 
pipe_pickle = pickle.loads(saved_model)
pipe_pickle.predict(x_test) # predicting testing data


# In[ ]:




