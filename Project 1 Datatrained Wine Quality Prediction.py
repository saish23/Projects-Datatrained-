#!/usr/bin/env python
# coding: utf-8

# In[360]:


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


# Importing metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score

# Removing warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[113]:


link = 'https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv'
df = pd.read_csv(link)


# In[ ]:


"""# EDA :

1. there are total of 1599 rows and 12 columns , 
   all features are of float 64 data type
   no nulls are present.

2. By checking the data distribution, skewness is detected and outliers can also be seen.
   Skewness needs to be corrected and outliers need to be removed.
   
3. By checking the value counts of label i.e. 'Quality' column , it can be concluded that 
   data set is imbalanced, and it needs to be balanced and we will use SMOTE method.
   
4. As we are unsure about whether which most important features to be considered for predicting wine quality,
   we will use feature selection methods, we can observe that 'volatile acidity' has high negative corelation 
   with label ie 'Quality' thus we have decided to drop it.
   
5. By using various models to predict categorical data ie quality of wine, we conclude that KNN model is giving the best
   result: 
   
   Train result:  100.0 Percent
   Test result:  85.779 Percent
   
   Hyper parameter tuning is done and this result is obtained by chosing the best parameters i.e. 
   {'algorithm': 'ball_tree', 'leaf_size': 1, 'n_neighbors': 1}
   
"""


# In[114]:


df.head(10)


# In[115]:


df.describe()


# In[116]:


df.isna().sum()


# In[117]:


df.dtypes


# In[118]:


df.shape


# In[119]:


# Checking data distribution:

plt.figure(figsize=(20,15),facecolor='Yellow')
plotnumber = 1
for column in df.columns:
    if plotnumber <= 11:
        ax = plt.subplot(3,4,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize=20)
    plotnumber += 1
plt.show()


# In[120]:


df.columns


# In[121]:


# Using Z Statistics to remove outliers:

from scipy.stats import zscore

col = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                    'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

z_score = zscore(df[col])

abs_z_score = np.abs(z_score)

filtering_entry = (abs_z_score < 3).all(axis=1) # values lying in 3 times std will be removed

df = df[filtering_entry]

df.describe()


# In[122]:


df.skew().sort_values(ascending=False)


# In[123]:


# Checking skewness

df_new = pd.DataFrame(df,columns=df.columns)
df_new.skew().sort_values(ascending=False)


# In[124]:


# Checking data distribution:

plt.figure(figsize=(20,15),facecolor='Yellow')
plotnumber = 1
for column in df_new.columns:
    if plotnumber <= 11:
        ax = plt.subplot(3,4,plotnumber)
        sns.distplot(df_new[column])
        plt.xlabel(column,fontsize=20)
    plotnumber += 1
plt.show()


# In[125]:


df_corr = df_new.corr().abs()
plt.figure(figsize=(18,14))
sns.heatmap(df_corr,annot=True,annot_kws={'size':10})
plt.show()


# In[151]:


df_new.drop(columns = 'quality',axis = 1).corrwith(df_new.quality).plot(kind='bar',grid=True,figsize=(10,7),title='corelation between features and labels')
plt.show()


# In[197]:


df.columns


# In[338]:


# relevant features:

features = ['fixed acidity','citric acid','residual sugar','sulphates','alcohol','free sulfur dioxide','pH','chlorides','density','total sulfur dioxide']

df_relevant = df[features]

df_relevant.head()


# In[339]:


# Splitting data:

y = df_new.quality
X_old = df_relevant


# In[340]:


# CREATING SUBSET OF SELECTED FEATURES:

X_old = df[features]
y = df.quality


# In[341]:


# Transforming data to remove skewness

from sklearn.preprocessing import power_transform, PowerTransformer

pt = PowerTransformer()

X = pt.fit_transform(X_old)

X


# In[289]:


# Using PCA i.e. Principal Component Analysis that is a diamensionallity reduction technique:

pca = PCA()
pca.fit_transform(X)


# In[290]:


# Using Scree Plot to identify best components:

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Principal Components')
plt.ylabel('Variance Covered')
plt.title('PCA')
plt.show()


# In[337]:


pca = PCA(n_components=8)
pca.fit_transform(X)
new_pcomp = pca.fit_transform(X)
princi_comp = pd.DataFrame(new_pcomp,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'])
princi_comp


# In[296]:


X.shape


# In[266]:


y.value_counts()


# In[342]:


# Balancing dataset

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X,y)


# In[343]:


X_resampled.shape


# In[344]:


y_resampled.value_counts()


# In[345]:


# Spliting dataset

x_train,x_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size = 0.30 , random_state=42)


# In[346]:


# Defining function for model confidence and accuracy:

def metric_score(clf,x_train,x_test,y_train,y_test,train=True):
    if train == True:
        y_pred = clf.predict(x_train)
        print('Train result: ',round(accuracy_score(y_train,y_pred)*100,3),'Percent')
    elif train == False:
        pred = clf.predict(x_test)
        print('Test result: ',round(accuracy_score(y_test,pred)*100,3),'Percent')
        
        print('\n\n Test Classification report: \n\n',classification_report(y_test,pred,digits=2)) ##Model confidence/accuracy


# In[347]:


# Checking via svc model

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

metric_score(svc,x_train,x_test,y_train,y_test,train=True)

metric_score(svc,x_train,x_test,y_train,y_test,train=False)


# In[348]:


# HyperParameter Tuning in SVM and finding best parameters:

param_grid = {'C':[1,5,10,20],'gamma':[0.001,0.01,0.02,0.002]}

grd = GridSearchCV(svc,param_grid)

grd.fit(x_train,y_train)

grd.best_params_


# In[349]:


# Using best parameters for improved score:

svc = SVC(C=20,gamma=0.02)

svc.fit(x_train,y_train)

metric_score(svc,x_train,x_test,y_train,y_test,train=True)

metric_score(svc,x_train,x_test,y_train,y_test,train=False)


# In[350]:


# Checking via knn model

from sklearn.neighbors import KNeighborsClassifier

# Innitiate k neighbour classifier:

knn = KNeighborsClassifier()

# Model Training:

knn.fit(x_train,y_train)

# Calling metric_score function:

metric_score(knn,x_train, x_test, y_train, y_test,train=True)

metric_score(knn,x_train, x_test, y_train, y_test,train=False)


# In[351]:


# HyperParameter Tuning in SVM and finding best parameters:

param_gridknn = {'n_neighbors':[1,2,3,4,5],'algorithm':['ball_tree', 'kd_tree'],'leaf_size':[1,2,3,4,5]}

grd_knn = GridSearchCV(knn,param_gridknn)

grd_knn.fit(x_train,y_train)

grd_knn.best_params_


# In[352]:


# Innitiate k neighbour classifier:

knn = KNeighborsClassifier(algorithm='ball_tree',leaf_size=1,n_neighbors=1)

# Model Training:

knn.fit(x_train,y_train)

# Calling metric_score function:

metric_score(knn,x_train, x_test, y_train, y_test,train=True)

metric_score(knn,x_train, x_test, y_train, y_test,train=False)


# In[353]:


# Checking via GBC model

from sklearn.ensemble import GradientBoostingClassifier

gbdt_clf = GradientBoostingClassifier()

# Training the model
gbdt_clf.fit(x_train,y_train)

metric_score(gbdt_clf,x_train,x_test,y_train,y_test,train=True)

metric_score(gbdt_clf,x_train,x_test,y_train,y_test,train=False)


# In[354]:


# HYPER PARAMETER TUNING:

# Tuning parameters using GridSearchCV:

params = {'max_depth':[4],
          'min_samples_split':[2,4]} # at which rate our model should learn

grd = GridSearchCV(gbdt_clf,param_grid=params)

grd.fit(x_train,y_train)


# In[355]:


gbdt_clf_f = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, min_samples_split=2)

# Training the model
gbdt_clf_f.fit(x_train,y_train)


# In[356]:



metric_score(gbdt_clf_f,x_train,x_test,y_train,y_test,train=True)

metric_score(gbdt_clf_f,x_train,x_test,y_train,y_test,train=False)


# In[364]:


from sklearn.pipeline import Pipeline

pipe = Pipeline([('Transformer',PowerTransformer()),('pca',PCA(n_components=10)),('knn',KNeighborsClassifier(algorithm='ball_tree',leaf_size=1,n_neighbors=1))])

pipe.fit(x_train,y_train)


# In[366]:


y_ppred = pipe.predict(x_test)


# In[367]:


accuracy_score(y_test,y_ppred)


# In[370]:


# Saving model to pickle string

import pickle 
saved_model = pickle.dumps(pipe) 
pipe_pickle = pickle.loads(saved_model)
pipe_pickle.predict(x_test) # predicting testing data


# In[ ]:





# In[ ]:




