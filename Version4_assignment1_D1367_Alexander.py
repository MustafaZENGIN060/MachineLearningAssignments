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


# In[89]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # Robustscaler is used when outlier could be present

scaler = StandardScaler()


# In[90]:


scaler.fit(X_train)  # Apply just for X_tarin not for X_test (Kalıp çıkarma)


# In[91]:


X_train_scaled = scaler.transform(X_train) # Apply transform according to fit
X_train_scaled


# In[92]:


X_test_scaled = scaler.transform(X_test)
X_test_scaled


# In[ ]:


#checking std = 1 and mean = 0
#this gives us the z-scores. so it's also called z-score scaling

#These values show where in the normal distribution they correspond to the z score.


# In[93]:


pd.DataFrame(X_train_scaled).agg(["mean", "std"]).round() #Applying aggregation across all the columns, mean and std will be found for each column in the dataframe


# In[96]:


pd.DataFrame(X_test_scaled).agg(["mean", "std"]).round()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





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


# In[51]:


comparing= pd.DataFrame(my_dict)
comparing


# In[52]:


result_sample = comparing.sample(25)
result_sample


# In[53]:


result_sample.plot(kind='bar', figsize = (15,9))
plt.show()


# In[54]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[55]:


R2_score= r2_score(y_test, y_pred)
R2_score


# In[56]:


mae= mean_absolute_error(y_test, y_pred)
mae


# In[57]:


mse = mean_squared_error(y_test,y_pred)
mse


# In[58]:


rmse = np.sqrt (mse)
rmse


# In[59]:


sonuc_mean = df2["Compressive Strength (28-day)(Mpa)"].mean()
sonuc_mean


# In[60]:


mae / sonuc_mean


# In[61]:


mse / sonuc_mean


# In[62]:


rmse / sonuc_mean


# In[ ]:


# rmse ve mae yakin, outlier'lar cok asiri degil


# In[63]:


def adj_r2(y_test, y_pred, df2):
    r2 = r2_score(y_test, y_pred)
    n = df2.shape[0]   # number of observations
    p = df2.shape[1]-1 # number of independent variables 
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2


# In[64]:


adj_r2(y_test, y_pred, df)


# In[ ]:


# adj_r2= R2'in düzenlenmis hali'.  R2'ye yakin, demek ki feauture"lar anlamli.


# In[65]:


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model testing performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")


# In[68]:


#R2_score = r2_score(actual, pred)


# In[67]:


eval_metric(y_test, y_pred)


# In[69]:


y_train_pred = model.predict (X_train)


# In[70]:


eval_metric (y_train, y_train_pred)


# In[71]:


residuals = y_test-y_pred


# In[72]:


plt.figure(figsize = (10,6))
sns.scatterplot(x = y_test, y = residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()


# In[73]:


sns.kdeplot(residuals)


# In[ ]:


# Normal Distrubution a yakin.


# In[81]:


pip install yellowbrick


# In[80]:


from yellowbrick.regressor import ResidualsPlot

# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();        # Finalize and render the figure


# In[ ]:


# calismadi (?) - belki programi resetleyince calisir


# In[ ]:





# In[79]:


from yellowbrick.regressor import PredictionError
# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show(); 


# In[ ]:


# prediction error for lineer regression - Yellowbrick yukarida yine calismadi.- belki programi resetleyince calisir


# In[82]:


final_model = LinearRegression()


# In[83]:


final_model.fit(X,y)


# In[84]:


final_model.coef_


# In[85]:


final_model.intercept_


# In[86]:


df.head()


# In[87]:


coeff_df = pd.DataFrame(final_model.coef_, index = X.columns, columns = ["Coefficient"] )


# In[88]:


coeff_df


# In[ ]:


# Water and Slump have the best effect on target


# In[ ]:


# Cross_Validation


# In[ ]:


# We do cross-validation to check whether the one-time scores we receive are consistent or not

# cross validation is only applied to the train set.


# In[97]:


from sklearn.metrics import SCORERS


# In[98]:


list(SCORERS.keys())


# In[99]:


from sklearn.model_selection import cross_validate


# In[100]:


model2 = LinearRegression()
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['r2', 'neg_mean_absolute_error','neg_mean_squared_error',                                                             'neg_root_mean_squared_error'], cv = 5)


# In[101]:


scores


# In[102]:


pd.DataFrame(scores, index = range(1,6))


# In[103]:


scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()


# In[104]:


sns.lineplot(data = scores.iloc[:,2:]);


# In[ ]:





# In[ ]:





# In[ ]:





# # 2. Regularization

# ## 2.1 Ridge (Apply and evaluate)

# In[ ]:


#Ridge and lasso and elastic-net regression are a model tuning method that is used to analyse any data that suffers from multicollinearity, underfiting and overfiting.


# In[105]:


from sklearn.linear_model import Ridge


# In[107]:


ridge_model = Ridge(alpha=1, random_state=60)   # I"m from Tokat :)


# In[108]:


ridge_model.fit(X_train_scaled, y_train)


# In[109]:


y_pred = ridge_model.predict(X_test_scaled)
y_train_pred = ridge_model.predict(X_train_scaled)


# In[110]:


rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
rs


# In[111]:


pd.concat([ls, rs], axis=1)


# In[112]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()


# In[113]:


lm.fit(X_train_scaled, y_train)


# In[114]:


y_pred = lm.predict(X_test_scaled)
y_train_pred = lm.predict(X_train_scaled)


# In[115]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_val(y_train, y_train_pred, y_test, y_pred, name):
    
    scores = {name+"_train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    name+"_test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# In[116]:


ls =train_val(y_train, y_train_pred, y_test, y_pred, "linear") # Evaluate the result.
ls


# In[117]:


train_val(y_train, y_train_pred, y_test, y_pred, "linear")


# In[118]:


sns.lineplot(data = scores.iloc[:,2:]);


# In[119]:


rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
rs


# In[120]:


pd.concat([ls, rs], axis=1)


# In[ ]:


# For Ridge Regression CV with alpha : 1


# In[121]:


model = Ridge(alpha=1, random_state=60)
scores = cross_validate(model, X_train_scaled, y_train,
                    scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)


# In[122]:


pd.DataFrame(scores, index = range(1, 6))


# In[123]:


scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()


# In[124]:


sns.lineplot(data = scores.iloc[:,2:])


# In[125]:


ridge_model.coef_


# In[127]:


rm_df2 = pd.DataFrame(ridge_model.coef_, columns = ["ridge_coef_1"])


# In[129]:


lm_df2 = pd.DataFrame(lm.coef_, columns = ["lm_coef"])
lm_df2


# In[130]:


pd.concat([lm_df2,rm_df2], axis = 1)


# In[ ]:


#Choosing best alpha value with Cross-Validation


# In[131]:


from sklearn.linear_model import RidgeCV


# In[132]:


alpha_space = np.linspace(0.01, 1, 100)
alpha_space


# In[133]:


ridge_cv_model = RidgeCV(alphas=alpha_space, cv = 5, scoring= "neg_root_mean_squared_error")


# In[134]:


ridge_cv_model.fit(X_train_scaled, y_train)


# In[135]:


ridge_cv_model.alpha_ #Ridge(alpha=1)


# In[136]:


#rmse for ridge with CV
ridge_cv_model.best_score_


# In[137]:


y_pred = ridge_cv_model.predict(X_test_scaled)
y_train_pred = ridge_cv_model.predict(X_train_scaled)


# In[138]:


rcs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge_cv")
rcs


# In[139]:


pd.concat([ls, rs, rcs], axis = 1)


# In[140]:


ridge_cv_model.coef_


# In[141]:


rcm_df2 = pd.DataFrame(ridge_cv_model.coef_, columns=["ridge_cv_coef_0.02"])


# In[143]:


pd.concat([lm_df2,rm_df2, rcm_df2], axis = 1)


# In[ ]:





# In[ ]:





# ## 2.2 Lasso (Apply and evalute)

# In[144]:


from sklearn.linear_model import Lasso, LassoCV


# In[145]:


lasso_model = Lasso(alpha=1, random_state=60)
lasso_model.fit(X_train_scaled, y_train)


# In[146]:


y_pred = lasso_model.predict(X_test_scaled)
y_train_pred = lasso_model.predict(X_train_scaled)


# In[147]:


lss = train_val(y_train, y_train_pred, y_test, y_pred, "lasso")
lss


# In[148]:


pd.concat([ls, rs, rcs, lss], axis = 1)


# In[ ]:


# For Lasso CV with Default Alpha : 1


# In[149]:


model = Lasso(alpha=1, random_state=42)
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)


# In[150]:


pd.DataFrame(scores, index = range(1, 6))


# In[151]:


scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()


# In[152]:


train_val(y_train, y_train_pred, y_test, y_pred, "lasso")


# In[153]:


sns.lineplot(data = scores.iloc[:,2:])


# In[154]:


lasso_model.coef_


# In[155]:


lsm_df2 = pd.DataFrame(lasso_model.coef_, columns = ["lasso_coef_1"])


# In[156]:


pd.concat([lm_df2, rm_df2, rcm_df2, lsm_df2], axis = 1)


# In[ ]:


# Choosing best alpha value with Cross-Validation


# In[158]:


lasso_cv_model = LassoCV(alphas = alpha_space, cv = 5, max_iter=100000, random_state=60) 


# In[159]:


lasso_cv_model.fit(X_train_scaled, y_train)


# In[160]:


lasso_cv_model.alpha_


# In[161]:


np.where(alpha_space[::-1]==lasso_cv_model.alpha_)


# In[162]:


alpha_space[::-1]


# In[163]:


#mse score for CV
lasso_cv_model.mse_path_[99].mean()


# In[164]:


y_pred = lasso_cv_model.predict(X_test_scaled)   #Lasso(alpha =0.0699)
y_train_pred = lasso_cv_model.predict(X_train_scaled)


# In[165]:


lcs = train_val(y_train, y_train_pred, y_test, y_pred, "lasso_cv")
lcs


# In[166]:


pd.concat([ls,rs, rcs, lss, lcs], axis = 1)


# In[167]:


lasso_cv_model.coef_


# In[168]:


lcm_df2 = pd.DataFrame(lasso_cv_model.coef_, columns = ["lasso_cv_coef_0.01"])


# In[169]:


pd.concat([lm_df2, rm_df2, rcm_df2, lsm_df2, lcm_df2], axis = 1)


# In[ ]:





# ## 2.3 Elastic-Net (Apply and evaluate )
# * Use Gridsearch for hyperparameter tuning instead of ElacticnetCV

# In[170]:


from sklearn.linear_model import ElasticNet, ElasticNetCV


# In[171]:


elastic_model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=60) # l1_ratio: 1: Lasso or 0:Ridge
elastic_model.fit(X_train_scaled, y_train)


# In[172]:


y_pred = elastic_model.predict(X_test_scaled)
y_train_pred = elastic_model.predict(X_train_scaled)


# In[173]:


es = train_val(y_train, y_train_pred, y_test, y_pred, "elastic")
es


# In[174]:


pd.concat([ls,rs, rcs, lss, lcs, es], axis = 1)


# In[ ]:


# For Elastic_net CV with Default alpha = 1 and l1_ratio=0.5


# In[175]:


model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)


# In[176]:


scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:]


# In[177]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# In[178]:


elastic_model.coef_


# In[179]:


em_df2 = pd.DataFrame(elastic_model.coef_, columns=["elastic_coef_(alp:1, l1:0.5)"])


# In[180]:


pd.concat([lm_df2, rm_df2, rcm_df2, lsm_df2, lcm_df2, em_df2], axis = 1)


# In[ ]:


#Choosing best alpha and l1_ratio values with Cross-Validation


# In[181]:


elastic_cv_model = ElasticNetCV(alphas = alpha_space, l1_ratio=[0.1, 0.5, 0.7,0.9, 0.95, 1], cv = 5, 
                                max_iter = 100000,random_state=60)


# In[182]:


elastic_cv_model.fit(X_train_scaled, y_train)


# In[183]:


elastic_cv_model.alpha_   #Best alpha 


# In[184]:


elastic_cv_model.l1_ratio_


# In[185]:


#mse score for CV
elastic_cv_model.mse_path_[5][-1].mean()


# In[186]:


y_pred = elastic_cv_model.predict(X_test_scaled)
y_train_pred = elastic_cv_model.predict(X_train_scaled)


# In[187]:


ecs = train_val(y_train, y_train_pred, y_test, y_pred, "elastic_cv")
ecs


# In[188]:


pd.concat([ls,rs, rcs, lss, lcs, es, ecs], axis = 1)


# In[189]:


elastic_cv_model.coef_


# In[190]:


ecm_df2 = pd.DataFrame(elastic_cv_model.coef_, columns=["elastic_coef_(alp:0.06999999999999999, l1:0.95)"])


# In[191]:


pd.concat([lm_df2, rm_df2, rcm_df2, lsm_df2, lcm_df2, em_df2, ecm_df2], axis = 1)


# In[ ]:


# Let's make all for Grid search..


# In[193]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


elastic_model = ElasticNet(max_iter=10000, random_state=60)


# In[195]:


param_grid = {"alpha":[0.01, 0.012, 0.2, 0.5, 0.6, 0.7, 1],
            "l1_ratio":[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]}


# In[196]:


grid_model = GridSearchCV(estimator = elastic_model, param_grid = param_grid, scoring = 'neg_root_mean_squared_error',
                         cv =5, verbose =2)


# In[197]:


grid_model.fit(X_train_scaled, y_train)


# In[ ]:


#pc hizlidir, sirf bu kurs icin aldim, Allah (cc) zeval vermesin..


# In[198]:


grid_model.best_params_


# In[199]:


pd.DataFrame(grid_model.cv_results_)


# In[200]:


pd.DataFrame(grid_model.cv_results_)


# In[201]:


grid_model.best_index_


# In[202]:


grid_model.best_score_


# In[ ]:


# Using Best Hyper Parameters From GridSearch


# In[203]:


y_pred = grid_model.predict(X_test_scaled)
y_train_pred = grid_model.predict(X_train_scaled)


# In[204]:


train_val(y_train, y_train_pred, y_test, y_pred, "GridSearch")


# In[ ]:


# Final Model


# In[205]:


final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(poly_features)


# In[ ]:


# I need a Polynomial Conversion


# In[206]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


# We create an artificial overfiting situation by taking poly feature from 5 degrees


# In[207]:


polynomial_converter = PolynomialFeatures(degree=5, include_bias=False)

poly_features = polynomial_converter.fit_transform(X)


# In[208]:


poly_features.shape


# In[209]:


final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(poly_features)


# In[210]:


final_model = Lasso(alpha=0.07) #lasso_cv_model


# In[211]:


final_model.fit(X_scaled, y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




