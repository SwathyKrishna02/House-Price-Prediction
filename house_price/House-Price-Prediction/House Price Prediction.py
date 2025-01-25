#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[2]:


data=pd.read_csv("House_Rent_Dataset.csv")
data


# In[3]:


data.shape


# In[4]:


data.dtypes


# In[5]:


data.drop(["Posted On","Floor","Point of Contact"],axis=1,inplace=True)


# In[6]:


data["Area Type"].unique()


# In[7]:


data["Area Type"].value_counts()


# # Data Cleaning

# In[8]:


data.isnull().sum()


# In[9]:


data


# In[10]:


le_area_type=LabelEncoder()
data["Area Type"]=le_area_type.fit_transform(data["Area Type"])

le_area_locality=LabelEncoder()
data["Area Locality"]=le_area_locality.fit_transform(data["Area Locality"])

le_city=LabelEncoder()
data["City"]=le_city.fit_transform(data["City"])

le_furnishing_status=LabelEncoder()
data["Furnishing Status"]=le_furnishing_status.fit_transform(data["Furnishing Status"])

le_tenant_preferred=LabelEncoder()
data["Tenant Preferred"]=le_tenant_preferred.fit_transform(data["Tenant Preferred"])

data


# In[11]:


plt.boxplot(data["BHK"],patch_artist=True)


# In[12]:


plt.boxplot(data["Rent"],patch_artist=True)


# In[13]:


print(f'Old shape of data: {data.shape}')

Q1=data.Rent.quantile(0.25)
Q3=data.Rent.quantile(0.75)
IQR=Q3-Q1
print(Q1,Q3,IQR)

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

upper_index=np.where(data["Rent"]>=upper)[0]
lower_index=np.where(data["Rent"]<=lower)[0]

data.drop(index=upper_index,inplace=True)
data.drop(index=lower_index,inplace=True)

print(f'New Shape of data: {data.shape}')


# In[14]:


plt.boxplot(data["Rent"],patch_artist=True)


# In[15]:


plt.boxplot(data["Size"],patch_artist=True)


# In[16]:


data.reset_index(inplace=True)
data.drop("index",axis=1,inplace=True)


# In[17]:


print(f'Old Shape of data: {data.shape}')

Q1=data.Size.quantile(0.25)
Q3=data.Size.quantile(0.75)
IQR=Q3-Q1
print(Q1,Q3,IQR)

lower=Q1-1.5*IQR
upper=Q3+1.5*IQR

upper_index=np.where(data["Size"]>=upper)[0]
lower_index=np.where(data["Size"]<=lower)[0]

data.drop(index=upper_index,inplace=True)
data.drop(index=lower_index,inplace=True)

print(f'New Shape of data: {data.shape}')


# In[18]:


plt.boxplot(data["Size"],patch_artist=True)


# In[19]:


plt.boxplot(data["Area Type"],patch_artist=True)


# In[20]:


plt.boxplot(data["Area Locality"],patch_artist=True)


# In[21]:


plt.boxplot(data["City"],patch_artist=True)


# In[22]:


plt.boxplot(data["Furnishing Status"],patch_artist=True)


# In[23]:


plt.boxplot(data["Tenant Preferred"],patch_artist=True)


# In[24]:


plt.boxplot(data["Bathroom"],patch_artist=True)


# In[25]:


# Scatter Plot: BHK vs Rent

plt.figure(figsize=(8,6))
sns.scatterplot(data=data,x="BHK",y="Rent",hue="City",palette="viridis")
plt.title("BHK vs Rent Across Cities", fontsize=14)
plt.xlabel("Number of BHK")
plt.ylabel("Rent (in ₹)")
plt.grid(True)
plt.show()


# In[26]:


# Box Plot: Rent by BHK

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="BHK", y="Rent", palette="viridis")
plt.title("Rent Distribution by BHK", fontsize=14)
plt.xlabel("Number of BHK")
plt.ylabel("Rent (in ₹)")
plt.show()


# In[27]:


# Bar Plot: Average Rent by BHK

avg_rent_by_bhk = data.groupby("BHK")["Rent"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_rent_by_bhk, x="BHK", y="Rent", palette="viridis")
plt.title("Average Rent by BHK", fontsize=14)
plt.xlabel("Number of BHK")
plt.ylabel("Average Rent (in ₹)")
plt.show()


# In[28]:


# Scatter Plot: Size vs Rent

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x="Size", y="Rent", hue="City", size="BHK", sizes=(20, 200),palette="viridis")
plt.title("Size vs Rent Across Cities", fontsize=14)
plt.xlabel("Size (in sqft)")
plt.ylabel("Rent (in ₹)")
plt.grid(True)
plt.show()


# In[29]:


# Box Plot: Rent by City

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="City", y="Rent", palette="viridis")
plt.title("Rent Distribution by City", fontsize=14)
plt.xlabel("City")
plt.ylabel("Rent (in ₹)")
plt.show()


# In[30]:


# Bar Plot: Average Rent by City

avg_rent_by_city = data.groupby("City")["Rent"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_rent_by_city, x="City", y="Rent", palette="viridis")
plt.title("Average Rent by City", fontsize=14)
plt.xlabel("City")
plt.ylabel("Average Rent (in ₹)")
plt.show()


# In[31]:


# Bar Plot: Average Rent by Bathrooms

avg_rent_by_bathroom = data.groupby("Bathroom")["Rent"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_rent_by_bathroom, x="Bathroom", y="Rent", palette="viridis")
plt.title("Average Rent by Number of Bathrooms", fontsize=14)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Average Rent (in ₹)")
plt.show()


# In[32]:


# Bar Plot: Average Rent by Furnishing Status

avg_rent_by_furnishing = data.groupby("Furnishing Status")["Rent"].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_rent_by_furnishing, x="Furnishing Status", y="Rent", palette="viridis")
plt.title("Average Rent by Furnishing Status", fontsize=14)
plt.xlabel("Furnishing Status")
plt.ylabel("Average Rent (in ₹)")
plt.show()


# In[33]:


correlation=data.corr()
correlation


# In[34]:


sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# In[35]:


min_correlation = correlation.min().min()
print(f"The lowest correlation value is: {min_correlation}")


# In[36]:


min_corr_pair = correlation.stack().idxmin()
print(f"The pair of features with the lowest correlation is: {min_corr_pair}")


# In[37]:


x=data.drop("Rent",axis=1)
y=data["Rent"]


# In[38]:


x


# In[39]:


y


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=25)


# In[41]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[42]:


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[43]:


lr = LinearRegression()
lr.fit(x_train, y_train)
lr_test_score = lr.score(x_test, y_test)
lr_train_score = lr.score(x_train, y_train)

rf = RandomForestRegressor(n_estimators=100,max_depth=10)
rf.fit(x_train, y_train)
rf_test_score = rf.score(x_test, y_test)
rf_train_score = rf.score(x_train, y_train)

dt= DecisionTreeRegressor(max_depth=10)
dt.fit(x_train, y_train)
dt_test_score = dt.score(x_test, y_test)
dt_train_score = dt.score(x_train, y_train)

knn= KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)
knn_test_score = knn.score(x_test, y_test)
knn_train_score = knn.score(x_train, y_train)

svm=SVR(C=1.0)
svm.fit(x_train, y_train)
svm_test_score = svm.score(x_test, y_test)
svm_train_score = svm.score(x_train, y_train)

ab = AdaBoostRegressor(n_estimators=50,learning_rate=1.0)
ab.fit(x_train, y_train)
ab_test_score = ab.score(x_test, y_test)
ab_train_score = ab.score(x_train, y_train)

gb = GradientBoostingRegressor(n_estimators=100,max_depth=3)
gb.fit(x_train, y_train)
gb_test_score = gb.score(x_test, y_test)
gb_train_score = gb.score(x_train, y_train)

xb = XGBRegressor(n_estimators=100,max_depth=6)
xb.fit(x_train, y_train)
xb_test_score = xb.score(x_test, y_test)
xb_train_score = xb.score(x_train, y_train)


# In[44]:


pd.DataFrame({"Model":["Linear Regression","Random Forest","Decision Tree","KNN","SVM","AdaBoost","GradientBoosting","XGB"],
             "Training Score":[lr_train_score,rf_train_score,dt_train_score,knn_train_score,svm_train_score,ab_train_score,gb_train_score,xb_train_score],
             "Testing Score":[lr_test_score,rf_test_score,dt_test_score,knn_test_score,svm_test_score,ab_test_score,gb_test_score,xb_test_score]})


# In[45]:


y_pred_xb=xb.predict(x_test)
y_pred_xb


# In[46]:


y_pred_rf=rf.predict(x_test)
y_pred_rf


# In[47]:


mae_xb=mean_absolute_error(y_test,y_pred_xb)
mse_xb=mean_squared_error(y_test,y_pred_xb)
print(f'XGBoost - MAE: {mae_xb}, MSE: {mse_xb}')

r2_xb=r2_score(y_test,y_pred_xb)
print(f"XGBoost R² Score: {r2_xb}")

xb_cv_scores = cross_val_score(xb, x_train, y_train, cv=5, scoring='r2')
print(f"XGBoost Cross-Validation Scores: {xb_cv_scores}")
print(f"XGBoost Mean Cross-Validation Score: {xb_cv_scores.mean()}")

print()

mae_rf=mean_absolute_error(y_test,y_pred_rf)
mse_rf=mean_squared_error(y_test,y_pred_rf)
print(f'Random Forest - MAE: {mae_rf}, MSE: {mse_rf}')

r2_rf=r2_score(y_test,y_pred_rf)
print(f"Random Forest R² Score: {r2_rf}")

rf_cv_scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='r2')
print(f"Random Forest Cross-Validation Scores: {rf_cv_scores}")
print(f"Random Forest Mean Cross-Validation Score: {rf_cv_scores.mean()}")


# ## So Random Forest is the better model overall in this case.

# In[50]:


pickle.dump(rf,open('rfModel.pkl','wb'))




