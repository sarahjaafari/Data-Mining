# Sara Aljaafari - 169044425
# Assignent 1 - CP421

# Question 1 - 1.a)
from sklearn.datasets import fetch_california_housing
import pandas as pd
#1.b)
housing_dataset=fetch_california_housing()
housing =pd.DataFrame(data=housing_dataset.data, columns=housing_dataset.feature_names)
print(housing.head())

# 2.a)
print(housing.shape)
print(housing.columns)
print(housing.dtypes)
print(housing.describe())

#2.b)
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
#plt.show()

# 3.a)
import numpy as np
missing=np.random.choice(housing.index,size=int(len(housing)*0.1),replace=False)
housing.loc[missing, 'AveRooms']=np.nan
housing.loc[missing, 'AveOccup']=np.nan

# 3.b)
housing['AveRooms'].fillna(housing['AveRooms'].mean(), inplace=True)
housing['AveOccup'].fillna(housing['AveOccup'].mean(), inplace=True)

# 4.a b)
from sklearn.preprocessing import StandardScaler
target=housing_dataset.target
housing_features = housing.copy()
scaler=StandardScaler()
housing_scaled=pd.DataFrame(scaler.fit_transform(housing_features), columns=housing_features.columns)
housing_scaled['target'] =target
print(housing_scaled.head())

# QUESTON 2
# 1)
from sklearn.model_selection import train_test_split
X=housing_scaled.drop('target',axis=1)
y=housing_scaled['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

#2.a)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
# Linear regression
linear_model=LinearRegression()
linear_model.fit(X_train, y_train)
lasso_model =Lasso(alpha=0.1, random_state=42)
lasso_model.fit(X_train, y_train)
ridge_model=Ridge(alpha=1, random_state=42)
ridge_model.fit(X_train, y_train)

#2.b)
from sklearn.model_selection import GridSearchCV
alphas= {'alpha': [0.01, 0.1, 1, 10]}
lasso_grid=GridSearchCV(Lasso(),alphas, cv=5,scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
ridge_grid=GridSearchCV(Ridge(),alphas, cv=5,scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
lasso_best_est=lasso_grid.best_estimator_
ridge_best_est=ridge_grid.best_estimator_
print(lasso_grid.best_params_)
print(ridge_grid.best_params_)

#3)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
linear_pred=linear_model.predict(X_test)
lasso_pred=lasso_best_est.predict(X_test)
ridge_pred=ridge_best_est.predict(X_test)
def evaluate_model(name, y_true, y_pred):
    mse=mean_squared_error(y_true, y_pred)
    mae=mean_absolute_error(y_true, y_pred)
    r2= r2_score(y_true, y_pred)
    print({name})
    print(f"MSE:{mse}")
    print(f"MAE:{mae}")
    print(f"R^2:{r2}")
    print()
#evaluations
evaluate_model("Linear Regression",y_test,linear_pred)
evaluate_model("Lasso",y_test,lasso_pred)
evaluate_model("Ridge",y_test, ridge_pred)

# QUESTION 3
#1)
medianval =housing_scaled.target.median()
binary_target = (housing_scaled.target>medianval).astype(int)
housing_scaled['binary_target']=binary_target
print(housing_scaled.head())

#2
from sklearn.linear_model import LogisticRegression
X = housing_scaled.drop(['target','binary_target'], axis=1)
y = housing_scaled['binary_target']
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model= LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

#3)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
#import seaborn as sns
y_pred = logistic_model.predict(X_test)
y_pred_prob = logistic_model.predict_proba(X_test)[:,1]
#evaluations
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
roc_auc=roc_auc_score(y_test,y_pred_prob)
# confusion matric
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)

#QUESTION 4
#1)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
kmcluster = housing_scaled.drop(['target','binary_target'],axis=1) 
ssd=[]
kk=range(1,11)
for k in kk:
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(kmcluster)
    ssd.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
plt.plot(kk,ssd,marker='o') 
#plt.show()
# ELbow point=3 (it's where I realized the rate of decrease in SSD slows significantly)

#1c0
elbow=3
kmeans=KMeans(n_clusters=elbow, random_state=42)
kmeans.fit(kmcluster)
labels=kmeans.labels_
pca= PCA(n_components=2)
pca2=pca.fit_transform(kmcluster)
plt.figure(figsize=(10,5))
plt.scatter(pca2[:,0],pca2[:,1],c=labels,cmap='viridis',marker='o')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
#plt.show()

#2)
from sklearn.mixture import GaussianMixture
guassmix=GaussianMixture(n_components=2, random_state=42)
guassmix_fit= guassmix.fit_predict(kmcluster)
plt.figure(figsize=(10,5))
plt.scatter(pca2[:,0],pca2[:,1],c=guassmix_fit,cmap='viridis',marker='o')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
#plt.show()

#3)
from sklearn.metrics import silhouette_score
kmeans_silhouette = silhouette_score(kmcluster, labels)
guassmix_silhouette = silhouette_score(kmcluster, guassmix_fit)
print(f"KMeans Silhouette Score: {kmeans_silhouette}")
print(f"Gaussian Mixture Silhouette Score: {guassmix_silhouette}")

# K means is easy to implement and efficient and better for large datasets. But its weaknes
# is that the number of cluster need to be specified in advance, despite that it is more
#efficient to test elbow method to find the best number of clusters.
# Guassian mixture is more felxible with clustering as it can model clusters with different
# sizes. But it is more complex to implement and understand than K means.
