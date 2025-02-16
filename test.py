from sklearn.datasets import fetch_california_housing
import pandas as pd
#1.b)
housing_dataset = fetch_california_housing()
housing = pd.DataFrame(data=housing_dataset.data, columns=housing_dataset.feature_names)
print(housing.head())
print(housing.shape)
print(housing.columns)
print(housing.dtypes)
print(housing.describe())

#2.b)
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
#plt.show()