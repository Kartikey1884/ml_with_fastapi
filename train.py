import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('my_housing.csv').iloc[:,:-1].dropna()
print("read the dataset")
X=df.drop(columns=['median_house_value'])
y=df['median_house_value']
print("split the dataset into X and y")

model=LinearRegression().fit(X,y)
print("trained the model")

joblib.dump(model,'housing_model.pkl')
print("saved the model to housing_model.pkl")