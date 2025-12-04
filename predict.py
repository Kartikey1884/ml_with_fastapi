import joblib
import numpy as np

saved_model = joblib.load('housing_model.pkl')
print("loaded the model from housing_model.pkl")

def make_predict(data:dict)->float: 
      features = np.array([[
            data['longitude'],
            data['latitude'],
            data['housing_median_age'],
            data['total_rooms'],

            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income']
      ]])
      return saved_model.predict(features)[0]
