import joblib
import numpy as np
from typing import List

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


def make_batch_predict(data:List[dict])->List[float]:
      x = np.array([[
            x['longitude'],
            x['latitude'],
            x['housing_median_age'],
            x['total_rooms'],
            x['total_bedrooms'],
            x['population'],
            x['households'],
            x['median_income']
      ] for x in data])
      return saved_model.predict(x)
