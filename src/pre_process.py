# Data Preprocessing
# pre_prrocess.py

import pandas as pd

def PreProcess(df, num_features = 16):
    data = df
    data.dropna(inplace=True)
    features = ['tempmax', 'tempmin','temp', 'feelslikemax',
        'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipcover', 'snow', 'snowdepth', 'windgust',
        'windspeed','sealevelpressure', 'cloudcover', 'visibility',
        'solarradiation', 'solarenergy', 'uvindex', 'moonphase']
    feature_data = data[features]
    variance = feature_data.var()
    important_features = variance.sort_values(ascending=False)
    print("Important features for clustering:")
    print(important_features)
    features = important_features.index[:num_features] 
    data = data[features]
    means = data.mean()
    stds = data.std()
    data = (data - means) / stds
    return data
