import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.dropna()

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [-1])], remainder='passthrough')
X = ct.fit_transform(X)

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')

model=Sequential()
model.add(Dense(64, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'));

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100)

# A 40 year old male who has hypertension, has heart disease, is married, has a private job, lives at urban, has a glucose level of 170, has a bmi of 20, and smokes

prediction = model.predict([[0, 0, 1, 0, 40, 1, 1, 1, 0, 0, 170, 20]])
print(prediction)