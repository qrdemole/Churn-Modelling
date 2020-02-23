# 1. Read Dataset
# 2. Extract valuable features
# 3. Data preprocessing
#   3.1 Remove strings from data
#   3.2 Does Spain is more important than France?(Achtung! Dummy variables trap!)
#   3.3 Informations are equal to each other
# 4. Divide dataset to train and test set
# 5. Create Neural Network
# 6. Compile and Fit
# 7. Let's predict
# 8. Congrats!

import pandas as pd
dataset = pd.read_csv("Churn_Modelling.csv")
pd.options.display.max_columns = dataset.shape[1] + 1
pd.options.display.width = 1000
# print(dataset)

x = dataset.iloc[:, 3:13].values
y = dataset['Exited'].values
print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
x[:, 2] = gender_encoder.fit_transform(x[:, 2])
country_encoder = LabelEncoder()
x[:, 1] = country_encoder.fit_transform(x[:, 1])

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(categorical_features=[1])
x = one_hot.fit_transform(x).toarray()
x = x[:, 1:]

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=200)


from sklearn.preprocessing import StandardScaler
standard_sc = StandardScaler()
x = standard_sc.fit_transform(x)
print(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model = Sequential()
model.add(Dense(input_dim=11, units=32, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.6)
print(y_pred)
