# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Data encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X_1 = LabelEncoder()
X[:,1] = le_X_1.fit_transform(X[:,1])
le_X_2 = LabelEncoder()
X[:,2] = le_X_2.fit_transform(X[:,2])

hotencoder = OneHotEncoder(categorical_features = [1])
X = hotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Artificial neural networks
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Initialising
classifier = Sequential()

# Input and hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compile model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting model to training set
classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 100)

# Predicition on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
