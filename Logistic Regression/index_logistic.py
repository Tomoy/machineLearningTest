print("Loading libraries...") 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logisticRegressionModel import Model

print("Loading the data...") 

train_data = pd.read_csv("data_sets/training_clean.csv", index_col=0)
test_data = pd.read_csv("data_sets/submission_clean.csv", index_col=0)

print("Setting up the data...") 

x = np.array(train_data.drop(['Survived'], axis=1))
x_test = np.array(test_data) ## ERROR FOUND HERE
# ValueError : labels['Survived'] not contained in axis
y = np.array(train_data['Survived'])

print("Training...")

linearModel = Model()
linearModel.train(x, y, 5, 0.001)

print("Welcome to the death predictor!\n")

print("Test : ")
test = linearModel.predict(x)
print(test)
print("Prediction : ")
prediction = linearModel.predict(x_test)
print(prediction)

