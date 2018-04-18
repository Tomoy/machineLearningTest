import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiLinearRegressionModel import Model

data_set = pd.read_csv("data_sets/housePricingMultipleVariables.csv", delimiter=';')
#print(data_set.columns)
x = data_set[['SQM2','AGE']]
y = data_set['PRICE']

#print(x)

print("Welcome to the flat price predictor!")
inputSQM = int(raw_input("Please enter the Square meters of your flat \n"))
inputAge = int(raw_input("Please enter the ages of your flat (In Years) \n"))

# Our linear model : line = [slope, shift]

linearModel = Model(2)
linearModel.train(x, y, 100, 0.000001)

print "Prediction for " + str(inputSQM) + " sqm and " + str(inputAge) + " years = ", linearModel.predict([[inputSQM, inputAge]]), " Euros"
