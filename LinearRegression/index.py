import matplotlib.pyplot as plt
import numpy as np
import random 
from linearregressionModel import Model 

# Generating random pseudolinear data
optimalM = -1 # Underlying slope without noise (the "m" in y=mx+b)
optimalB = 4 # Underlying shift without noise (the "b" in y=mx+b)
variance = 3 # Intensity of the noise
x = np.arange(0,10,0.1) # x-values 
y = [optimalM * i + optimalB + random.normalvariate(0,variance) for i in x] # y values = slope * x values + shift + noise
# The numpy vector "line" has two components, m and b so line= np.array([m,b])

# Our linear model : line = [slope, shift]

linearModel = Model()
linearModel.train(x, y, 0.01, 0.0001)

plt.scatter(x,y,label="dataset") # Scatter plot of the data set of points
plt.plot(x,linearModel.line[0]* x + linearModel.line[1], label="line") # Plot of the optimal line we computed 
plt.legend()
plt.show()  

print("Prediction for 35: ", linearModel.predict(35))