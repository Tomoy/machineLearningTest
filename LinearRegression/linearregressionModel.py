import math
import random 
import numpy as np

#learning_rate = 0.0001
#tolerance = 0.01

# Generating random pseudolinear data
#optimalM = -1 # Underlying slope without noise (the "m" in y=mx+b)
#optimalB = 4 # Underlying shift without noise (the "b" in y=mx+b)
#variance = 3 # Intensity of the noise
#x = np.arange(0,10,0.1) # x-values 
#y = [optimalM * i + optimalB + random.normalvariate(0,variance) for i in x] # y values = slope * x values + shift + noise
# The numpy vector "line" has two components, m and b so line= np.array([m,b])

# Our linear model : line = [slope, shift]

class Model:
    
    def __init__(self):
        self.line = np.array([random.normalvariate(0,1),random.normalvariate(0,1)])

    # Sum of the distances between the line and the y-value of the data points C = sum_i (y_i - (m*x_i + b))^2
    #def cost_function(self,x,y):
    #    return np.linalg.norm(y-self.line[0]*x-self.line[1])

    # Derivatives of C with respect to m and b
    def gradient(self,x,y):
        # line[0] = m
        # line[1] = b
        derivative_m = 2*np.sum( (y-(self.line[0] * x+self.line[1]))*(-x) ) # Formula for the deriative of C with respect to m
        derivative_b = 2*np.sum( (y-(self.line[0] * x+self.line[1]))*(-1) ) # Formula for the deriative of C with respect to b
        return np.array([derivative_m, derivative_b]) 

    # Update function in the loop which changes the line a little bit to reduce the cost function value
    def update_line(self,x,y, learning_rate):
        return self.line - learning_rate * self.gradient(x,y)

    def train(self,xValues, yValues, tolerance, learning_rate):
        error=2*tolerance
        while(error >= tolerance):
            print("Error: ", error)
            self.line = self.update_line(xValues,yValues, learning_rate)
            error = np.linalg.norm(self.gradient(xValues,yValues))

    def predict(self, x):
        return self.line[0] * x + self.line[1]
