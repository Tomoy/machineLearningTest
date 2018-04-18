import math
import random 
import numpy as np

class Model:
    def __init__(self, n_of_parameters):
        self.linear_map = np.array([random.normalvariate(0,1) for i in range(n_of_parameters+1)])

    # Derivatives of C with respect to m and b
    def gradient(self,x,y):
        x_int = np.concatenate( (x, np.array([[1] for i in range(len(x))]) ), axis=1)
        transpodedX = x_int.T        
        #print("np.dot multipl: ", (np.array(y) - np.dot(x_int, self.linear_map)) * (-transpodedX))
        return 2 * np.sum( ((np.array(y) - np.dot(x_int, self.linear_map))  * (-transpodedX ))) # Formula for the deriative of C with respect to m

    # Update function in the loop which changes the line a little bit to reduce the cost function value
    def update_line(self,x,y, learning_rate):
        return self.linear_map - learning_rate * self.gradient(x,y)

    def train(self,xValues, yValues, tolerance, learning_rate):
        error = 2 * tolerance
        while(error >= tolerance):
            #print("Error: ", error)
            self.linear_map = self.update_line(xValues,yValues, learning_rate)
            error = np.linalg.norm(self.gradient(xValues,yValues))

    def predict(self, x):
        x_int = np.concatenate( (x, np.array([[1] for i in range(len(x))]) ), axis=1).T
        return np.dot(self.linear_map, x_int)[0]
    
