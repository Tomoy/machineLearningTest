import numpy as np

def sigmoid(z):
    z = np.array(z, dtype='f')
    return 1/(1.0 + np.exp(-z))

def sigmoidPrime(z):
    w = sigmoid(z)
    return w * (1-w)

class Model:
    def __init__(self):
        self.linear_map = np.array([])

    # Derivatives of C with respect to m and b
    def gradient(self,x,y):
        x_int = np.concatenate( (x, np.ones((len(x),1)) ), axis=1)
        return 2 * np.sum( (np.array(y) - sigmoid(np.dot(x_int, self.linear_map))) * sigmoidPrime(np.dot(x_int,self.linear_map)) * (-x_int.T), axis=1) # Formula for the deriative of C with respect to m

    # Update function in the loop which changes the line a little bit to reduce the cost function value
    def update_line(self,x,y, learning_rate):
        return self.linear_map - learning_rate * self.gradient(x,y)

    # Train the values of m and b
    def train(self,xValues, yValues, tolerance, learning_rate):
        error = 2 * tolerance
        # Initializing the number of coefficients of m required for our model 
        # according to our training data
        self.linear_map = np.random.normal(0,1,len(xValues.T)+1)
        while(error >= tolerance):
            #print("Error: ", error)
            self.linear_map = self.update_line(xValues,yValues, learning_rate)
            error = np.linalg.norm(self.gradient(xValues,yValues))

        print(len(self.linear_map))
        print(len(xValues.T))


    def predict(self, x):
        x_int = np.concatenate( (x, np.ones((len(x),1)) ), axis=1)
        # This contains the probabilities of survival
        return sigmoid(np.dot(x_int,self.linear_map))
        # This contains a list just like the line above but we applied the function
        # if x > 0.5 then 1 else 0
        # on each entry of the list
        return np.where(sigmoid(np.dot(x_int,self.linear_map)) < 0.5, 0, 1)
    
