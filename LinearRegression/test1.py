import math
import random 
import numpy as np



factor1 = np.array([2058.00247131, 1029.79212325,  618.58952007,  735.45166487,
        824.13258163, 1569.08859442,  712.87173841,  929.82124327])


x_int = np.array([[100,  50,  30,  60,  40, 120,  20,  50],
       [  1,   3,   1,   2,   4,   7,   4,   2],
       [  1,   1,   1,   1,   1,   1,   1,   1]])
    
test = factor1 * x_int

print("Test: ", test)

# linear_map = np.array([random.normalvariate(0,1) for i in range(2+1)])

# def gradient(y):
#     x_int = np.array([[100,  50,  30,  60,  40, 120,  20,  50],
#        [  1,   3,   1,   2,   4,   7,   4,   2],
#        [  1,   1,   1,   1,   1,   1,   1,   1]])
#     print("X Int: ", x_int)
#     print("Y: ", y)
#     print("np.dot substraction: ", (np.array(y) - np.dot(x_int, linear_map)))
#     transpodedX = x_int.T
#     print("TransportedX: ", transpodedX)
#     print("np.dot multipl: ", (y - np.dot(x_int, linear_map))) * (-transpodedX)
#     return 2 * np.sum( ((np.array(y) - np.dot(x_int, linear_map)) ) * (-x_int).T )

# gradient(np.array([[1000, 600, 700, 800 ,1500, 700, 900]]))
