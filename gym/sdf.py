import numpy as np

a =np.array([[1,1,1,1],[1,1,1,1]])
b= np.array([[1,1,1,1],[1,1,1,0]])
print((a==b).all())