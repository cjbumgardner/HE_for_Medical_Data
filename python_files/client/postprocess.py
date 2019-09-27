import numpy as np 

def log_reg_sigmoid(linear_pred):
    e = np.exp(-linear_pred)
    sigmoid = 1/(1+e)
    return sigmoid


