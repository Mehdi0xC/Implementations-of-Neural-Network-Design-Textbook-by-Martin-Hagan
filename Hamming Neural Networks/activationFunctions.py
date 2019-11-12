#IMPORT ESSENTIAL FILES
import numpy as np
#MAIN PART
def sign(x):
    if(x >= 0):
        return 1
    else:
        return 0
    return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ssign(x):
    return np.sign(x)

def lin(x):
    return x

def satl(x):
    if(x<0):
        return 0
    elif(x<=1):
        return x
    else:
        return 1

def posl(x):
    if(x<0):
        return 0
    else:
        return x

def compet(x):
    x[np.where(x!=np.max(x))] = 0
    x[np.where(x==np.max(x))] = 1
    return x 
