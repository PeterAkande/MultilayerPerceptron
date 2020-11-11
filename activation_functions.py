import numpy as np
#Todo: add softmax and relu activations
def sig_derv(z):
	return (z)*(1-(z))
	

def sigmoid(z):
	return 1/(1+np.exp(-z))