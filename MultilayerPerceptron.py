from pickle import load
from pickle import dump
import numpy as np
import random
import sys
from utils import one_hot_encode
from activation_functions import sigmoid
from activation_functions import sig_derv
"""
@author:Akande Peter Oluwatobi
"""

class MLP(object):
	"""MLP using stacked sigmoids for multiclass classification"""
	def __init__(self,layers):
		#l: a list--len(list) is the amount of layes the neural net should have and the value of each indx is the corresponding amount of units/neurons in each layer.
		#Weights are initialized using np.random.randn
		#for unit variance and zero mean
		self.layers=layers
		self.weights=[np.random.randn(self.layers[i+1],self.layers[i]) for i in range(len(self.layers)-1)]
		#No harm in initializing biases with 0
		self.biases=[np.zeros((self.layers[i],1)) for i in range(1,len(self.layers))]
	def train(self,data,labels,epochs=100,batchsize=5,l_r=0.1,one_hot=False):
		"""data: preprocessed data
		labels:labels of the inputed data
		epochs: Times for each iteration
		batchsize: Amount of data to be processed at once...required for SGD
		l_r: Learning rate
		one_hot:For multiclass classification,set it to true if your Labels have been One_hot encoded"""
		data=np.array(data)
		labs=np.array(labels)
		#dzdbs is the derivative of inputs
		#entering a node
		#since a neural net is 
		#typically represented by the equation
		#y=mx+b
		#derivative of it would give 1
		#so we initialize dzdbs with ones.
		dzdbs=[np.ones((self.layers[i],1)) for i in range(1,len(self.layers))]
		for i in range(epochs):
			#For SGD, randomly pick a batch
			#to train with
			k=random.randint(0,data.shape[0]-batchsize)
			datamat=data[k:k+batchsize]
			labels=labs[k:k+batchsize]
			if not one_hot:
				labels=one_hot_encode(labels,self.layers[-1])
			labels=labels.T
			activations=self.forward_propagation(datamat,labels)
			loss=self.cost(datamat.shape[0],activations,labels)
			if i%100==0:
				#print loss at 100 iterations
				#interval
				print("Loss at {}th Epoch is {}".format(i,loss))
			dcdbs,dcdws=self.backpropagation(datamat.shape[0],labels,activations,dzdbs)
			self.update(dcdws,dcdbs,l_r)
	def forward_propagation(self,datamat,labels):
		#activation of first layer
		#cant be put in the loop cos of
		#it is the input!			
		activations=[]		
		activations.append(np.array(datamat))
		zs=[]
		"""list to store output before activation function ia applied"""
		z=np.dot(self.weights[0],activations[0].T)+self.biases[0]
		#FORWARD PROPAGATION
		for ij in range(1,len(self.layers)-1):	
			zs.append(z)
			activation=sigmoid(z)
			activations.append(activation.T)
			z=np.dot(self.weights[ij],activations[len(activations)-1].T)+self.biases[ij]
		zs.append(z)
		activation=sigmoid(z)
		activations.append(activation)
		return activations
	def cost(self,N,activations,labels):
		"""calculate loss using mean squared ertpr loss function"""
		cost=(1/(2*N))*np.power((labels-activations[-1]),2)
		return cost.sum()
	def backpropagation(self,N,labels,activations,dzdbs):
		#Performs BackPropagation
		#Would Explain the little
		# abracadabra (To beginners tho)
		#That happened Here ^_^
		# in a blog post soon.
		
		#ty is the cost derivative
		ty=(-1/N)*(labels-activations[-1])
		dcdas,dcdbs,dcdws,dadzs=[],[],[],[]
		dcdzs=[]
		dcdas.append(ty)
		for ig in range(len(activations)-1):
			activations[ig]=activations[ig].T
		#BACKWARD PROPAGATION
		for ik in range(len(self.layers)-2,-1,-1):
			dadz=sig_derv(activations[ik+1])
			dadzs.append(dadz)
			dcdz=dadz*dcdas[len(dcdas)-1]
			dcdzs.append(dcdz)
			dcdw=np.dot(dcdz,activations[ik].T)
			dcdws.append(dcdw)
			dcdb=dcdz*dzdbs[ik]
			dcdb=np.sum(dcdb,axis=1)
			dcdbs.append(dcdb)
			dcda=np.dot(self.weights[ik].T,dcdz)
			dcdas.append(dcda)
		return dcdbs,dcdws
	def update(self,dcdws,dcdbs,l_r):
		u=[i for i in range(len(self.weights))]
		h=[i for i in range(len(self.weights)-1,-1,-1)]
		#Bias Update
		for i in range(len(dcdbs)):
			dcdbs[i]=dcdbs[i].reshape(-1,1)
		#Weights Update
		for i,j in zip(u,h):
			self.weights[i]-=l_r*dcdws[j]
			self.biases[i]-=l_r*dcdbs[j]
	def save_weights(self,wei,bias):
		#Save weights to Disk
		with open(wei,"wb") as weii:
			pickle.dump(self.weights,wei)
		with open(bias,"wb") as bia:
			pickle.dump(self.biases,bia)
		print("Model saved!")
	#@classmethod  so the load_weights function
	#van be called without defining any layers
	#or initializations.
	#can simply be called by
	#MLP.load_weights(wei,bia)
	#And predictions can follow
	@classmethod
	def load_weights(self,wei,bia):
		#load Weights from Disk
		with open(wei,"rb") as weii:
			self.weights=pickle.load(weii)
		with open(bia,"rb") as bias:
			self.biases=pickle.load(bias)
	def predict(self,data,tag=None):
		#Function to make predictions
		#Returns a list of predictions
		#If tag is given, a mapping from integer 
		#predictions to Categorical Predictions
		#is done
		if data.ndim!=2:
			data=data.reshape(1,-1)
		z=np.dot(self.weights[0],data.T)+self.biases[0]
		for i in range(1,len(self.weights)):	
			act=sigmoid(z)
			z=np.dot(self.weights[i],act)+self.biases[i]
		act=sigmoid(z)
		res=np.argmax(act,axis=0)
		res1=[]
		if tag is not None:
			for i in res:
				res1.append(tag[i])
			return res1
				
		return res