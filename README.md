
# MultilayerPerceptron
Just Got up and Decided to clear the drought of repositories on my GitHub page.
It Works on the principle of Back Propagation (All MLPs Do 😂), the ouput layer is Made up of Stacked Sigmoids...Lazy to adapt it to Softmax 😪.
The network is initialized with the training set and the accompanying labels and  list, whose length connotes the amount of layers and the value of each layer index in the list denotes the amount of nodes that particular  layer should have,also a one_hot parameter ,that is by  default  set  to false...But can be Set to True if labels are already one_hot_encoded.
The first layer of the list to be passed should be the Number of features and the last item of the list should be the amount of classes.

#Example usage
#Import class

from MultilayerPerceptron import MLP

...
...#Your preferred
#Preprocessing Technique
#And Loading Of Data
Layers=[60,40,20,10]
#The MLP
#has 2 hidden Layers
#,len(Layers)-2.
#first item
#of Layers,60,
#is the amount of
#features in each
#instance of training data
#Last item of Layers
#10, is number of classes
n=MLP(Layers)
n.train(data,labels, epochs,batchsize,l_r,one_hot=True)
#l_r is the learning_rate
#save weights with
#save_weights function
n.save_weights(weights_path,bias_path)
n.load_weights(weights_path,bias_path)
#Load weights from disk.
#load_weights is
#wrapped with
#classmethod decorator
#so after weights
#and bias has been
#saved with n.save_weights(...)
#The weights can
#then be used to make
#predictions for a later time
#Without initializing
#the layers...E.g 
#n=MLP.load_weights(...)
n.predict(data)
#n.predicts can
#then be used to
#make predictions...

#😄



The utils function is mainly for preprocessing
Some functions included are:
(1). Standardization :::utils.norm(mat), it returns the transformed dataset, the std learned and the mean
,Which can then be used on the test set by using utils.norm_test(mat,std,mean)
(2). Min-Max Scaling::utils.min_max(mat),it returns the transformed dataset,the min, and the range.
,Which can then be used on the test set by using utils.min_max_test(mat,mins,_range)
###PCA and whitening should be added soon



Would write a blog post on how MLPs work, using this repo as reference, soon..
