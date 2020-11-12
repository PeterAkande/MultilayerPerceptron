import numpy as np
from warnings import warn
"""@author:Akande Peter Oluwatobi
#Todo: Add PCA and Whitening preprocessing technique
"""
def one_hot_encode(l,n):
	ret=np.zeros((len(l),n))
	for i,j in enumerate(l):
		ret[i,j]=1
	return ret
"""	
def min_max(mat):
    le=len(mat)
    pl=len(mat[0])
    for i in range(pl):
        hj=[float(mt[i]) for mt in mat]
        min1=min(hj);max1=max(hj)
        rane=max1-min1
        for l in range(le):
            mat[l][i]=(float(mat[l][i])-min1)/rane
    return mat"""
def min_max(mat):
	mat=np.array(mat,dtype=np.float32)
	mins=np.min(mat,axis=0)
	maxs=np.max(mat,axis=0)
	_range=maxs-mins
	mat=(mat-mins)/_range
	return mat,mins,_range
def min_max_test(mat,mins,_range):
	#Perform min_max scaling on test data from 
	#min and range learnt from training data
	mat=np.array(mat,dtype=np.float32)
	res=(mat-mins)/_range
	return res
def labels(lab):
	#This transforms categoral label to integer 
	#encoded labels
	#It is mainly suitable for Binary labels
	#use labels Instead
	warn("This Works Only for Binary Labels,Use \"label\" function instead",UserWarning)
	gh=lab[0]
	for o,x in enumerate(lab):
		if x==gh:
			lab[o]=1
		else:
			lab[o]=0
	return lab

def lab(hjk,mt):
    """Cant still remember what i used this function for And lazy to do so ^_^ """
    d={}
    l=[]
    kl=[]
    for i in mt:
        if i not in d.keys():
            d[i]=0
        d[i]+=1
    s=sorted(d.items(),key=lambda x:x[1],reverse=True)
    for i,j in zip(mt,hjk):
        if i==s[0][0] or i==s[1][0]:
            l.append(i)
            kl.append(j)
    return kl,l

def label(labels):
	"""this function transforms categorical labels to integer encoded labels"""
	lab_list=[]
	tag={}
	lab=list(set(labels))
	for i,j in enumerate(lab):
		tag[j]=i
	for i in labels:
		lab_list.append(tag[i])
	return lab_list,tag
def str_to_float(array):
	"""For datasets whose digit are read as strings"""
	array=np.array(array,dtype=np.float32)
	return array
def norm(array):
	"""This normalizes the dataset...another option is to use the scikit-learn's StandardScaler module"""
	array=np.array(array)
	mean=np.mean(array,axis=0)
	
	std=np.std(array,axis=0)
	ret=(array-mean)/std
	return ret,mean,std
	return ret,mean,std
def norm_test(array,std,mean):
	"""this is to normalize test data...using the standard deviation and mean learnt from the training data, evquivalent yo scikit-lean's transform method on test data after fit_transform on test data"""
	ret=(array-mean)/std
	return ret
