import pandas as pd
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error 	
from sklearn.metrics import mean_absolute_percentage_error 	
from tensorflow.keras.models import Sequential
import random
import model as mymodel
import myio
import math
NUMBER_BYTES_PER_KILOBYTE=1024
						
def computeR2(actual,preds):


	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	#no null readings?
	if numNonNulls== len(preds):
		r2=r2_score(actual,preds)		
	#there is at least two non-null prediction (R2 for 1 is not appropriate or not prediction not defined)?
	elif numNonNulls >1:
		r2=r2_score(actual[nonNullIxs],preds[nonNullIxs])#subset of  predictions with only non-null values			
	else:
		r2=0			
	
	if r2 <0:
		r2=0 #a model worse than the naive model who predicts mean can be replaced by zero R with 0 R2				
			
	return r2
	
	
def computeMSE(actual,preds):

	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	
	#no null readings?
	if numNonNulls== len(preds):
		return mean_squared_error(actual,preds)
	#there is at least one non-null prediction?
	elif numNonNulls >0:		
		return mean_squared_error(actual[nonNullIxs],preds[nonNullIxs])
	else:
		return math.inf #infinite error		


def computeMAPE(actual,preds,smallAbsValLim=0):

	#can't have nan values here in computation
	nonNullIxs =  np.isfinite(preds)		
	
	numNonNulls=0
	for f in nonNullIxs:
		#non null element?
		if f:
			numNonNulls = numNonNulls +1
			
	
	#no null readings?
	if numNonNulls== len(preds):
		return computeMAPEHelper(actual,preds,smallAbsValLim)
	#there is at least one non-null prediction?
	elif numNonNulls >0:		
		return computeMAPEHelper(actual[nonNullIxs],preds[nonNullIxs],smallAbsValLim)
	else:
		return math.inf #infinite error		
		
#compuate mean absolute percentage error (as a ratio, so will need multiply by 100 to convert ratio to percentage)
#smallAbsValLim: the value used to limit the range of small values near 0. By default no smal values are replaced,
#		but when set to a positive number, any prediction or expect value within 0 and that limit, in terms of absoluate
#		value are convert to the respective postiive or negative limit, which ever the value is closest too.
#		Example: with smallValueReplaceLim = 2, any value in the range [0,2) and (-2,0) will be converted to 2 and -2, respectively.
#		This limits large unmeaninful errors due to small actual values
def computeMAPEHelper(actual,pred,smallAbsValLim=0):

	#small value  replacement
	if smallAbsValLim>0:
		if len(actual) != len(pred):
			raise Exception("Cannot compute MAPE, the predictions (length: "+len(actual)+") and expected (length: "+len(pred)+") value should be the same length, but we different lengths.")
			
		n=len(actual)
		sum = 0
		
		lowerLim=-1*smallAbsValLim
		for i in range(n):			
			x = actual[i]
			y = pred[i]
			
			#within positive limit of zero?
			if x >= 0 and x < smallAbsValLim: 
				#convert to smallest positive value
				x=smallAbsValLim
			elif x > lowerLim and x < 0: 
				#convert to largest negative value
				x=lowerLim
				
			#within positive limit of zero?
			if y >= 0 and y < smallAbsValLim: 
				#convert to smallest positive value
				y=smallAbsValLim
			elif y > lowerLim and y < 0: 
				#convert to largest negative value
				y=lowerLim
			
			sum=sum+abs((x-y)/x)
		
		return sum/n
	else:
		return mean_absolute_percentage_error(actual,pred) #no value replacement
#append content of list 2 to list 1
def listAppend(list1,list2):
	for val in list2:
		list1.append(val)
		
#returns list of row indices of dataset randonmly shuffled
def randomRowIndexShuffle(df):
	if isinstance(df,pd.DataFrame):
	
		#number of samples in dataset
		nSamples = len(df.index) 
	elif isinstance(df,int):
		nSamples = df
	else:
		raise Exception ("Cannot perform random shuffle. Invalid type for df Expected pandas dataframe or number of samples")
	
	
	
	

	#randomly shuflle indices	
	indices = []
	for i in range(nSamples):
		indices.append(i)
	random.shuffle(indices)

	return 	indices		

	
#converts a list of unique numbers into a a list of rankings
#such that the smallest number has rank 0, and the largest number has rank = len of list -1	
def rankUniqueNumberList(uniqueNums):
	
	#integrety check to make sure everything is unique and a number
	uniqueItemMap={}
	for i in range(len(uniqueNums)):
		val=uniqueNums[i]
		isInt = (type(val)==int) or (type(val)==np.int64)
		isFloat =(type(val)==float) or (type(val)==np.float64)
		if not  isInt and not isFloat:
			raise Exception("Cannot rank unique number list. Expected all numbere but not a number ("+str(val)+" type "+str(type(val))+") found at index : "+str(i))
		if val in uniqueItemMap:
			raise Exception("Cannot rank unique number list. Assumption of unique elements was broken")
		else:
			uniqueItemMap[val]=None
	
	
	ranks = []
	
	#iterate over each number
	for i in range(len(uniqueNums)):
		num = uniqueNums[i]
		
		rank = 0
		#find the rank of the number
		for j in range(len(uniqueNums)):
			#don't compare the number to itself
			if i == j:
				continue
			
			#found a smaller number?
			if uniqueNums[j] < num:
				rank = rank +1
		
		ranks.append(rank)
			
	return ranks

#returns true if numpy shapes (a list of dimensions) are exactly the same
#and false if they differ in number of dimensiosn or dimension lengths
#shapeA: shape to compare
#shapeB: shape to compare
#dimBlacList: list of dimension indices that can differ in size and still count as equal
def shapesEqual(shapeA,shapeB,dimIgnoreList=[]):
	#check number of dimensiosn
	if len(shapeA) != len(shapeB):
		return False
		
	#check dimension lengths
	for i in range(len(shapeA)):
		
		ignoreI = False
		#dimension i can be ignored?
		for j in dimIgnoreList:
			if i == j:
				ignoreI=True
				break
		if ignoreI:
			continue
			
		aDim = shapeA[i]
		bDim = shapeB[i]
		
		if aDim != bDim:
			return False
	return True
	
#bytes taken from numpy array
def npArrayMemoryUsage(arr):
	return arr.size * arr.itemsize
#KB
def computeModelWeightMemoryUsage(model):
	
	if isinstance(model,mymodel.Model):
		myio.logWrite("Cannot compute model weight memory usage, because a model wrapper was provided instead of the raw model.",myio.LOG_LEVEL_ERROR)
		return 0
	#deep learning model?
	if isinstance(model,Sequential):
		totalByteSize=0
		weights=model.weights    
		
		#flatten the weights into lists of bias and weights for each layer
		flatten_w = [i.numpy().flatten() for i in weights]
		for layerWeights in flatten_w:
			#compute size of wights of flattened layer
			totalByteSize= totalByteSize + npArrayMemoryUsage(layerWeights)

		totalKBSize = totalByteSize /NUMBER_BYTES_PER_KILOBYTE
		return totalKBSize
	else:
		#not implemented yet
		return 0