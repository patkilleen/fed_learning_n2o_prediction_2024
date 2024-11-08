import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

TIMESTAMP_COLUMN_NAME="timestamp"
SAMPLE_ID_EXTRA_COLUMN_NAME="sampleid"


#wrapper class to store the min and max of taraget variable, and 
#holds flag to determine if denormalization/unscaling is desired
class DataUnscaler:

	
	def __init__(self, minY,maxY,normalizeDataFlag):
		self.minY=minY
		self.maxY=maxY
		self.normalizeDataFlag=normalizeDataFlag
		
	def unscale(self,test_y):
		#should we apply denormalization to unscale data?
		if self.normalizeDataFlag:
			return denormalize(self.minY,self.maxY,test_y)
		else:
			return test_y
	
	
#compute number of timesteps in a tensor given dataset resolution (in minutes) and the
#tensor length in minutes
#def computeTensorTimeSteps(temporalResMins,tensorLenMins):
#	return (tensorLenMins/temporalResMins) +1
#note that this will be inintially called after DF normalized for CNN or LSTM
#and then in the folds we will take subset of the result of this function via the sampleid of the row for each index in
#fold pairs
#tensorNumTimeSteps: number of timesteps that a tensor will have
#temporalRes: resolution of dataset (in minutes)
#anySizeTensorFlag: when false, does not return tenors that have missing samples due to missing timestamps. true means return all any-sized tensors
def createTensors(df,tensorNumTimeSteps,temporalRes,anySizeTensorFlag):
	originalTimestamps=df[TIMESTAMP_COLUMN_NAME]
	res=[]
	timestamps=[]
	temporalRes = int(temporalRes)
	tensorNumTimeSteps=int(tensorNumTimeSteps)
	
	tensorLenthHours = (temporalRes/60.0) * tensorNumTimeSteps #number of hours the tensor covers
	#number of rows in a tensor
	#E.g.: 30 min (0.5 h) resolution  dataset with tensors size 2 h = 2 / 0.5 + 1 = 5. +1 since the anchor teimstamp also included
	#so like 11:00,11:30,12:00,12:30,13:00 would be 2 h window with 30 min resolution	
	
	df = df.copy(deep=True)#make a copy so we affect anything. Also we assume sorted by timestamp already
	df = df.sort_values(by=[TIMESTAMP_COLUMN_NAME])#sort by timestamp (from early to later readings)	
	df[TIMESTAMP_COLUMN_NAME] = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME])#convert the timestamp column to a datetime format	
	
	#set the index of dataframe to the timestamp
	df.set_index(TIMESTAMP_COLUMN_NAME,inplace=True)
	
	finishedFlag=False
	
	i=0
	
	for i in range(len(df.index)):

		wStart = df.index[i] #starting of the window is first timestamp of dataset
		wEnd = wStart+ timedelta(hours=tensorLenthHours)#ending of the window
	
		#get all samples within time window [tStart,tEnd) in sensor data				
		#get an index of all samples withint lag window
		windowFilter = (df.index>= wStart) & (df.index< wEnd)
		
		#get the samples in window
		subset = df.loc[windowFilter]
		
		#any size tensor returned
		if anySizeTensorFlag:
			res.append(subset)
			#last timestamp of window is considered the tensor's timestamp, since it represents a history of values up to most recent timestamp			
			timestamps.append(subset.index[-1]) 
		else:
			#only include tensors of expected size. Skip tensors that had missing timestamps
			if len(subset.index) == tensorNumTimeSteps:
				res.append(subset)
				#last timestamp of window is considered the tensor's timestamp, since it represents a history of values up to most recent timestamp				
				timestamps.append(subset.index[-1]) 
			
		
		#the window is outside the timestamp range of the dataset?
		if wStart >df.index[-1]:
			finishedFlag=True
		
	#discard
	return res,timestamps
	
#given an array of indices, only extracts the tensors of those indices from df
def createTensorSubset(df,tensorNumTimeSteps,temporalRes,indices):
	tensors,timestamps =createTensors(df,tensorNumTimeSteps,temporalRes,anySizeTensorFlag=True)
	#E.g.: 30 min (0.5 h) resolution  dataset with tensors size 2 h = 2 / 0.5 + 1 = 5. +1 since the anchor teimstamp also included
	#so like 11:00,11:30,12:00,12:30,13:00 would be 2 h window with 30 min resolution
	
	res = []
	resTimestamp=[]
	for i in indices:
		#skip out of bounds indices
		if i < 0 or i >= len(tensors):
			continue
		t = tensors[i]
		
		#only consider tensors of desired index, and only those of appropriate size
		if len(t.index) == tensorNumTimeSteps:
			res.append(t)
			resTimestamp.append(timestamps[i])
	
	
	return res,resTimestamp

	
#dataset is read into memory, filtering out non-selected features, and adding a sampling id column
#inputDatasetPath: file path to dataset with timestamps and a target variable to prepare into ML model friendly input format
#selectedFeaturesPath: file path to CSV that has entries for which column is selected (only 1 row, 1 indicating selected, 0 indicatign not)
def readDataset(inputDatasetPath,selectedFeaturesPath):
	
	if not os.path.exists(inputDatasetPath):
		raise Exception("Input dataset file "+inputDatasetPath+" does not exist.") 
		
	if not os.path.exists(selectedFeaturesPath):
		raise Exception("Selected features dataset file "+selectedFeaturesPath+" does not exist.") 
		
	#read the datsaets
	inDF = pd.read_csv(inputDatasetPath, sep=",")
	selectedFeatDF=pd.read_csv(selectedFeaturesPath, sep=",")
	
	#quality checks: make sure the column names align exactly, other than timestamp and the last feature
	if inDF.columns[0] != TIMESTAMP_COLUMN_NAME:
		raise Exception("Input dataset expected timestamp as first column: "+str(inputDatasetPath))
	
	
	#make sure columns align with both datsets (ignoring timestamp(1st column) and n2o (last colmn)
	for i in range (len(selectedFeatDF.columns)):
		if inDF.columns[i+1] != selectedFeatDF.columns[i]:#-1 since selecte feature dataframe doesn't have a timestamp
			raise Exception("Input dataset ("+inputDatasetPath+") and selected feature dataset ("+selectedFeaturesPath+") don't share the same feature columns ")
	
	#return a dataframe with only the selected sensors
	resMap={}
	
	#create sample id column
	sampleIds = []
	for i in range(len(inDF.index)):
		sampleIds.append(i)
	
	selectedFeatures=[]
	
	#sample id as 1st column, then timestamp as 2nd
	resMap[SAMPLE_ID_EXTRA_COLUMN_NAME]=sampleIds	
	resMap[TIMESTAMP_COLUMN_NAME]=inDF[TIMESTAMP_COLUMN_NAME]
	
	#only include selected features
	for col in selectedFeatDF.columns:
		#feature selected?
		if selectedFeatDF[col][0] == 1:
			resMap[col]=inDF[col]
			selectedFeatures.append(col)
	
	#add target variable to end 
	n2oCol = inDF.columns[-1]
	resMap[n2oCol]=inDF[n2oCol]
	
	
	resDF=pd.DataFrame(data=resMap)
	
	#make sure every value exists (no missing values)
	for i in range(len(resDF.columns)):
		c = resDF.columns[i]
		
		if c != TIMESTAMP_COLUMN_NAME:
			if np.any(np.isnan(resDF[c])):
				raise Exception("Missing value in column "+c+" of the selected feature data.")
	return resDF,selectedFeatures
	
#	return inDF

#here the idea is we have train and test dataset files. In ML, both test and train must share same features
#so how we approach this with flexibility is that the actaul files may have differing features/columns, 
#but this function will make sure to parse the fils such that only columns from training dataset
#taht are selected (specified by selectedFeaturesPath with entry of 1 for selected column) 
#will be found in the resulting train and test datasets. If the test datset doesnt' have a selected
#column from training, we drop that column
def readTrainTestDataset(inputTrainDatasetPath,inputTestDatasetPath,selectedFeaturesPath):

	
	if not os.path.exists(inputTrainDatasetPath):
		raise Exception("Input training dataset file "+inputTrainDatasetPath+" does not exist.") 
	
	if not os.path.exists(inputTestDatasetPath):
		raise Exception("Input test dataset file "+inputTestDatasetPath+" does not exist.") 
		
	if not os.path.exists(selectedFeaturesPath):
		raise Exception("Selected features dataset file "+selectedFeaturesPath+" does not exist.") 

	#read the datsaets
	inTrainDF = pd.read_csv(inputTrainDatasetPath, sep=",")
	inTestDF = pd.read_csv(inputTestDatasetPath, sep=",")
	selectedFeatDF=pd.read_csv(selectedFeaturesPath, sep=",")
	
	#quality checks: make sure the column names align exactly, other than timestamp and the last feature
	if inTrainDF.columns[0] != TIMESTAMP_COLUMN_NAME:
		raise Exception("Input dataset expected timestamp as first column: "+str(inputTrainDatasetPath))
	if inTestDF.columns[0] != TIMESTAMP_COLUMN_NAME:
		raise Exception("Input dataset expected timestamp as first column: "+str(inputTestDatasetPath))
	
	
	
	#make sure columns align with both datsets (ignoring timestamp(1st column) and n2o (last colmn)
	#(don't check test datset, since its fine if some selected features don't exist in test, we will just exclude them 
	#from the resulting train/test datasets)
	for i in range (len(selectedFeatDF.columns)):
		if inTrainDF.columns[i+1] != selectedFeatDF.columns[i]:#-1 since selecte feature dataframe doesn't have a timestamp
			raise Exception("Input dataset ("+inputTrainDatasetPath+") and selected feature dataset ("+selectedFeaturesPath+") don't share the same feature columns ")
	
	#return a dataframe with only the selected sensors
	trainResMap={}
	testResMap={}
	
	#create sample id column
	trainSampleIds = []
	testSampleIds = []
	for i in range(len(inTrainDF.index)):
		trainSampleIds.append(i)
	for i in range(len(inTestDF.index)):
		testSampleIds.append(i)
	
	
	selectedFeatures=[]
	
	#sample id as 1st column, then timestamp as 2nd
	trainResMap[SAMPLE_ID_EXTRA_COLUMN_NAME]=trainSampleIds	
	trainResMap[TIMESTAMP_COLUMN_NAME]=inTrainDF[TIMESTAMP_COLUMN_NAME]
	testResMap[SAMPLE_ID_EXTRA_COLUMN_NAME]=testSampleIds	
	testResMap[TIMESTAMP_COLUMN_NAME]=inTestDF[TIMESTAMP_COLUMN_NAME]
	
	#only include selected features
	for col in selectedFeatDF.columns:
		#feature selected, and exists in test dataset?
		if (selectedFeatDF[col][0] == 1) and (col in inTestDF):
		
			#add selecte feature to train/tests datasets
			trainResMap[col]=inTrainDF[col]
			testResMap[col]=inTestDF[col]
			selectedFeatures.append(col)
	
	if len(selectedFeatures) == 0:
		raise Exception("Could not read train/test datasets into memory. no features were selected or the test dataset did not have any features selected in the training dataset")
		
	#add target variable to end 
	trainN2oCol = inTrainDF.columns[-1]
	trainResMap[trainN2oCol]=inTrainDF[trainN2oCol]
	testN2oCol = inTestDF.columns[-1]
	testResMap[testN2oCol]=inTestDF[testN2oCol]
	
	
	trainResDF=pd.DataFrame(data=trainResMap)
	testResDF=pd.DataFrame(data=testResMap)
	
	#make sure every value exists (no missing values)
	for i in range(len(trainResDF.columns)):
		c = trainResDF.columns[i]
		
		if c != TIMESTAMP_COLUMN_NAME:
			if np.any(np.isnan(trainResDF[c])):
				raise Exception("Missing value in column "+c+" of training data of the selected feature data.")
				
	for i in range(len(testResDF.columns)):
		c = testResDF.columns[i]
		
		if c != TIMESTAMP_COLUMN_NAME:
			if np.any(np.isnan(testResDF[c])):
				raise Exception("Missing value in column "+c+" of test data of the selected feature data.")
				
	return trainResDF,testResDF,selectedFeatures
	

	
#min-max scaling
#returns deep copy of dataset with features and target variable normalized via min-max scaling
#where a target dataset is normalized (targetDF) using the min and max values of features in a source dataset (sourceDF)
#if sourceDF isn't specified, we use the min and max of the target dataset for scaling
#both targetDF and sourceDF should have same column names
def normalize(targetDF,sourceDF=None):
	
	if not sourceDF is None:
		#integrity check. We expect both sets to have same columns (same number)
		if len(targetDF.columns) != len(sourceDF.columns):
			raise Exception("Cannot normlize datsets, the 2 datsets given don't have the same number of columns")
			
		#make sure all features have same name
		
		for i in range(len(targetDF.columns)-1): #we do -1 to ignore the last column, since its the target variable and may have a dffirent name
			if targetDF.columns[i] != sourceDF.columns[i]:
				raise Exception("Cannot normalize datsets, some of the column names do not match ("+targetDF.columns[i]+" vs. "+sourceDF.columns[i]+")")
			
		
	#avoid changign original dataframe by making a copy
	df =targetDF.copy(deep=True)
	
	#normalize data between 0 and 1 via min max scaling
	
	for i in range(len(df.columns)):	
		c = df.columns[i]
		#dont' include timestamp or sample id in normalization
		if c != TIMESTAMP_COLUMN_NAME and c != SAMPLE_ID_EXTRA_COLUMN_NAME: #this check is for compatibility, just in case we didn't set the timestamp as index in dataframe ()
		
			
			#iterate over each column
			#scale the column values between 0 and 1 via min max scaling
			if sourceDF is None: 
				minVal = df[c].min()
				maxVal = df[c].max()
			else:#scale columns by max and min of source dataset
				#use column name of sourceDF cause last target variable column might have a different name
				c2 = sourceDF.columns[i]
				minVal = sourceDF[c2].min()
				maxVal = sourceDF[c2].max()
				
			quotient =maxVal-minVal
			if quotient == 0:
				myio.logWrite("A feature ("+c+") has constant values, but is included anyway in the ML process. Consider removing it from the input dataset.",myio.LOG_LEVEL_WARNING)
				continue
			df[c]=(df[c]-minVal)/(quotient)
	
	
	return df
def denormalize(minVal,maxVal,normalizedVal):
	quotient =maxVal-minVal
	res =(normalizedVal*quotient)+minVal
	return res


def extractTrainTestValues(df,numNonFeatCols=2):
	X=df.values[:,numNonFeatCols:-1]#don't include sampleid (index 0), timestamp (index 1),and the n2o (last column, index -1)
	y=df.values[:,-1] #only the n2o data, column -1 (last scolumn)
	X=np.asarray(X).astype(np.float32)
	y=np.asarray(y).astype(np.float32)
	return X,y			


#we take the sub indices pointing to sample/row indices in dfSubset, get the actual index in normDF
#via the sample id column, then extract as many fully-sized tensors as possible from normDF
#as feature, label (X,y) output
#shape of results is (samples,timesteps, features)
#temporalRes: temporeal resolution of dataset in minuts
def createTensorSets(normDF,dfSubset,dfSubsetIndices,tensorNumTimeSteps,temporalRes):
	
	indices =[]
	#samplesIds=dfSubset[SAMPLE_ID_EXTRA_COLUMN_NAME]
	
	tmpDF =dfSubset.iloc[dfSubsetIndices]
	samplesIds=tmpDF[SAMPLE_ID_EXTRA_COLUMN_NAME]
	
	for ptrIx in samplesIds: #use the subset indices as pointer		
		ptrIx = int(ptrIx)
		
		indices.append(ptrIx)
	
	#now that we have locations of sample subset in the full dataset
	#with consecutive timestamps sampels, extract tensors
	tensors,timestamps =createTensorSubset(normDF,tensorNumTimeSteps,temporalRes,indices)
	X=[] #feature matrix
	y=[] #label list
	for t in tensors:
		tX,ty =extractTrainTestValues(t,1) #only 1 non feature column (sample id), since timestamp column removed in the tensor extract proceed
		X.append(tX)
		#even thought there are labels for eahc rough, we will only have 1 label for a tensor
		#so we take the target variable of the most recent sample as label, to have an input tesnro
		#represent some historic values that changed 
		
		y.append(ty[-1])
	#gotta now convert datset into 3d numpy arrays with 1 dimension being sample id and 2nd being the feature, and 3rd being the time
	return np.asarray(X),np.asarray(y),timestamps
	
#removes any row from the dataframe normDF that doesn't have 'tensorNumberTimeSteps' concurrent 
#samples immeditly after the row at each time step.  For example, timesteps of 0,1,2,3,6,8,9,10,...
#using tensor length 3 means 2,3, and 6 don't have 3 readings after and so those rows are removed
def removeNonConcurrentTensorSamples(normDF,tensorNumberTimeSteps,tempResolution):
	
	tensors,timestamps = createTensors(normDF,tensorNumberTimeSteps,tempResolution,True)
	rowIndicesToRemove = []
	#find rows that don't have enough consecutive time steps
	for  i in range(len(tensors)):
		tensor = tensors[i]
		if len(tensor.index) != tensorNumberTimeSteps:
			rowIndicesToRemove.append(i)
	
	if len(rowIndicesToRemove)>0:
		#remove the rows
		normDF.drop(rowIndicesToRemove,inplace=True)
		
		#so that the gaps from removed column removed and index is re-aligned to consecutive numbers
		normDF.reset_index(drop=True, inplace=True)
				
			

