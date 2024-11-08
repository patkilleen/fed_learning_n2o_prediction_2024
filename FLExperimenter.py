import argparse
import copy
from tensorflow.keras.models import clone_model
from datetime import datetime, timedelta
import time
import math


import pandas as pd
import numpy as np

import traceback
import sys
import os
import random
import functools #for partial functions where return a function with assigned argument parameters

import experimenter
import model as mymodel
import dataset
import common as mycommon
import myio
DEBUGGING_FLAG=False#set to false when run on linux

	
	
try:	
	from keras_tuner import RandomSearch
	from keras_tuner import Objective
	import tensorflow as tf; 	
except ImportError as e:
	#only consider it error when not debugging and package import failed
	if not DEBUGGING_FLAG:
		print("Failed to import packages: "+str(e)+"\n"+str(traceback.format_exc()))
		exit()



STACKED_ENSEMBLE_NUMBER_OF_FOLDS=5
CENTRALIZED_LEARNING_STR="centralized learning"
LOCAL_AND_ENSEMBLE_LEARNING_STR="local learning and ensemble learning"
FEDERATED_LEARNING_STR="federated learning"

LOCAL_LEARNING_STR="local"
AVERAGE_ENSEMBLE_LEARNING_STR="avg-ensemble"
STACKED_ENSEMBLE_LEARNING_STR="stacked-ensemble"


PREFIX_LOCAL_ENSEMBLE_EXP_TYPE_LIST=[LOCAL_LEARNING_STR,AVERAGE_ENSEMBLE_LEARNING_STR,STACKED_ENSEMBLE_LEARNING_STR]			
METRICS_KEY_STRING_LIST=["R2","MSE","RMSE","MAPE"]
FOLD_RES_FILE_COL_STR="outer-fold"
ACTUAL_N2O_RES_FILE_COL_STR="actual_N2O"
PRED_INNER_FOLD_RES_FILE_COL_STR="predicted_inner_fold0"
PRED_AVG_FOLD_RES_FILE_COL_STR="predicted_inner_fold_average"
	
LOCAL_RES_MAP_COL_NAMES=[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME, \
						dataset.TIMESTAMP_COLUMN_NAME, \
						FOLD_RES_FILE_COL_STR, \
						ACTUAL_N2O_RES_FILE_COL_STR, \
						PRED_INNER_FOLD_RES_FILE_COL_STR, \
						PRED_AVG_FOLD_RES_FILE_COL_STR]
						
#federated learning client that has a local model, local training data, and local testing data
#Note that execution time is assuming instant network transmission, so network transmission delays
#are not consideredin this simulation when computing total execution time
#for the network communication cost. We assume upload and download costs are the same, and simply
#track the total number of bytes sent over the network.
class FLClient:

	#train_X: training data features/predictor variables
	#train_y: labels/target variable tied to each sample in train_X
	#test_X: testing data features/predictor variables
	#test_y: labels/target variable tied to each sample in test_X
	#testTimeStamps: list of timestamps associated to each sample in the test set
	#localTestSampleIds: the sample ids (index of the sample in the input dataset used to from train/test data) of each sample in the test set
	def __init__(self, id,train_X,train_y,test_X,test_y,testTimeStamps,localTestSampleIds):
		self.id = id
		self.train_X=train_X
		self.train_y=train_y
		self.test_X=test_X
		self.test_y=test_y	
				
		self.testTimeStamps = testTimeStamps
		self.localTestSampleIds=localTestSampleIds			
	
		self.modelWrapper = None #will store a model
		self.communicationCost=0 #tracks total communication costs
		self.execTime=0 #tracks total execution time
		
		if self.getNumberOfTrainSamples() == 0:
			raise Exception("Cannot have a client with an empty trianing dataset")
		
		if self.getNumberOfTestSamples() == 0:
			raise Exception("Cannot have a client with an empty test dataset")
			
		if len(testTimeStamps) == 0:
			raise Exception("Cannot have a client with an empty test timestamp list")
		
		if len(localTestSampleIds) == 0:
			raise Exception("Cannot have a client with an empty test sample id list")
			
	#updates the local model weights and train another round using train/test data of updated model
	#globalModelWrapper: model used replace the local model's weights with
	def FLModelUpdate(self,globalModelWrapper):
		
		startTime = time.time()
		#first client update?
		if self.modelWrapper is None:
			raise Exception("Cannot update local model for client "+str(self.id)+" with global model, since local model is None. Did you forget to call client.setModelWrapper before, to set the model?")			

		FLClient.copyModelWeights(globalModelWrapper.model,self.modelWrapper.model)
		
		#model weight update received from server
		self.communicationCost = self.communicationCost + mycommon.computeModelWeightMemoryUsage(globalModelWrapper.model)
	
		#evaluate the model by training using all the trainign data		
		self.modelWrapper.fit(self.train_X,self.train_y,self.test_X,self.test_y)
	
		endTime = time.time()
		
		ellapsedTime =endTime-startTime
		self.execTime=self.execTime+ellapsedTime
		#only model weights are sent back to server, so compute size of weights		 for total network communcation
		self.communicationCost = self.communicationCost + mycommon.computeModelWeightMemoryUsage(self.modelWrapper.model)
		
		
		
		return self.modelWrapper
	
	#simulates client receive final trained model from server
	#globalModelWrapper: the final global model trained via federated learning
	def FLReceiveFinalGlobalModel(self,globalModelWrapper):
		##model weight update received from server for the final global model
		self.communicationCost = self.communicationCost + mycommon.computeModelWeightMemoryUsage(globalModelWrapper.model)
		
		startTime = time.time()
		
		#update local model weights with the final global model weights
		FLClient.copyModelWeights(globalModelWrapper.model,self.modelWrapper.model)
		
		endTime = time.time()
		
		ellapsedTime =endTime-startTime
		self.execTime=self.execTime+ellapsedTime
	
	#evaluates the client's current local model using its local test data
	#unscaler: unscaling containing  minimum and maximum value of target variable to denormalize/unscale the data to compute mse and RMSE with proper units
	def clientEvaluateModel(self,unscaler):
		
		if self.modelWrapper is None:
			raise Exception("cannot evaluate predictions from uninitialized model for client "+str(self.id)+". Did you forget to call 'FLModelUpdate' first?") 
		
		startTime = time.time()		
		
		model,r2,mse,rmse,mape,pred = evaluateModel(self.test_X,self.test_y,self.modelWrapper,unscaler)
		
		endTime = time.time()
		ellapsedTime =endTime-startTime
		self.execTime=self.execTime+ellapsedTime
		
		return model,r2,mse,rmse,mape,pred
	
	#make local model predictions on samples found in X
	def predict(self,X):
		if self.modelWrapper is None:
			raise Exception("cannot make predictions from uninitialized model for client "+str(self.id)) 
		
	#overrides the model wrapper of the client with a new one
	def setModelWrapper(self,wrapper):
		
		if not isinstance(wrapper,mymodel.Model):
			raise Exception("Cannot set client's model. Expected object of type mymodel.Model but was type "+str(type(wrapper)))
			
		self.modelWrapper=wrapper
		
	#returns number of training samples
	def getNumberOfTrainSamples(self):
		return self.train_X.shape[0] 
	#returns number of training samples
	def getNumberOfTestSamples(self):
		return self.test_X.shape[0] 
	
	#returns the total  number of samples in the test and train set
	def getTotalNumberOfSamples(self):
		return self.getNumberOfTrainSamples() + self.getNumberOfTestSamples()
	
	#returns the number of features
	def getNumberOfFeatures(self):
		#number of feature will be determined by shape of dataset, but
		#this shape is different for tensor-based models
		#tabular models: shape = (number samples, number features)
		#tensor-based models: shape = (number samples, number time steps or tensors,number features)
		
		#tabluar model (sample x features)?
		if len(self.train_X.shape) == 2:
			return self.train_X.shape[1]
		elif len(self.train_X.shape) == 3: #tensor based model (samples x time step x features)?
			return self.train_X.shape[2]
			
		else:
			raise Exception("Cannot get the number of features from client. Unknown dataset format")
		
	#weight values are copied from a source model to a target model
	def copyModelWeights(srcModel,targetModel):
	
		#error check (same number of layers required)
		if len(srcModel.layers) != len(targetModel.layers):
			raise Exception("Cannot copy model weights to a target model, due to differing number of layers (source model = "+str(len(srcModel.layers))+" layers and target model = "+str(len(targetModel.layers))+" layers).")
			
		# iterate over each layer and set the targetModel's weights from the source model's weights		
		for layerIx in range(len(srcModel.layers)):
			srcLayer =srcModel.layers[layerIx]
			targetLayer = targetModel.layers[layerIx]
			
			
			srcWeights = srcLayer.get_weights()
			targetWeights = targetLayer.get_weights()
			
			#make sure there are the same number of weights between the same layers of both models
			if len(srcWeights) != len(targetWeights):						
				raise Exception("Cannot copy model weights over to target model, since one of the layers has a differing number of weights.")
			
			for wix in range(len(srcWeights)):
				if not mycommon.shapesEqual(srcWeights[wix].shape,targetWeights[wix].shape):
					raise Exception("Cannot copy model weights over to target model, since one of the layers has a differing number of weights.")
			
			#make sure its a deep copy so upper layer logic doesn't affect this local copy
			srcWeights = copy.deepcopy(srcWeights) 
						
			targetLayer.set_weights(srcWeights)
	#returns a string summarizing the client properties
	def summaryStr(self):		
		resStr = "FLClient: {client id: "+str(self.id)+","
		resStr = resStr +" number samples (train/test): "+str(self.getTotalNumberOfSamples())+" ("+str(self.getNumberOfTrainSamples())+","+str(self.getNumberOfTestSamples())+"),"				
		resStr = resStr+ " 1st test sample id: "+str(self.localTestSampleIds[0])+","
		resStr = resStr +" last test sample id: "+str(self.localTestSampleIds[-1])+","		
		resStr = resStr +" 1st test time stamp: "+str(self.testTimeStamps[0])+","
		resStr = resStr +" last test time stamp: "+str(self.testTimeStamps[-1])+"}\n"		
		return resStr
	
#provides simple API over list of clients and their datasets
class FLClientSet:

	#clients: list of clients
	#allowSingleClientFlag: when True, means 1 client is acceptable, False means 1 client rasises Exception
	#						unless there is a good reason to set allowSingleClientFlag to True, it shoud be kept false
	def __init__(self, clients, allowSingleClientFlag=False):
	
		#integrity checks
		
		#at least 2 clients required for federated learning
		if len(clients)<=1:
			#exception is only raise in the default case
			if not allowSingleClientFlag:
				raise Exception("Cannot create a set of clients for federated learning. At least two clients are required, but "+str(len(clients))+" were provided.")
		
		#all clients must share the same number of features and same number of tensor-length/time-step for tensor-based models
		#their number of samples can differ
		expectedXShape = clients[0].train_X.shape
		expectedyShape = clients[0].train_y.shape
		
		#flag to indicate inconsitencies in data strcutures between clients (e.g., True when different number of features between 2 clients)
		dataStrutIssueFlag=False
		
		for cix in range(len(clients)):
			#can skip comparing 1st client to itself
			if cix == 0:
				continue
				
			client = clients[cix]
			#[0] so that number of samples, 0th dimension can differ and still consider the shape equal
			if not mycommon.shapesEqual(expectedXShape,client.train_X.shape,dimIgnoreList=[0]): 
				dataStrutIssueFlag=True
				break
			if not mycommon.shapesEqual(expectedXShape,client.test_X.shape,dimIgnoreList=[0]):			
				dataStrutIssueFlag=True
				break
			if not mycommon.shapesEqual(expectedyShape,client.train_y.shape,dimIgnoreList=[0]):						
				dataStrutIssueFlag=True
				break
			if not mycommon.shapesEqual(expectedyShape,client.test_y.shape,dimIgnoreList=[0]):						
				dataStrutIssueFlag=True
				break		
				
	
		if dataStrutIssueFlag:
			raise Exception("Cannot create a set of clients. The clients do not share the same dataset structure.")
			
		self.clients = clients
		
		self.totalNumTrainSamples = 0
		self.totalNumSamples=0
		#compute the total number of training samples among all clients
		for client in clients:							
			self.totalNumTrainSamples = self.totalNumTrainSamples +client.getNumberOfTrainSamples()
			self.totalNumSamples = self.totalNumSamples + client.getTotalNumberOfSamples()
			
		self.numFeatures = clients[0].getNumberOfFeatures()
	
	#returns iterator over list of clients
	def  __iter__(self):
		#return an iterator over the clients in this set
		return iter(self.clients)
	
	#returns a client at a given index
	def getClient(self, ix):
		if ix < 0 or ix >= self.getNumberOfClients():
			raise Exception("Cannot get client due to Index out of bounds exception. Expected index between 0 and "+str(self.getNumberOfClients())+" but was "+str(ix)+".")
			
		return self.clients[ix]
		
	#returns the number of featurs each clients have
	def getNumberOfFeatures(self):
		return self.numFeatures
		
	#returns the number of clients in the set
	def getNumberOfClients(self):
		return len(self.clients)
	
	#returns the total number of training samples over all clients
	def getTotalNumberOfTrainSamples(self):
		return self.totalNumTrainSamples
		
	#returns the total number of samples in both training and testing datasets over all client
	def getTotalNumberOfSamples(self):
		return self.totalNumSamples
		
#run all experiments in the configuration file configInputPath, and report results in outputDir
#the configuration file will indicate whether centralize, local learning + ensemble learning, or federated learning is run
#in the case of local learning + ensemble learning experiment types, 3 types of experiments will be run
#for each experiment, local leanring, average ensemble, and stacked ensemble
#configInputPath: path to configuration file that controls the experiments
#outputDir: directory to output result files
def runFLExperiments(configInputPath,outputDir):
	
	#read the experiment configuration file 
	configDF = pd.read_csv(configInputPath, sep=",")
		
	#make sure all needed columns are present
	myio.configFileIntegrityCheck(configDF,myio.FEDERATED_LEARNING_CONFIG_FILE_TYPE)
	
	numExperiments = len(configDF.index)
	
	#create the files and subdirectories to store results
	logFilePath,resFilePath,expDir,outDirPath=myio.setupOutputDirectory(configInputPath,outputDir, numExperiments)
		
	myio.openLogFile(logFilePath)
	
	myio.logWrite("Starting FL experiments from experiment file "+configInputPath,myio.LOG_LEVEL_INFO)
	
	#check if a GPU-accelerated deep learning is available
	
	#list the gpu devices detected by tensor lflow
	gpus = tf.config.list_physical_devices('GPU')
	
	if len(gpus)==0:
		myio.logWrite("Running deep learning models on the CPU, since the GPU was not detected",myio.LOG_LEVEL_WARNING)
		processingDevice = "CPU"
	else:
		processingDevice = "GPU"

	#open the result fil
	resFile = open(resFilePath,"a")
		
	#create head	
	resFile.write("experiment id,processing device,algorithm,number of input tensors,FL aggregator,client heterogeneity, year,chamber,temporal resolution,number of features,number of global instances,seed,iteration,experiment type,fold,number of local instances,clientid,R2,MSE,RMSE,MAPE,execution time(s),network communication cost (KB)\n")
	
	#iteratre over each experiments
	for eix in range(numExperiments):
		#go to the next experiment if an experiment fails
		try:
			
			
			experimentStartTime=time.time()			
			myio.flushLogFile()
			
			inputDatasetPath=configDF[myio.INPUT_DATASET_PATH_KEY][eix]
			
			selectedFeaturePath=configDF[myio.SELECTED_SENSOR_PATH_KEY][eix]
			
			if not os.path.isfile(inputDatasetPath):
				myio.logWrite("Could not find input dataset "+str(inputDatasetPath)+" for experiment "+str(eix)+". Skipping this experiment.",myio.LOG_LEVEL_ERROR)
				continue
			
			#read the dataset into memory 
			inDF,selectedFeatures = dataset.readDataset(inputDatasetPath,selectedFeaturePath)
			
			nFeatures = len(selectedFeatures)
			
			normalizeDataFlag = configDF[myio.APPLY_MIN_MAX_SCALING_FLAG_KEY][eix]
			
			year = configDF[myio.YEAR_KEY][eix]
			chamber =configDF[myio.CHAMBER_KEY][eix]
			tempResolution =configDF[myio.TEMPORAL_RESOLUTION_KEY][eix]
			
			algName = configDF[myio.ALGORITHM_KEY][eix]
			
			if mymodel.isDeepLearningModel(algName):
				
				actualProcessingDevice=processingDevice #deep learning models use  GPU acceleration if available
			else:
				actualProcessingDevice="CPU"#basic ML models use the CPU
				
			tensorNumberTimeSteps = configDF[myio.INPUT_TENSOR_NUMBER_TIME_STEPS_KEY][eix]
						
	
			numIterations=configDF[myio.ITERATIONS_KEY][eix]
			
			expType = configDF[myio.FEDERATED_LEARNING_EXPERIMENT_TYPE_KEY][eix]
			
			numClients=configDF[myio.FEDERATED_LEARNING_NUMBER_OF_CLIENTS_KEY][eix]
			
			#for experiments with at least 2 clients, create output directory for each client of the experiment 
			#centralzied experiments will just have the usual output files without partitioned into clients
			if expType != CENTRALIZED_LEARNING_STR:				
				myio.setupFedLearnClientOutputDirectory(expDir,eix,numClients)
			
			#reset rng seed
			rngSeed=configDF[myio.RNG_SEED_KEY][eix]
			random.seed(rngSeed)
			
			
			clientDataHete = configDF[myio.FEDERATED_LEARNING_CLIENT_HETEROGENEITY_KEY][eix] #non-IID or IID
			#determine if its IID or NON-IID
			if  clientDataHete == "IID":
				clientIIDFlag=True
			elif clientDataHete == "non-IID":
				clientIIDFlag=False
			else:
				raise Exception("Unknown client heterogeneity. Expected 'IID' or 'non-IID' but was "+clientDataHete)
				

			learnRateOverride=configDF[myio.FEDERATED_LEARNING_LEARNING_RATE_OVERRIDE_KEY][eix]
			epochsOverride = configDF[myio.FEDERATED_LEARNING_EPOCH_OVERRIDE_KEY][eix]
			batchSizeOverride=configDF[myio.FEDERATED_LEARNING_BATCH_SIZE_OVERRIDE_KEY][eix]
			
			outerCVType=configDF[myio.FEDERATED_LEARNING_CROSS_VALIDATION_TYPE_KEY][eix]
			blockCVClusterSize=configDF[myio.FEDERATED_LEARNING_BLOCK_CV_CLUSTER_SIZE_KEY][eix]
			
			numberFLCVFolds =configDF[myio.FEDERATED_LEARNING_NUMBER_OF_FOLDS_KEY][eix]
			
			#only deep learling models can have federated learning applied
			if expType==FEDERATED_LEARNING_STR and not mymodel.isDeepLearningModel(algName):
		
				raise Exception("Federated learning in this project does not support non-deep-learning models ("+algName+"). Only centralized learning, local learning, and ensemble learning are supported.")
				
	
		
			#check taht algorithm has appropriate tensor length definition
			if tensorNumberTimeSteps > 1 and not mymodel.isModelWithInputTensors(algName):
				raise Exception("Configuration file issue. Algorithm that expects no multi-dimensional input samples in time had tensor length > 1")
				
			aggregatorFunc = configDF[myio.FEDERATED_LEARNING_AGGREGATOR_KEY][eix]
					
			originalN2OData=inDF.values[:,-1]
							
			#compute min and max n2o to  unscale the normalized
			#emissiosn for results presentation
			maxN2O = max(originalN2OData)
			minN2O = min(originalN2OData)
				
			#used for model evaluation when computing metrics that need to be same scale and units as target variable
			unscaler = dataset.DataUnscaler(minN2O,maxN2O,normalizeDataFlag)
			
			#only normlize dataset using min-max scaling if user hasn't already pre-normalized the dataset
			if normalizeDataFlag:
				#normalize the local dataset 
				normDF = dataset.normalize(inDF)
			else:
				normDF =inDF #data was already normalized/scaled in the file, don't normalize
			
			
			#create a strign of feature for user-friendly logging purposes
			featuresStr="["
			for featureName in selectedFeatures:				
				featuresStr = featuresStr + ",\n\t"+featureName
			featuresStr=featuresStr+"]"
			
			nSamples = len(normDF.index) 
			
			myio.logWrite("Experiment "+str(eix)+"/"+str(numExperiments-1)+":  running a "+str(numClients)+"-client FL ("+clientDataHete+" split + "+expType+") experiment with algorithm "+algName+" using dataset with "+str(nSamples)+" samples and "+str(nFeatures)+" features:\n ("+featuresStr+")." ,myio.LOG_LEVEL_INFO)
			
			
			#parition original normalized dataset into local dataset for each client using IID or non-IID partitioning
			clientDFList,trimmedClientDFList,newTempRes =  partitionClientData(normDF,tempResolution, numClients,clientIIDFlag,tensorNumberTimeSteps)
			
			#used for tracking average overall performance for logging approx. results
			globalAvgMetricMaps=createGlobalAvgMetricMaps(expType,numClients)		
			metricCounter=0
			
			#create the map to store client results			
			resultMaps = createClientResultMaps(numClients,expType)					
						
			
			#run many iterations of cross-validaiton 
			for itIx in range(numIterations):
				
				#each client has its own output file for each iteration, so clear the result buffers
				clearClientResultMaps(resultMaps)
				
					
	
				#cross-validation  folds creation to iterate over all 
				#creates trian/test partitions for every client in cross-validation fasion, where result of every fold iterated
				#so a set of train/test data partitions will be done iteratively (so this is like multiple iterations of the entire FL process)				
				#WHERE a single loop iteration has all clients perform one fold of cross-validation
				for clientSet,foldIx in crossValClientSetCreation(clientDFList,trimmedClientDFList,newTempRes,algName,nFeatures, tensorNumberTimeSteps,numClients,outerCVType,blockCVClusterSize,numberFLCVFolds):

					myio.logWrite("About to evaluate each client's model for iteration "+str(itIx)+" and fold "+str(foldIx) ,myio.LOG_LEVEL_DEBUG)
					
					nFeatures=clientSet.getNumberOfFeatures()
					#when tensor length is positive, then input samples are  nFeatures x tensorNumberTimeSteps
					#for basic ML and dnn, NO dimension in time, so just nFeatures x  1 =
					if tensorNumberTimeSteps>1:
						inputShape=[nFeatures,tensorNumberTimeSteps]
					else:
						inputShape=[nFeatures]
					
					totalNSamples = clientSet.getTotalNumberOfSamples()
					
					#first we create result file output row CSV prefix				
					resFileRowPrefix = str(eix)+","#experiment id
					resFileRowPrefix = resFileRowPrefix+str(actualProcessingDevice)+","#processing device
					resFileRowPrefix = resFileRowPrefix+str(algName)+","#algorithm
					resFileRowPrefix = resFileRowPrefix+str(tensorNumberTimeSteps)+","#number of input tesnors
					resFileRowPrefix = resFileRowPrefix+str(aggregatorFunc)+","#fl aggregator algorithm (e.g., FedAvg)
					resFileRowPrefix = resFileRowPrefix+str(clientDataHete)+","#fl client data split heterogeneity
					resFileRowPrefix = resFileRowPrefix+str(year)+","#year
					resFileRowPrefix = resFileRowPrefix+str(chamber)+","#chamber
					resFileRowPrefix = resFileRowPrefix+str(newTempRes)+","#temporal resolution
					resFileRowPrefix = resFileRowPrefix+str(nFeatures)+","#number of features
					resFileRowPrefix = resFileRowPrefix+str(totalNSamples)+","#number of total samples in pre-partitioned dataset					
					resFileRowPrefix = resFileRowPrefix+str(rngSeed)+","#rng seed
					resFileRowPrefix = resFileRowPrefix+str(itIx)+","#iteration
					
					#there are a few prefixes for local and ensbmle learning
					#so create a map to make choosing the proper prefix simple
					if expType==LOCAL_AND_ENSEMBLE_LEARNING_STR:
						#will have 3 different prefixes for this type of experiment, so 
						#make a map for simple lookup
						localEnsemblePrefixMap={}
						
						for prefixExpType in PREFIX_LOCAL_ENSEMBLE_EXP_TYPE_LIST:																
							localEnsemblePrefixMap[prefixExpType]=resFileRowPrefix+prefixExpType+","+str(foldIx)+","								
					else:
						resFileRowPrefix = resFileRowPrefix+expType+","#learning type (centralized learning, local learning				
						resFileRowPrefix = resFileRowPrefix+str(foldIx)+","#fold
					
					
					##################
					# CENTRALIZED LEARNING
					####################
					if expType==CENTRALIZED_LEARNING_STR:
						#run model evaluation experiments
						r2,mse,rmse,mape,execTime,netComCost,pred,tmpCentralizedClientSet= centralizeLearningEval(clientSet,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler)
						r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,predList=[r2],[mse],[rmse],[mape],[execTime],[netComCost],[pred]#use [] notation to put metrics in 1d list needed  
						
						#track sum of metrics
						addToGlobalAvgMetricMaps(globalAvgMetricMaps,expType,r2List,mseList,rmseList,mapeList)
						
						
						#output the results to file						
						resFileOutput = parseResFileRowOutString(resFileRowPrefix,tmpCentralizedClientSet,r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,clientIDOverride=True) #use 1 as num clients since 1 cnetralized result
						resFile.write(resFileOutput)
						resFile.flush()
												
						#store results for later use
						populateClientResultMaps(resultMaps[expType],tmpCentralizedClientSet,predList,foldIx,unscaler)											
						
					##################
					# LOCAL LEARNING AND ENSEMBLE LEARNING
					####################
					elif expType==LOCAL_AND_ENSEMBLE_LEARNING_STR:
						
						#local learning and ensembel learning all done in one batch after training, since
						#they all use the same models. The evaluation process is just different
						
						
						#run model evaluation experiments
						for learningTypeStr in PREFIX_LOCAL_ENSEMBLE_EXP_TYPE_LIST:
							if learningTypeStr==LOCAL_LEARNING_STR:
								#local learning
								r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,predList = localLearningEval(clientSet,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler)																														
							elif learningTypeStr==AVERAGE_ENSEMBLE_LEARNING_STR:
								#average ensemble model sharing					
								r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,predList = averageEnsembleLearningEval(clientSet,unscaler)																								
							elif learningTypeStr==STACKED_ENSEMBLE_LEARNING_STR:
								#stacked ensemble model sharing
								r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,predList = stackedEnsembleLearningEval(clientSet,STACKED_ENSEMBLE_NUMBER_OF_FOLDS,unscaler)
							else:
								#this should never happen
								raise Exception ("Internal design error for local and ensemble learnign logic. Unknown learnin type '"+learningTypeStr+"'.")
																					
							#track sum of metrics
							addToGlobalAvgMetricMaps(globalAvgMetricMaps,learningTypeStr,r2List,mseList,rmseList,mapeList)
							
							#write reulsts to file
							resFileOutput = parseResFileRowOutString(localEnsemblePrefixMap[learningTypeStr],clientSet,r2List,mseList,rmseList,mapeList,execTimeList,netComCostList) 
							resFile.write(resFileOutput)
							
							#store results for later use							
							populateClientResultMaps(resultMaps[learningTypeStr],clientSet,predList,foldIx,unscaler)												
																													
						resFile.flush()
						
					##################
					# FEDERATED LEARNING
					####################
					elif expType==FEDERATED_LEARNING_STR:
						
						numClientsPerRound = configDF[myio.FEDERATED_LEARNING_NUM_SEL_CLIENTS_PER_ROUND_KEY][eix]
						numRounds=configDF[myio.FEDERATED_LEARNING_NUMBER_OF_ROUNDS_KEY][eix]
						
						
						
						#count total number  of samples in training set
						#and track number of samples in each local training dataset
						totalNumTrainSamples=clientSet.getTotalNumberOfTrainSamples()
		
						#run model evaluation experiments
						r2List,mseList,rmseList,mapeList,execTimeList,netComCostList,predList  = federatedLearningEval(totalNumTrainSamples,clientSet,year,algName,aggregatorFunc,numRounds,numClientsPerRound,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler)
						
						#track sum of metrics
						addToGlobalAvgMetricMaps(globalAvgMetricMaps,expType,r2List,mseList,rmseList,mapeList)
							
						#output results to result file
						resFileOutput = parseResFileRowOutString(resFileRowPrefix,clientSet,r2List,mseList,rmseList,mapeList,execTimeList,netComCostList) 
						resFile.write(resFileOutput)
						resFile.flush()
						
						#store results for later use							
						populateClientResultMaps(resultMaps[expType],clientSet,predList,foldIx,unscaler)
						
					else:
						#this shoudl never happen
						raise Exception("Internal design error in federated learning experimenter. Unknown experiment type: '"+str(expType)+"'")
						
					#for tracking summ of evaluation metrics  and taking average, count number of summations performed					
					metricCounter=metricCounter+1
									
					foldIx=foldIx+1
				#end cross-validation over clients loop	
				
				#create a prediction file for each client				
				createPredictionResFile(resultMaps,expDir,eix,itIx,expType)
				
			#end iterations	
					
			
			#take average performance over all clients and parse to a userfriendly string for logging
			globalPerformanceStr = parseAvgExpPerformanceStr(globalAvgMetricMaps,metricCounter)
			
			experimentEndTime = time.time()
			totalExperimentTime = experimentEndTime-experimentStartTime
			
			myio.logWrite("FL experiment "+str(eix)+"  ("+clientDataHete+" split + "+expType+") finished after "+str(round(totalExperimentTime,2))+" seconds for algorithm "+algName+globalPerformanceStr,myio.LOG_LEVEL_INFO)
			
		#end try experiment
		except Exception as e:
			myio.logWrite("Aborting experiment "+str(eix)+" due to an exception: "+str(e)+"\n"+str(traceback.format_exc()),myio.LOG_LEVEL_ERROR)
			continue
		#end try catch experiments
	#end experiments
	myio.logWrite("Finished FL experiments from experiment file "+configInputPath+". Output files written to "+outDirPath+".",myio.LOG_LEVEL_INFO)
	
	myio.closeLogFile()
	resFile.close()

#split normDF into numClients blocks that are IID when clientIIDFlag=True and are non-IID when clientIIDFlag=False
		#then each dataframe is considered a new dataset with a new resolution of tempRes/numClients
		#e.g., 30 min resolution split in 4 clients would make every client have 4 times fewer samples
		#so the resulting subsets would each be 30 * 4 min = 120 min = 2h resolution datasets
		#so will create numClients dataframes from dataset
#normDF: normalized dataframe to be split into multiple partitions for each client, where each parition will have local sample ids
#tempRes: temporal resolution (minutes) of the input dataset
#numClients: number of clients to parition dataset into
#clientIIDFlag: flag when true indicating paritions will be IID/homgenous, and False means paritions will be non-IID/heterogenous
#tensorNumberTimeSteps: 0 when a tabular based model  (e.g., DNN, RF)without tensor input, and number of tensors (> 1) for tensor-based models (.e.g LSTM, CNN)
def partitionClientData(normDF,tempRes,numClients,clientIIDFlag,tensorNumberTimeSteps):

	tempResMins = tempRes
	tempResHours= tempRes/60.0
	#sort by timestamp (from early to later readings)
	normDF = normDF.sort_values(by=['timestamp'])
	
	df =normDF.copy(deep=True)
	
	#convert the timestamp column to a datetime format
	df['timestamp'] = pd.to_datetime(normDF['timestamp'])

	#set the index of dataframe to the timestamp
	df.set_index('timestamp',inplace=True)
	
	#list of index sets (for each client) that indicate what sample index belongs to what client
	clientSampleIndexSet=[]	
	for i in range(numClients):
		clientSampleIndexSet.append([])
	
	nSamples = len(df.index)
	#IID/homogeneous paritioning?
	if  clientIIDFlag:
		
		#partition samples such that every other sample from modulo belongs to one of clients
		#e.g., for 3 client system, sample at time 0 = client0,sample at time 1 = client1, sample at time 2 = client2, sample at time 3= client0, sample at time 4 = client1,...
		#create list of all row indices of normDF that belong to each client
		#to address missing values, we iterate over the timestamps at the given temporal reoslution, even if a row is missing
		#for that timestep, to avoid offseting the phase of timesteps of clients from missing values
		
		#define start and end to locate samples within dataset
		wStart = df.index[0] 
		wEnd = wStart+ timedelta(hours=tempResHours)
		sampleId=0
		timeStepCounter = 0
		
		#iterate over the datasets until all sample indices have been partitioned/assigned to a client
		while sampleId < nSamples:
		
			#index of client
			cix = timeStepCounter%numClients
			
			windowFilter = (df.index>= wStart) & (df.index< wEnd)
			
			#get the samples in window (there should be at most 1 sample for non-duplicate timestamp entries)
			subset = df.loc[windowFilter]
			
			#a sample exists at that time stamp?
			if len(subset.index)>0:
				
				#get the list of indices for client cix and add the index of this sample
				clientSampleIndices=clientSampleIndexSet[cix]				
				clientSampleIndices.append(sampleId) 
				sampleId = sampleId +1 #sample id only increment once we found a non-emtpy sample
				
				#more than one sample, which should happen with a proper dataset?
				if  len(subset.index)>1:
					myio.logWrite("During partitioning of IID client dataset, rows with duplicate timestamps found or dataset is higher resolution than "+str(tempResHours)+" hours at around row "+str(sampleId)+". Ignoring the additional samples.",myio.LOG_LEVEL_ERROR)
			
			timeStepCounter = timeStepCounter +1
						
			wStart = wEnd
			wEnd = wStart+ timedelta(hours=tempResHours)
											
		#the tempoeral resolution of each client's dataset is less fine-grained / decrease (more clients means larger gaps between reading)
		#by definition of this IID paritioning
		#e.g., a 30 minute resolution dataset that's split into 4 clients this way would have 120  min temporal resoultion (30 * 4) for each client
		newTempRes = tempResMins*numClients
	else: #non-IID/heterogenous paritionin
		#the clients will each have their own part of the season. 
		#e.g., a 4 client system would mean client 0 has early season readings,
		#client 1 has mid season, client 2 has late season, and client 3 has end of season
		#last client may have fewer samples if the dataset is not perfectly divisible by
		#number of clients
		samplesPerClient = math.ceil(nSamples/numClients)
		
			
		#iteration over each row index and tie it to the appropriate client
		for i in range(nSamples):
			cix = int(math.floor(i / samplesPerClient))
			clientSampleIndices=clientSampleIndexSet[cix]
			clientSampleIndices.append(i)
		
		#in this paritioning scheme, clients retain their temporal resolution
		newTempRes = tempResMins
		
	
	#partition the dataframe into subsets for each clients based on the list of row indices tied to each client
	clientDFList = []
	for clientSampleIndices  in clientSampleIndexSet:
		clientDF = normDF.iloc[clientSampleIndices] #we use normDF here and not df to include the timestamp column (df uses it for index)
		clientDF.reset_index(drop=True, inplace=True)
		#print("client sample indices: "+str(clientSampleIndices))
		#print("client DF: "+str(clientDF))
		#make sure the new dataframe subset is a deep copy, not just a readn-only slice
		clientDF = clientDF.copy(deep=True)
		
		clientDFList.append(clientDF)
				
	
	#make sure the data aren't mixed between clients (or duplicated) via sample id
	if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
		sampleIdMap={}
		for clientDF in clientDFList:
			for sampleId in clientDF[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME]:
				#duplicate sample or sample shared between to clients?
				if sampleId in sampleIdMap:
					myio.logWrite("Client dataset paritioning sample "+str(sampleId)+" duplicated or shared between to clients",myio.LOG_LEVEL_ERROR)
				else:
					sampleIdMap[sampleId]=None	
	
	
	#to avoid false alarm  wanring from pandas (were working with a deep copy, so its fine)
	defOption = pd.options.mode.chained_assignment
	pd.options.mode.chained_assignment = None  # default='warn'
	
	#update the sample ids of each dataset to make them local
	for clientDF in clientDFList:	
		#reset index so that its no longer a subset of global dataset. now its local
		#clientDF.reset_index(drop=True, inplace=True)		
		#make sure to keep the timestamp as index over dataframe
		#set the index of dataframe to the timestamp
		#clientDF.set_index('timestamp',inplace=True)
		globalClientSampleIds=clientDF[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME]
		#make sure the sample ids in the partition are updated to be local, not global
		locaSampleIds = mycommon.rankUniqueNumberList(globalClientSampleIds)
		
		#iterate over the new indices of the sample id to update the sample id of dataset to
		#make the dataset contain sample id relative to local dataset, not global dataset
		for i in range(len(locaSampleIds)):
			clientDF[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME][i]=locaSampleIds[i]
			
				
	#undo warning supression
	pd.options.mode.chained_assignment = defOption
	
	trimmedClientDFList = []
	
	
	
	#trim each dataset to make sure missing values addressed for the tensor-based models 
	#that require consecutive readings in a time block
	for cix in range(len(clientDFList)):
		clientDF=clientDFList[cix]
		
		#myio.logWrite("First 3 samples of normalized dataset for client "+str(cix)+": "+str(clientDF.values[0:3,:]),myio.LOG_LEVEL_DEBUG)
		#myio.logWrite("Last sample of normalized dataset for client "+str(cix)+": "+str(clientDF.values[-1,:]),myio.LOG_LEVEL_DEBUG)
		

		#tensor based model?
		if tensorNumberTimeSteps > 0:
			#keep track of origional dataset before we remove reading that would lead to incomplete tensors (missing concecutive readigns)
			trimmedClientDF = clientDF.copy(deep=True)  
			
			#remove rows were there is missing consecutive readings after to form a tensor
			#the sample ids don't change so that using sampleid as pointer to row in norm DF still possible
			dataset.removeNonConcurrentTensorSamples(trimmedClientDF,tensorNumberTimeSteps,newTempRes) 
			trimmedClientDFList.append(trimmedClientDF)
		else:
			trimmedClientDFList.append(clientDF)
	
		
	#printing debug log messages?
	if myio.GLOBAL_LOG_LEVEL==myio.LOG_LEVEL_DEBUG:
		#make a log entry for debuging purposes
		if clientIIDFlag:
			iidStr ="IID"
		else:
			iidStr ="non-IID"
		
		#parse a string to capture every client's number of sampels
		clientNSamplesStr=". Client number of samples = "
		for clientDF in trimmedClientDFList:
			clientNSamples = len(clientDF.index)
			clientNSamplesStr = clientNSamplesStr + str(clientNSamples) + ", "
		myio.logWrite("Partitionned the dataset into "+str(numClients)+" "+iidStr+" partitions, resulting in local datasets of "+str(newTempRes)+" min temporal resolution."+clientNSamplesStr ,myio.LOG_LEVEL_DEBUG)
		
	return clientDFList,trimmedClientDFList,newTempRes
#clientDFList: list of dataframes that represent local client datasets
#trimmedClientDFList: list of dataframes with the non-full tensor blocks removed that represent local client datasets
#tempRes: temporal resolution of each client's dataset
#algName: algorithm name
#nFeatures: number of features
#tensorNumberTimeSteps: number of time steps in an input tensor for tensor-based models (e.g., LSTM and CNN)
#numClients: number of clients to partition the dataset into
#crossValType: type of cross validation to use ('random CV' or 'block CV')
#blockCVClusterSize: used when crossValType is equal 'block CV'. Determines sizes of blockS/clusters (in hour units) for clustered sampling
#numFolds: number of folds in the cross-validation
def crossValClientSetCreation(clientDFList,trimmedClientDFList,tempRes,algName,nFeatures,tensorNumberTimeSteps,numClients,crossValType,blockCVClusterSize,numFolds):
	
			
	myio.logWrite("Performing cross-validation for each client and creatign a client set at each iteration..." ,myio.LOG_LEVEL_DEBUG)
				
			
	#this matrix will store all the clients for each client set and fold
	#row = client ix
	#col = fold ix
	#e.g., clientMatrix[2,3] would be client 2's dataset split for fold 3 of the cross-validation process
	clientMatrix=[]
	#iterate over each client's dataset
	for cix in range(len(clientDFList)):
		clientDF=clientDFList[cix]
		trimmedClientDF = trimmedClientDFList[cix]
	
		
		#list of client objects that represent all cross-validation folds for a single client
		clientRow=[]
		clientMatrix.append(clientRow)
		
		#get sample ids that will be shuffled
		shuffledIndices = experimenter.createRowIndexOrdering(crossValType,trimmedClientDF,blockCVClusterSize,numFolds,tensorNumberTimeSteps)				
		finalShuffleOrderList = experimenter.populateCVOrderingList(crossValType,shuffledIndices)
			
		if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
			#map that tracks frequency a sample was part of testing
			outerSampleTestCountMap={}
			outerSampleTrainCountMap={}
		
		foldIx=0
		#do an entire cross-validation process over a client's dataset to store each of the folds in client object to be placed in the matrix
		for trainIxs,testIxs in experimenter.crossValSplit(crossValType,trimmedClientDF,numFolds,shuffledIndices):		
			
			#extract train/test datasets			
			trainDF = trimmedClientDF.iloc[trainIxs]			
			testDF = trimmedClientDF.iloc[testIxs]
					
			#the sample id (index of samples), with respect to client's local dataset
			localTestSampleIds=testDF[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME]
			
			#model has tensors?
			if mymodel.isModelWithInputTensors(algName):
				train_X,train_y,trainTimeStamps=dataset.createTensorSets(clientDF,trimmedClientDF,trainIxs,tensorNumberTimeSteps,tempRes) 
				test_X,test_y,testTimeStamps=dataset.createTensorSets(clientDF,trimmedClientDF,testIxs,tensorNumberTimeSteps,tempRes) 
				#make sure dataset has shape [samples,timesteps,features]
				train_X = train_X.reshape((train_X.shape[0], tensorNumberTimeSteps, nFeatures))
				test_X = test_X.reshape((test_X.shape[0], tensorNumberTimeSteps, nFeatures))			
				
				
				clientTestTimeStamps=testTimeStamps				
				
			else:
				
			
				#extract the features X from target  variable y									
				train_X,train_y = dataset.extractTrainTestValues(trainDF)						 
				test_X,test_y = dataset.extractTrainTestValues(testDF)
				
				clientTestTimeStamps=testDF[dataset.TIMESTAMP_COLUMN_NAME]
			
			#np.copy to  make deep copies since the crossValSplit logic frees fold dataset memory between fold-iterations
			#and we want to keep a matrix of for an entire cross-validation process later on
			client = FLClient(cix,np.copy(train_X),np.copy(train_y),np.copy(test_X),np.copy(test_y),np.array(clientTestTimeStamps),np.array(localTestSampleIds)) 			
	
			clientRow.append(client)
			foldIx = foldIx+1
			#make sure test and train data aren't mixed and that samples aren't missed/forgotten in one of fold splits
			if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
				
				for testIx in testIxs:
					if testIx in outerSampleTestCountMap:
						myio.logWrite("Inner client dataset sample was part of outter test set more than once in client "+str(cix),myio.LOG_LEVEL_ERROR)
					else:
						outerSampleTestCountMap[testIx]=1
					for trainIx in trainIxs:					
						if testIx==trainIx:
							myio.logWrite("Inner client dataset  Outter CV train-test data contamination in client "+str(cix),myio.LOG_LEVEL_ERROR)
				for trainIx in trainIxs:
					if trainIx in outerSampleTrainCountMap:
						outerSampleTrainCountMap[trainIx] = outerSampleTrainCountMap[trainIx] +1
					else:
						outerSampleTrainCountMap[trainIx]=1
			
			
			
		if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:			
			#samples part of test set exactly once
			for testIx in outerSampleTestCountMap:				
				if outerSampleTestCountMap[testIx]!=1:
					myio.logWrite("One of the samples in the inner CV was part of the test set more than once ("+str(outerSampleTestCountMap[testIx])+" times)",myio.LOG_LEVEL_ERROR)
			#samples part of train set nubmber of folds -1
			for trainIx in outerSampleTrainCountMap:				
				if outerSampleTrainCountMap[trainIx]!=foldIx - 1:
					myio.logWrite("One of the samples in the inner CV was not part of the train set number of folds ("+str(foldIx)+") -1 times",myio.LOG_LEVEL_ERROR)
					
		
	#now we proceed to act like a generator where we slice the matrix row by row
	clients = []
	#iterate over every fold
	for foldIx in range(numFolds):
		clients.clear()
		#iterate over all clients for a given fold
		for cix in range(numClients):			
			clients.append(clientMatrix[cix][foldIx])
		
		clientSet = FLClientSet(clients) #all clients with local train/test data configured for fold foldIx
		
		#printing debug log messages?
		if myio.GLOBAL_LOG_LEVEL==myio.LOG_LEVEL_DEBUG:
		
			#parse a string to capture every client's number of train and test samples
			clientNSamplesStr=". Client summary info for fold "+str(foldIx)+" = "
			for cTmp in clientSet:								
				clientNSamplesStr = clientNSamplesStr +cTmp.summaryStr()+","
			myio.logWrite(clientNSamplesStr ,myio.LOG_LEVEL_DEBUG)		
		
		yield clientSet,foldIx
		
			
#(specified by outerCVType), and do so 'numberFLCVFolds' times
#for clientSet,foldIx in crossValClientSetCreation(dataset,numClients,clientDataHete,outerCVType,numberFLCVFolds):
	
	#split dataset into seperate datasets for each clients
#partitioned the dataset into clients either non-IID or IID 
#clientDataBlocks = createClient(dataset,numClients,clientDataHete)
	
#create a result file output row from a list of performance emtrics from each client's model
#given a prefix string
#example row output would have form: <prefix>,<client id>,<R2>,<MSE>,<RMSE>,<MAPE>,<execution time per client>,<communication cost per client>
#clientIDOverride: flag when true indicates to just put -1 constant as client for all results. False means client id listed normally
def parseResFileRowOutString(rowPrefixStr,clientSet,r2List,mseList,rmseList,mapeList,executionTimeList,commCostList,clientIDOverride=False):
	
	#integrity check
	#all results should have same length, since each client has the same number of result metrics
	if (len(r2List)!= len(mseList)) or \
		(len(r2List) != len(rmseList)) or \
		(len(r2List) != len(mapeList)) or \
		(len(r2List) != len(executionTimeList)) or  \
		(len(r2List) != len(commCostList)):	
		raise Exception("Cannot parse result file row. Expected same number of result metrics for each metrics, but the length was different.")
		
	numClients =len(r2List)
	outStr=""
	#iterate over each client's results to parse a result string over multiple rows for each client
	for cix in range(numClients):
		client = clientSet.getClient(cix)
		r2 = r2List[cix]
		mse = mseList[cix]
		rmse = rmseList[cix]
		mape = mapeList[cix]
		execTime=executionTimeList[cix]
		netComCost=commCostList[cix]
		nLocalSamples = client.getTotalNumberOfSamples()
		#override the client id?
		if clientIDOverride:
			clientID = -1
		else:
			clientID = cix
		#append performance metrics to the prefix for each model/client
		rowiStr = rowPrefixStr +str(nLocalSamples)+","+str(clientID)+ ","+str(r2)+ ","+str(mse)+ ","+str(rmse)+ ","+str(mape)+","+str(execTime)+","+str(netComCost)+"\n"
		outStr = outStr + rowiStr
		
	return outStr

#creates a set of maps to store performance metrics 
#to enable averaging the performance over all iterations and experiment type for a single experiment
def createGlobalAvgMetricMaps(expType,numClients):
	globalAvgMetricMaps={}
		
	#determine number of metrics per list
	if expType == CENTRALIZED_LEARNING_STR:
		metricListSize =1 #only 1 set of reasults for centralized learning
	else:
		metricListSize =numClients #metric associated to each client 
		
	#determine the type of experiment keys 
	if expType == LOCAL_AND_ENSEMBLE_LEARNING_STR:
	
		experimentTypes = PREFIX_LOCAL_ENSEMBLE_EXP_TYPE_LIST ##thre is a map for local, avg ensemble, and stacked ensemble
	else:
		experimentTypes	=[expType] #only a single map for given experiment type
		
		
	#go over all sets of metrics to create
	for metricExpTypeKey in experimentTypes:																
		globalAvgMetrics={}
		globalAvgMetricMaps[metricExpTypeKey]=globalAvgMetrics
		
		#create an entry for each metric
		for metricKey in METRICS_KEY_STRING_LIST:
			#list of 0s to enable summation
			globalAvgMetrics[metricKey]=np.zeros(metricListSize)	
					
	return globalAvgMetricMaps


#add the performance metrics 
def addToGlobalAvgMetricMaps(globalAvgMetricMaps,expType,r2List,mseList,rmseList,mapeList):

	globalAvgMetrics=globalAvgMetricMaps[expType]
	
	#METRICS_KEY_STRING_LIST=["R2","MSE","RMSE","MAPE"]
	#make sure to order lists of metric in same order as keys
	metricsList =[r2List,mseList,rmseList,mapeList]
	
	#add metric value to list
	for metricIx in range(len(METRICS_KEY_STRING_LIST)):
		metricKey=METRICS_KEY_STRING_LIST[metricIx]
		metricValues = metricsList[metricIx]
		
		globalAvgMetrics[metricKey] = globalAvgMetrics[metricKey] + np.array(metricValues)

#given the sum of all performance metrics stored in a map, take the average and format 
#the average to a user-friendly string
#globalAvgMetricMaps: map of sum of metrics
#metricCounter: number of summations
def parseAvgExpPerformanceStr(globalAvgMetricMaps,metricCounter):
	
			
	globalPerformanceStr = "\nAverage performance over all clients:\n"
	#go over each group of experiment in the experiment
	for expTypeKey in globalAvgMetricMaps:
		expTypePerfStr="\t"+expTypeKey+":\n"
				
		globalAvgMetrics=globalAvgMetricMaps[expTypeKey]
		
		#iterate over each metric and compute the average and append a rounded version to result string
		for metricKey in globalAvgMetrics:
			metricSum= globalAvgMetrics[metricKey]
			
			#average out the performance of each client
			#(average performance of a client over all folds and iterations)
			metricClientAvg = metricSum/metricCounter
			
			#take the average of all average client performance of this metric
			metricFinalAvg= metricClientAvg.mean()
			
			expTypePerfStr= expTypePerfStr + "\t\t"+metricKey+": "+str(round(metricFinalAvg,2))+"\n"
		globalPerformanceStr = globalPerformanceStr +expTypePerfStr
			
	
	return globalPerformanceStr

#initializes and returns a list of maps to store client results
def createClientResultMaps(numClients,expType):
	resMap={}
	#create the map to store client results

	if expType == CENTRALIZED_LEARNING_STR:
		# clients send everything to cloud, so only 1 result map
		numClients = 1

	if expType==LOCAL_AND_ENSEMBLE_LEARNING_STR: #local learning and esnemble will have 3 different maps, local learning, average ensemble, and stacked ensemble
				
		resMap[LOCAL_LEARNING_STR]=_createClientResultMapsHelper(numClients)					
		resMap[AVERAGE_ENSEMBLE_LEARNING_STR]=_createClientResultMapsHelper(numClients)					
		resMap[STACKED_ENSEMBLE_LEARNING_STR]=_createClientResultMapsHelper(numClients)					
				
	else:
		resMap[expType]= _createClientResultMapsHelper(numClients)					
				
	return resMap		
	
def _createClientResultMapsHelper(numClients):
	#each client will have result map used to created prediction files
	localOutMapList=[]
	#iteration over each client to create their map
	for cix in range(numClients):
		localOutMap={}
		#iterate over all columsn that will be part of the map and create empty list
		for colName in LOCAL_RES_MAP_COL_NAMES:
			localOutMap[colName]=[]
		
		localOutMapList.append(localOutMap)
	return localOutMapList

		
		
#clears/empties each list of results in each of the client result maps
def clearClientResultMaps(resMap):
	#go over each type of result for this experiment
	for k in resMap:
		localOutMapList = resMap[k]
		#iteration over each client's map
		for cix in range(len(localOutMapList)):
			localOutMap=localOutMapList[cix]
			#iterate over all columsn that will be part of the map and create empty list
			for colName in LOCAL_RES_MAP_COL_NAMES:
				localOutMap[colName].clear()

#populate the result maps that are used at the end of an experiment's iteration to output results 
#to a file with timestamps, actual values, and predictions, etc.
#localOutMapList: list of result maps for each client
#clientSet: set of clients and their datsaets
#clientPreds: list of prediction sets for each client
#unscaler: used for unscaling predictions back to original scale
def populateClientResultMaps(localOutMapList,clientSet,clientPreds,foldIx,unscaler):
	
	#iterate over each client's results
	for cix in range(clientSet.getNumberOfClients()):
		localOutMap = localOutMapList[cix]
		
		client = clientSet.getClient(cix)
		
		#predictions of client and unscale
		pred = clientPreds[cix]
		pred = unscaler.unscale(pred)
			
		#convert predictions to appropriate scale
		test_y = unscaler.unscale(client.test_y)
				
		#sample ids
		mycommon.listAppend(localOutMap[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME],client.localTestSampleIds)
				
		#time stamps
		mycommon.listAppend(localOutMap[dataset.TIMESTAMP_COLUMN_NAME],client.testTimeStamps)		
		
		
		#each prediction made for the same fold		
		foldIxList=[]		
		for i in range(len(pred)):								
			foldIxList.append(foldIx)
			
		
		#fold id
		mycommon.listAppend(localOutMap[FOLD_RES_FILE_COL_STR],foldIxList)		
		
		#actual 2no
		mycommon.listAppend(localOutMap[ACTUAL_N2O_RES_FILE_COL_STR],test_y)
		
		#predictions
		##we have 2 identical predictions columns just so that output analysis is consistent for resultAnalyzer.py
		mycommon.listAppend(localOutMap[PRED_INNER_FOLD_RES_FILE_COL_STR],pred)		
		mycommon.listAppend(localOutMap[PRED_AVG_FOLD_RES_FILE_COL_STR],pred)		
			
			
def createPredictionResFile(resMap,expDir,eix,itNum,expType):
	newItNum=itNum
	for k in resMap:
		localOutMapList	= resMap[k]
		#override iteration number for local/avg ensbmel, and stacked ensemble learnin
		#so will have 3 files for each iteration, one for local, avg ensemble, and stacked ensemble in same experiment directory
		if expType == LOCAL_AND_ENSEMBLE_LEARNING_STR:
			newItNum=str(itNum)+"."+k
		
		if expType == CENTRALIZED_LEARNING_STR:
			#don't output client prediction file for centralized learnign
			_createPredictionResFileHelper(localOutMapList,expDir,eix,newItNum,False)
		else:
			_createPredictionResFileHelper(localOutMapList,expDir,eix,newItNum,True)
		
	
#given a map of results, create a result file for each client, and create a 
#global result file that contains the union of all results that seperates client results from
#each other by including a clientid column in the global result file
#results are save to:
#	experiments/e<eix>/i<itNum>.csv #the global result file
#	experiments/e<eix>/clients/c0/i<itNum>.csv #client 0 results
#	experiments/e<eix>/clients/c1/i<itNum>.csv #client 1 results
#							...
#	experiments/e<eix>/clients/cn/i<itNum>.csv #client n results
#localOutMapList: list of each client's result map
#expDir: experiment root directory
#eix: experiment number
#itNum: iteration number
#outputClientFilesFlag: true means files for each client will be output, false means only global will be output
def _createPredictionResFileHelper(localOutMapList,expDir,eix,itNum,outputClientFilesFlag):
	if outputClientFilesFlag:
		#create a prediction output file for each client
		for cix in range(len(localOutMapList)):		
			localOutMap = localOutMapList[cix]
			#make a check that all columns in prediction file CSV output are same length (bug free means they all same length)
			previousColLen=-1
			for k in localOutMap:
				
				if previousColLen != -1:
					#mis match of column lengths?
					if previousColLen !=len(localOutMap[k]):
						myio.logWrite("Issue outputing prediction file of an iteration due to differign length of column. Column "+str(k)+" has length "+str(len(localOutMap[k]))+" but expected "+str(previousColLen),myio.LOG_LEVEL_ERROR)
				else:
					previousColLen=len(localOutMap[k])
					
			localPredictionDF = pd.DataFrame(data=localOutMap)
																	
			localPredictionDF = localPredictionDF.sort_values(by=[dataset.TIMESTAMP_COLUMN_NAME])#sort by timestamp (from early to later readings)	


			predFilePath = myio.parseClientPredictionFilePath(expDir,eix,cix,itNum)	
			
			myio.logWrite("Writing local prediction results for experiment  "+str(eix)+" and client "+str(cix)+" to :"+ predFilePath,myio.LOG_LEVEL_DEBUG)
			
			localPredictionDF.to_csv(predFilePath,sep=",",index=False,encoding='utf-8')#note that there will be a prediction for a sample id for each inner fold, so to chart the predictions, could take average prediction over a sample id
		
	#create a single prediction result file for the iteration of the experiment
	globalOutDF=None
	#and combine all results into a single prediciton file
	for cix in range(len(localOutMapList)):		
		localOutMap = localOutMapList[cix]
		
		localDF=pd.DataFrame(data=localOutMap)						
		
		#client id column (all the same for a client's local dataset)
		clientIdList = []
		for i in range(len(localDF.index)):
			clientIdList.append(cix)
			
		#add client id column as first (0) column to dataframe
		localDF.insert(0,"clientid",clientIdList)
		
		#first client?
		if globalOutDF is None:						
			globalOutDF = localDF #set global dataframe as the 1st client's dataframe
		else:
			globalOutDF=pd.concat([globalOutDF,localDF]) #concatenate the client dataframe to global dataframe
			
	
	#output the global predictions of all clients combined															
	globalOutDF = globalOutDF.sort_values(by=[dataset.TIMESTAMP_COLUMN_NAME])#sort by timestamp (from early to later readings)	

	predFilePath = myio.parsePredictionFilePath(expDir,eix,itNum)	
	
	myio.logWrite("Writing  global prediction results for experiment  "+str(eix)+" to :"+ predFilePath,myio.LOG_LEVEL_DEBUG)
	
	globalOutDF.to_csv(predFilePath,sep=",",index=False,encoding='utf-8')	

#simulates case where all clients send their datasets to the cloud, and a centralized model is built and evaluated
def centralizeLearningEval(clientSet,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler):
	totalNetCommunication=0
	
	
	
	centralizedTrain_X=None
	centralizedTrain_y=None
	
	centralizedTest_X=None
	centralizedTest_y=None
	
	centrazliedTestTimeStamps=None
	centrazliedTestSampleIds=None
	
	#we take union of all the client datasets to simulated sendign all data to server
	for i in range(clientSet.getNumberOfClients()):
		client = clientSet.getClient(i)
		#first client?
		if i ==0:			
			centralizedTrain_X=np.copy(client.train_X)
			centralizedTrain_y=np.copy(client.train_y)
			
			centralizedTest_X=np.copy(client.test_X)
			centralizedTest_y=np.copy(client.test_y)
			
			centrazliedTestTimeStamps=np.copy(client.testTimeStamps)
			centrazliedTestSampleIds=np.copy(client.localTestSampleIds)
				
		
		else:
			#append the client datasets into a centralized dataset
			centralizedTrain_X=np.concatenate((centralizedTrain_X,client.train_X))
			centralizedTrain_y=np.concatenate((centralizedTrain_y,client.train_y))
			
			centralizedTest_X=np.concatenate((centralizedTest_X,client.test_X))
			centralizedTest_y=np.concatenate((centralizedTest_y,client.test_y))
			
			centrazliedTestTimeStamps=np.concatenate((centrazliedTestTimeStamps,client.testTimeStamps))
			centrazliedTestSampleIds=np.concatenate((centrazliedTestSampleIds,client.localTestSampleIds))

	#NECEssary to create a single client set for storing results
	dummyClient = FLClient(0,np.copy(centralizedTrain_X),np.copy(centralizedTrain_y),np.copy(centralizedTest_X),np.copy(centralizedTest_y),centrazliedTestTimeStamps,centrazliedTestSampleIds) 	
	tmpCentralizedClientSet = FLClientSet([dummyClient],allowSingleClientFlag=True)#1 client in set with the centralized dataset 
	
	
	#clients send all their data to the cloud
	totalNetCommunication = mycommon.npArrayMemoryUsage(centralizedTrain_X)
	totalNetCommunication = totalNetCommunication + mycommon.npArrayMemoryUsage(centralizedTrain_y)
	totalNetCommunication=totalNetCommunication+mycommon.npArrayMemoryUsage(centralizedTest_X)
	totalNetCommunication=totalNetCommunication+mycommon.npArrayMemoryUsage(centralizedTest_y)
	
	totalNetCommunication = totalNetCommunication/mycommon.NUMBER_BYTES_PER_KILOBYTE #convert to KB
	
	startTime=time.time()
	
	_,r2,mse,rmse,mape,pred = fitAndEvaluateModel(centralizedTrain_X,centralizedTrain_y,centralizedTest_X,centralizedTest_y,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler)
	
	endTime=time.time()
	
	ellapsedTimeSec = endTime-startTime
	
	return r2,mse,rmse,mape,ellapsedTimeSec,totalNetCommunication,pred,tmpCentralizedClientSet
	
#clients train a local model and evaluate their model only on their data. no model or data is shared between clients
def localLearningEval(clientSet,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler):

	
	localR2List=[]
	localMSEList=[]
	localRMSEList=[]
	localMAPEList=[]
	execTimeList=[]
	netCommCostList=[]
	predList=[]
	#train/evaluate local model for each client
	for client in clientSet:		
		
		startTime=time.time()
		
		model,r2,mse,rmse,mape,pred = fitAndEvaluateModel(client.train_X,client.train_y,client.test_X,client.test_y,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler)
		
		endTime=time.time()		
		ellapsedTimeSec = endTime-startTime
		execTimeList.append(ellapsedTimeSec)
		
		netCommCostList.append(0)#nothing sent over network other than performance metrics, so its virtually 0
		
		#design error check
		if not hasattr(model,"model"):
			myio.logWrite("Expected a trained local model, but the model does not exist. Ensemble experiments will fail.... Model object type: "+model.toString(),myio.LOG_LEVEL_ERROR)
						
		#save the model so that ensemble learning evaluation can proceed afterwards
		client.setModelWrapper(model)
		
		
		
		localR2List.append(r2)
		localMSEList.append(mse)
		localRMSEList.append(rmse)
		localMAPEList.append(mape)
		predList.append(pred)
						
			
	return localR2List,localMSEList,localRMSEList,localMAPEList,execTimeList,netCommCostList,predList
	
#given a set of clients with many local models, we simulate the clients sharing each other's model over the
#network and then a client simply takes the average of all model predictions for the client's local data, as the final
#prediction (average ensemble)
def averageEnsembleLearningEval(clientSet,unscaler):
				
				
	numClients=clientSet.getNumberOfClients()
	
	localR2List=[]
	localMSEList=[]
	localRMSEList=[]
	localMAPEList=[]
	execTimeList=[]
	netCommCostList=[]
	predList=[]
	#iterate over every client to have client do a local average ensemble
	for client in clientSet:
		
		startTime=time.time()
		
		predsSum=np.zeros(client.test_y.shape,dtype=client.test_y.dtype)		
		#iterate over every model shared with client i, to make a prediction over test data 
		#and average out the predictions
		for modelIx in range(numClients):					
			model= clientSet.getClient(modelIx).modelWrapper
			preds=model.predict(client.test_X)
			
			predsSum = predsSum + preds
		
		predsAvg = predsSum/numClients #average out all predictions
				
		
		endTime=time.time()	
		ellapsedTimeSec = endTime-startTime
		execTimeList.append(ellapsedTimeSec)
		
		#a client must send its model to all other clients, so size of the model weights is the network cost (we
		#assume all clients agreed before hand on model architecture, so only weights are shared and are part of the 
		#communcation cost)
		
		#myio.logWrite("about to access client model wrapper's model, here is type of modelwrapepr: " + str(type(client.modelWrapper)),myio.LOG_LEVEL_INFO)
						
		netCommCost=mycommon.computeModelWeightMemoryUsage(client.modelWrapper.model) * numClients
		netCommCostList.append(netCommCost)
		
		r2,mse,rmse,mape=evaluatePredictions(client.test_y,predsAvg,unscaler)
		
		
		localR2List.append(r2)
		localMSEList.append(mse)
		localRMSEList.append(rmse)
		localMAPEList.append(mape)
		predList.append(predsAvg)
	
		
	return localR2List,localMSEList,localRMSEList,localMAPEList,execTimeList,netCommCostList,predList

#given a set of clients with many local models, we simulate the clients sharing each other's model over the
#network and then a meta-model (linear regression) is trained using local model predictions, forming a stacked ensemble
def stackedEnsembleLearningEval(clientSet,numInnerFolds,unscaler):	

	
	numClients=clientSet.getNumberOfClients()
	
	localR2List=[]
	localMSEList=[]
	localRMSEList=[]
	localMAPEList=[]
	execTimeList=[]
	netCommCostList=[]
	predList=[]
	#iterate over every client to have client do a local ensemble
	for client in clientSet:
	
		startTime=time.time()
		
		#create level-1 datasets to train adn evalaute meta model
		#make a prediction over test data of client i using every local model
		#and keep track of predicitons from each local model
		predCols=[]
		for modelIx in range(numClients):
			model= clientSet.getClient(modelIx).modelWrapper
			preds=model.predict(client.test_X)
			predCols.append(preds)
		
		#make sure the predictions of a model are in column format (transpose rows with columns)
		#so one row has the predictions of each model for a sample
		predCols= np.array(predCols)				
		level1Dataset_X = np.transpose(predCols,(1,0))
		
		level1Dataset_y = client.test_y
		
		nSamples =level1Dataset_X.shape[0]
		nFeatures = numClients
		myio.logWrite("Stacked ensemble client "+str(client.id)+" number of level 1 dataset samples "+str(nSamples) ,myio.LOG_LEVEL_DEBUG)
		
		shapeDebugStr="("
		for s in level1Dataset_X.shape:
			shapeDebugStr = shapeDebugStr + str(s)+","
		shapeDebugStr = shapeDebugStr + ")"
		
		myio.logWrite("Stacked ensemble client "+str(client.id)+" level 1 dataset shape: "+ shapeDebugStr,myio.LOG_LEVEL_DEBUG)
		
		
		#keep track of the index shuffles each fold to re-order the predictions by index
		#to keep the original ordering for outputing predictions
		shuffledTestIxList=[]
		shuffledPredList =[]
		
		totalR2=0
		totalMse=0
		totalRmse=0
		totalMape=0
		metaModelFoldIx=0
		
		innerPredList=[]
		
		#shuffle the level 1 dataset and do 5-fold cross validation to train and evaluate
		shuffledIndices = mycommon.randomRowIndexShuffle(nSamples)
		
		
		if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
			#map that tracks frequency a sample was part of testing
			outerSampleTestCountMap={}
			outerSampleTrainCountMap={}
		
		if len(shuffledIndices) < numInnerFolds:
				myio.logWrite("Stacked ensemble less data than number of folds. Setting number of folds number of samples ("+str(len(shuffledIndices))+")" ,myio.LOG_LEVEL_WARNING)
				numInnerFolds=len(shuffledIndices)
				
		for trainIxs,testIxs in experimenter.crossValSplit(experimenter.RANDOM_CROSS_VAL_TYPE,nSamples,numInnerFolds,shuffledIndices):				
			
			myio.logWrite("Stacked ensemble client "+str(client.id)+" fold "+str(metaModelFoldIx)+" number of inner test samples "+str(len(testIxs))+" and number of inner train samples "+str(len(trainIxs)),myio.LOG_LEVEL_DEBUG)
			
			if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
				#make sure test and train data aren't mixed and that samples aren't missed/forgotten in one of fold splits
				for testIx in testIxs:
					if testIx in outerSampleTestCountMap:
						myio.logWrite("sample was part of outter test set more than once ",myio.LOG_LEVEL_ERROR)
					else:
						outerSampleTestCountMap[testIx]=1
					for trainIx in trainIxs:					
						if testIx==trainIx:
							myio.logWrite("Outter CV train-test data contamination.",myio.LOG_LEVEL_ERROR)
				for trainIx in trainIxs:
					if trainIx in outerSampleTrainCountMap:
						outerSampleTrainCountMap[trainIx] = outerSampleTrainCountMap[trainIx] +1
					else:
						outerSampleTrainCountMap[trainIx]=1
				
				
			innerTrain_X=level1Dataset_X[trainIxs]
			innerTrain_y = level1Dataset_y[trainIxs]
			
			innerTest_X=level1Dataset_X[testIxs]
			innerTest_y = level1Dataset_y[testIxs]
			
			
			
			#nSamples = len(level1Dataset_X)
			inputShape=[nSamples,nFeatures]# i don't think this is used by model class, but will provide it for completness
			hyperParams={}#no hyperparameters
			
			metaModel = mymodel.Model(mymodel.ALG_LINEAR_REGRESSION,hyperParams,inputShape)
			
			#train model (test data isn't necessary for linear regression, but we provide it nevertheless to follow our project's API)
			metaModel.fit(innerTrain_X,innerTrain_y,innerTest_X,innerTest_y)
			
			_,r2,mse,rmse,mape,pred= evaluateModel(innerTest_X,innerTest_y,metaModel,unscaler)
			
			#keep track of the order of the shuffled samples to undo the shuffling of the predictions and match 
			#the predictions to the test samples in the same order as the client's test data
			mycommon.listAppend(shuffledTestIxList,testIxs)
			mycommon.listAppend(shuffledPredList,pred)
			
			totalR2=totalR2+r2
			totalMse=totalMse+mse
			totalRmse=totalRmse+rmse
			totalMape=totalMape+mape
			
			
			metaModelFoldIx = metaModelFoldIx +1
		
		if experimenter.TRAIN_TEST_MIX_INTEGRITY_CHECK:
			if nSamples != len(outerSampleTestCountMap):
				myio.logWrite("One of the samples in the inner CV was not part of a test set (expected count "+str(len(outerTrainDF.index))+" but was "+str(len(innerSampleTestCountMap))+"). Did you set an inner CV cluster size that was too small?",myio.LOG_LEVEL_ERROR)
			if nSamples != len(outerSampleTrainCountMap):
				myio.logWrite("One of the samples in the inner CV was not part of a train set (expected count "+str(len(outerTrainDF.index))+" but was "+str(len(innerSampleTrainCountMap))+") Did you set an inner CV cluster size that was too small?",myio.LOG_LEVEL_ERROR)
				
			#samples part of test set exactly once
			for testIx in outerSampleTestCountMap:				
				if outerSampleTestCountMap[testIx]!=1:
					myio.logWrite("One of the samples in the inner CV was part of the test set more than once ("+str(outerSampleTestCountMap[testIx])+" times)",myio.LOG_LEVEL_ERROR)
			#samples part of train set nubmber of folds -1
			for trainIx in outerSampleTrainCountMap:				
				if outerSampleTrainCountMap[trainIx]!=metaModelFoldIx - 1:
					myio.logWrite("One of the samples in the inner CV was not part of the train set number of folds ("+str(metaModelFoldIx)+") -1 times",myio.LOG_LEVEL_ERROR)
					
		#undo the shfufling to restore the original ordering of predictions
		tmpDf = pd.DataFrame(data={"shuffled-index":shuffledTestIxList,"preds":shuffledPredList})
		tmpDf = tmpDf.sort_values(by=["shuffled-index"])
		clientPreds = tmpDf["preds"]
		#the predictions for current client
		predList.append(clientPreds)
		
		
		#average inner cross-valdiation performance over all folds for client i
		#and treat this average as final performance for client i
		avg_r2=totalR2/metaModelFoldIx
		avg_mse=totalMse/metaModelFoldIx
		avg_rmse=totalRmse/metaModelFoldIx
		avg_mape=totalMape/metaModelFoldIx
		
		endTime=time.time()	
		ellapsedTimeSec = endTime-startTime
		execTimeList.append(ellapsedTimeSec)
		
		#same as average ensemble learning, a client must send its model to all other clients, so size of the model weights is the network cost (we
		#assume all clients agreed before hand on model architecture, so only weights are shared and are part of the 
		#communcation cost) (this assumes upload and dowload cost are the same, 1 client upload, n -1 clients download)
		netCommCost=mycommon.computeModelWeightMemoryUsage(client.modelWrapper.model) * numClients
		netCommCostList.append(netCommCost)
		
		localR2List.append(avg_r2)
		localMSEList.append(avg_mse)
		localRMSEList.append(avg_rmse)
		localMAPEList.append(avg_mape)
	
	return localR2List,localMSEList,localRMSEList,localMAPEList,execTimeList,netCommCostList,predList

def federatedLearningEval(totalNumTrainSamples,clientSet,year,algName,aggregator,numRounds,numClientsPerRound,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler):			
	numClients = clientSet.getNumberOfClients()
	totalNetCommunication=0
	
	localR2List=[]
	localMSEList=[]
	localRMSEList=[]
	localMAPEList=[]
	execTimeList=[]
	netCommCostList=[]
	predList=[]
	
	#so the execution time of this process is not really formal, but 
	#we track it the follwoing way anyway since it should give some useful information on
	#exeuction time:
	#	-clients will track the time it takes to train and evaluate their models
	#	-the coordination server will tracks the total time it takes to aggregate model updates
	#	upon receiving the updates
	#	-at the end of the evaluation, the total time taken by of the coordination server  for each
	#	aggregation will be added to each client's total time. This way in results file, where client each
	# 	have their own exectution time, this should give a somewhat reasonble approximation of the 
	#	total execution time experience by each client. The whole experiment is a simulation in sequence of 
	#	a parralel, process, so execution time is an approximation for all but centralize learning exeperiments
	#note that this may not be correctly accounting for the fact clients sometimes do not participate in rounds
	#we leave it to higher layer logic to analyzer execution time by 
	
	totalCoordServerAggTime=0
	#deep copy of list of clients to shuffle without disturbing upper-layer logic
	shuffledClients = []
	for client in clientSet:
		shuffledClients.append(client)
		
	if aggregator == "FedAvg":
		startTime = time.time()
		globalModelWrapper = createModel(year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride) #start with same initilizer weights for all clients
			
		#build the same model for each client
		for client in clientSet:
			localModelWrapper=createModel(year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride)
			client.setModelWrapper(localModelWrapper)
			
		#simulate many rounds of communication 
		for r in range(numRounds):
			#take random subset of the clients
			random.shuffle(shuffledClients)
			#models = []
			#only consider up to  numClientsPerRound clients
			for i in range(numClientsPerRound):
				client=shuffledClients[i]				
				#client update: simulated sending  model to client, which would create a client update in parralel in practice, 
				#but is done sequentially in this simulation
				_ = client.FLModelUpdate(globalModelWrapper) #'_' since don't use model returned, since its a simulation, we have access to the model stored in the client
				
				#models.append(clientModel)
			
			#here the server would wait to gather all client weeight updates in practices, but in this simulation
			#network transmission is assumed to be 100% reliable and instantaneous since we access client models directly from memory
					
			#aggregate the weights over all clients
			modelUpateAggregation(globalModelWrapper.model,clientSet,totalNumTrainSamples)
			
		
		#end of federate learning
		
		#send the final global model to all clients and compute total network cost
		for client in clientSet:			
			client.FLReceiveFinalGlobalModel(globalModelWrapper)
			
			#append the total network communciation cost of each client to the final result list
			netCommCostList.append(client.communicationCost)
			
		#evaluate model
			
		
		#iterate over every client to evaluate client's performance
		for client in clientSet:
			
			_,r2,mse,rmse,mape,pred = client.clientEvaluateModel(unscaler)
				
			localR2List.append(r2)
			localMSEList.append(mse)
			localRMSEList.append(rmse)
			localMAPEList.append(mape)
			predList.append(pred)
			
			#clientR2=clientR2+r2
			#clientMse=clientMse+mse
			#clientRmse=clientRmse+rmse
			#clientMAPE=clientMAPE+mape
		
		
		endTime = time.time()
		ellapsedTime = endTime-startTime
		
		#to compute the time taken by aggregation server, since evertyhign is done in sequence/series
		#total server execution time = total execution time - total time ellapsed over all clients
		totalCoordServerAggTime = ellapsedTime
		for client in clientSet:
			totalCoordServerAggTime = totalCoordServerAggTime - client.execTime
		
		
		#now iterate over each client and add the total execution time of
		#of coordiation server's aggregations operations to approximate each
		#client's execution time by including the server's execution time in  their execution
		for client in clientSet:
			client.execTime = client.execTime+totalCoordServerAggTime
			execTimeList.append(client.execTime)
		
			
		return localR2List,localMSEList,localRMSEList,localMAPEList,execTimeList,netCommCostList,predList
			
	else:		
		raise Exception("Federated learning aggregator algorihtm '"+aggregator+"' not supported.")
		
#future work: look into https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
#if want to implement fedprox
def modelUpateAggregation(globalModel,clientSet,totalNumTrainSamples):
	
	#globalModelCopy = globalModel  deep copy
	#iterate each layer of model
	for layerIx in range(len(globalModel.layers)):
		globalLayer =globalModel.layers[layerIx]
		#globalLayerWeights =globalLayer.get_weights()
		globalWeightsSum=None				
		
		#iterate over each client's model
		for cix in range(clientSet.getNumberOfClients()):
			
			client = clientSet.getClient(cix)
			
			clientModel= client.modelWrapper.model
			nLocalTrainSamples = client.getNumberOfTrainSamples()
			
			
			#take average (normalized by number of samples on each client) wieght over all client for current layer
			weightUpateNorm = nLocalTrainSamples/totalNumTrainSamples#normaliztion of local weight				
			
			#weights of the client
			cWeights =clientModel.layers[layerIx].get_weights()		
			
			#first client?
			if cix == 0:
				#global weights initizliaed to  first clients' weight 
				globalWeightsSum = copy.deepcopy(cWeights) 
																												
			#iterate over both biases and weights of the layer
			for wix in range(len(globalWeightsSum)):

				if cix == 0:
					globalWeightsSum[wix]=globalWeightsSum[wix]*weightUpateNorm							
				else:
					
					globalWeightsSum[wix]= globalWeightsSum[wix]+(weightUpateNorm*cWeights[wix])
			
		
		#update global model layer-by-layer
		globalLayer.set_weights(globalWeightsSum)

#trains a model and evaluates it
def fitAndEvaluateModel(train_X,train_y,test_X,test_y,year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride,unscaler):
	
	#evaluate the model by training using all the trainign data
	model = createModel(year,algName,inputShape,learnRateOverride,epochsOverride,batchSizeOverride)
	model.fit(train_X,train_y,test_X,test_y)
		
	return evaluateModel(test_X,test_y,model,unscaler)
	
#evaluates a model
def evaluateModel(test_X,test_y,model,unscaler):
	
	pred = model.predict(test_X)
						
	r2,mse,rmse,mape=evaluatePredictions(test_y,pred,unscaler)
	
	return model,r2,mse,rmse,mape,pred

#evaluates predictions  and returns the unscaled version of input data dn predictions
def evaluatePredictions(test_y,pred,unscaler):
	#undo data normalization (if it exists) to have evaluation metrics in correct units
	unscaledTest_y=unscaler.unscale(test_y)
	unscaledPred = unscaler.unscale(pred)
	
	r2 = mycommon.computeR2(unscaledTest_y,unscaledPred)
	
	mse = mycommon.computeMSE(unscaledTest_y,unscaledPred)
	rmse = math.sqrt(mse)
	mape = mycommon.computeMAPE(unscaledTest_y,unscaledPred,smallAbsValLim=1) #smallAbsValLim=1 to snap all emssions between -1 and 1 to -1 and 1, which ever is closest
	
	return r2,mse,rmse,mape
	


#create a model, where the year will determine what hyperparaemters to use
def createModel(year,algName,inputShape,learnRateOverride=-1,epochsOverride=-1,batchSizeOverride=-1):

	# static hyperparameter are  chosen from results of preliminary experiments (different sets of hyperparameters for each year)
	hyperParameters=mymodel.getFLStaticModelHyperparameters(algName,year)
	
	#only override learning rate, epcochs, and batch size for deep learnign models
	if mymodel.isDeepLearningModel(algName):
		#override learning rate?
		if learnRateOverride != -1:
			hyperParameters["learning_rate"]=learnRateOverride
		#override number of epochs?
		if epochsOverride != -1:
			hyperParameters["epochs"]=epochsOverride
		#override batchSize?
		if batchSizeOverride != -1:
			hyperParameters["batchSize"]=batchSizeOverride
			
		
	return mymodel.Model(algName,hyperParameters,inputShape)

#only run the below code if ran as script
if __name__ == '__main__':
		
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--inFile", type=str, required=True,
		help="path to configuration file")
	ap.add_argument("-o", "--outDirectory", type=str, required=True,
		help="path to output directory")
	args = vars(ap.parse_args())
		
	inputConfigFile = args["inFile"]
	outDir = args["outDirectory"]
	runFLExperiments(inputConfigFile,outDir)