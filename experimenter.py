import traceback
import sys
import os
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
import random
import math
import functools #for partial functions where return a function with assigned argument parameters
import dataset
import myio
import argparse
import common as mycommon
DEBUGGING_FLAG=False#set to false when run on linux

import model
	
	
try:	
	from keras_tuner import RandomSearch
	from keras_tuner import Objective
	import tensorflow as tf; 	
except ImportError as e:
	#only consider it error when not debugging and package import failed
	if not DEBUGGING_FLAG:
		print("Failed to import packages: "+str(e)+"\n"+str(traceback.format_exc()))
		exit()

MAPE_SMALL_VALUE_OVERRIDE_THRESHOLD=1 #emissions between -1 and 0 are snapped to -1, and values between 0 and 1 are snapped to 1 to avoid large meaningless MAPE  errors

CV_SAMPLE_ORDERING_COLUMN_NAME="index of sample after CV shuffle"

RANDOM_CROSS_VAL_TYPE="random CV"
BLOCKED_CROSS_VAL_TYPE="block CV"



#debug flag that controls whether we check if test and train data have been mixed together
#and that every sample was part of the testing exactly once for a cross-validation process
TRAIN_TEST_MIX_INTEGRITY_CHECK=True

#inConfigFile: csv input file that lists paragrameters for each experiment, where each row defines an experiment to run. 
#outputDir: parent directory to store all output files of experiments
def runSingleDatsetCVExperiments(inConfigFile,outputDir):
	
	#read the experiment configuration file 
	configDF = pd.read_csv(inConfigFile, sep=",")
	
	#make sure all needed columns are present
	myio.configFileIntegrityCheck(configDF,myio.SINGLE_DATASET_CV_CONFIG_FILE_TYPE)
	
	numExperiments = len(configDF.index)
	
	
	logFilePath,resFilePath,expDir,outDirPath=myio.setupOutputDirectory(inConfigFile,outputDir, numExperiments)
	
	myio.openLogFile(logFilePath)
	#here were telling python than were not defining a local variable, instead were assigning a value to the global log variable so that other functions can 
	#log things without needing to pass the log reference everytwhere (for clarity reasons)
	#global globalLogFile 
	
	
	
	myio.logWrite("Starting ML cross-validation experiments from experiment file "+inConfigFile,myio.LOG_LEVEL_INFO)
	
	#list the gpu devices detected by tensor lflow
	gpus = tf.config.list_physical_devices('GPU')
	
	
	if len(gpus)==0:
		myio.logWrite("Running deep learning models on the CPU, since the GPU was not detected",myio.LOG_LEVEL_WARNING)
		processingDevice = "CPU"
	else:
		processingDevice = "GPU"
	
	
	resFile = open(resFilePath,"a")
		
	#create head
	resFile.write("experiment id,processing device,algorithm,"+myio.YEAR_KEY+","+myio.CHAMBER_KEY+","+myio.TEMPORAL_RESOLUTION_KEY+","+myio.INPUT_TENSOR_NUMBER_TIME_STEPS_KEY+","+myio.FEATURE_SELECTION_SCHEME_KEY+",number of features,number of instances,iteration,seed,outer fold,inner fold,MSE,RMSE,R2,MAPE,execution time(s)\n")
	#for every experiment
	for eix in range(numExperiments):
		try:
			experimentStartTime=time.time()
			foldPairStartTimeMS=time.time()#the first fold pair execution time will include all the data spliting, but this way summing all fold output execution times will give more accurate total duration
			myio.flushLogFile()
			#make sure the format of values for the experimetn are correct
			myio.configEntryIntegrityCheck(eix,configDF,myio.SINGLE_DATASET_CV_CONFIG_FILE_TYPE)
			
			inputDatasetPath = configDF[myio.INPUT_DATASET_PATH_KEY][eix]
			selectedFeaturePath=configDF[myio.SELECTED_SENSOR_PATH_KEY][eix]
			if not os.path.isfile(inputDatasetPath):
				myio.logWrite("Could not find input dataset "+str(inputDatasetPath)+" for experiment "+str(eix)+". Skipping this experiment.",myio.LOG_LEVEL_ERROR)
				continue
			
			normalizeDataFlag = configDF[myio.APPLY_MIN_MAX_SCALING_FLAG_KEY][eix]
			
			year =configDF[myio.YEAR_KEY][eix]
			chamber =configDF[myio.CHAMBER_KEY][eix]
			tempResolution =configDF[myio.TEMPORAL_RESOLUTION_KEY][eix]
			featSelectionScheme =configDF[myio.FEATURE_SELECTION_SCHEME_KEY][eix]
			tensorNumberTimeSteps = configDF[myio.INPUT_TENSOR_NUMBER_TIME_STEPS_KEY][eix]
			
			#optional flag indicating whether a result file will be output in the experiments/e'eix'/ directy
			#that contains an entry for each fold and iteration with the hyperparameter choice of the model			
			if myio.OUTPUT_HYPERPARAMETER_CHOICE_FLAG_KEY in configDF:
				outputHyperParamsFlag=configDF[myio.OUTPUT_HYPERPARAMETER_CHOICE_FLAG_KEY][eix]
			else:
				outputHyperParamsFlag=False
	
			#for experiments that desired the hyperparameter choices to be 
			#saved as output files
			if outputHyperParamsFlag:
				hyperParamOutFilePath = myio.parseHyperParamOutFilePath(expDir,eix)
				hyperParamFile = open(hyperParamOutFilePath,"a")
				#can't create file CSV header now, since hyperparameter names are dependent on chosen algorithm
				#populate the header later
				firstHyperParamOutFlag=True
				
				
			#tensorSizeInHours=(tempResolution/60.0)* tensorNumberTimeSteps
			#inputTensorLen= dataset.computeTensorTimeSteps(tempResolution,inputTensorLenMin) #cimpute number of timesteps per tensor
			

			trials = configDF[myio.NUMBER_OF_TRIALS_KEY][eix]
			
			#read the dataset into memory 
			inDF,selectedFeatures = dataset.readDataset(inputDatasetPath,selectedFeaturePath)
	
			myio.logWrite("First 3 samples of raw dataset: "+str(inDF.values[0:3,:]),myio.LOG_LEVEL_DEBUG)
			myio.logWrite("Last sample of raw dataset: "+str(inDF.values[-1,:]),myio.LOG_LEVEL_DEBUG)
				
			
			originalN2OData=inDF.values[:,-1]
			
			#compute min and max n2o to  unscale the normalized
			#emissiosn for results presentation
			maxN2O = max(originalN2OData)
			minN2O = min(originalN2OData)
			
			algName = configDF[myio.ALGORITHM_KEY][eix]
			if model.isDeepLearningModel(algName):
				
				actualProcessingDevice=processingDevice #deep learning models use  GPU acceleration if available
			else:
				actualProcessingDevice="CPU"#basic ML models use the CPU
				
			#check taht algorithm has appropriate tensor length definition
			if tensorNumberTimeSteps > 1 and not model.isModelWithInputTensors(algName):
				raise Exception("Configuration file issue. Algorithm that expects no multi-dimensional input samples in time had tensor length > 1")
				
			#do we normalize/scale the data?
			if normalizeDataFlag:
				#normalize it
				normDF = dataset.normalize(inDF)
			else:
				normDF=inDF
				
							
			
			if model.isModelWithInputTensors(algName):
				#keep track of origional dataset before we remove reading that would lead to incomplete tensors (missing concecutive readigns)
				trimmedNormDF = normDF.copy(deep=True)  
				
				#remove rows were there is missing consecutive readings after to form a tensor
				#the sample ids don't change so that using sampleid as pointer to row in norm DF still possible
				dataset.removeNonConcurrentTensorSamples(trimmedNormDF,tensorNumberTimeSteps,tempResolution)
								
			else:
				trimmedNormDF=normDF #they are the same for models that don't mind that some rows not concecutive
					
			myio.logWrite("First 3 samples of normalized dataset: "+str(trimmedNormDF.values[0:3,:]),myio.LOG_LEVEL_DEBUG)
			myio.logWrite("Last sample of normalized dataset: "+str(trimmedNormDF.values[-1,:]),myio.LOG_LEVEL_DEBUG)
			
			#number features
			nFeatures = len(selectedFeatures)
			if (len(trimmedNormDF.columns)-3)!=nFeatures:
				raise Exception("Failed to properly read the dataset and select features.")
			
			#when tensor length is positive, then input samples are  nFeatures x tensorNumberTimeSteps
			#for basic ML and dnn, NO dimension in time, so just nFeatures x  1 =
			if tensorNumberTimeSteps>1:
				inputShape=[nFeatures,tensorNumberTimeSteps]
			else:
				inputShape=[nFeatures]
													
			featuresStr="["
			for featureName in selectedFeatures:				
				featuresStr = featuresStr + ",\n\t"+featureName
			featuresStr=featuresStr+"]"
			
					
			nSamples = len(trimmedNormDF.index) 
			batchSize=None #all samples in the batch, since will easily fit into memory. I think this is unused
			numExecutionsPerTrial =  configDF[myio.NUMBER_OF_EXECUTIONS_PER_TRIAL_KEY][eix] 
			
			#isModelWithInputTensors
			
			myio.logWrite("Running experiment "+str(eix)+"/"+str(numExperiments-1)+" with algorithm "+algName+" using dataset with "+str(nSamples)+" samples and "+str(nFeatures)+" features:\n ("+featuresStr+")." ,myio.LOG_LEVEL_INFO)
		
			finalR2=0
			finalMSE=0
			finalRMSE=0
			finalMAPE=0
			metricCounter=0
			#for AI models, split dataset into tensors (blocks of consecuritve timestemps), removing any tensor with missing data
				
			#reset rng seed
			rngSeed=configDF[myio.RNG_SEED_KEY][eix]
			random.seed(rngSeed)
		
						
			
			iterationOutputDataMap = {}
			iterationOutputDataMap[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME]=[]
			iterationOutputDataMap[dataset.TIMESTAMP_COLUMN_NAME]=[]
			iterationOutputDataMap[CV_SAMPLE_ORDERING_COLUMN_NAME]=[]
			iterationOutputDataMap["outer-fold"]=[]
			
			
			iterationOutputDataMap["actual_N2O"]=[]
			#there is a prediction column for every inner fold, since each inner fold has a different model trained using different optimal hyperparameters			
			for i in range (configDF[myio.NUMBER_INNER_FOLDS_KEY][eix]):			
				iterationOutputDataMap["predicted_inner_fold"+str(i)]=[]					
			iterationOutputDataMap["predicted_inner_fold_average"]=[]
			#for each iteration
			for itIx in range(configDF[myio.ITERATIONS_KEY][eix]):
				outFoldIx=0
				

				iterationOutputDataMap[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME].clear()
				iterationOutputDataMap[dataset.TIMESTAMP_COLUMN_NAME].clear()
				iterationOutputDataMap[CV_SAMPLE_ORDERING_COLUMN_NAME].clear()
				iterationOutputDataMap["outer-fold"].clear()
						
				iterationOutputDataMap["actual_N2O"].clear()
				iterationOutputDataMap["predicted_inner_fold_average"].clear()
				
				for i in range (configDF[myio.NUMBER_INNER_FOLDS_KEY][eix]):			
					iterationOutputDataMap["predicted_inner_fold"+str(i)].clear()
				
				
				#we will record the order the samples are shuffled from outter CV split
				#by shuffling the dataset same way as CV split and reseting seed
				#we reset the seed so that the CVsplit that randomly shuffles the dataset will shuffle it identically as or previous call
				#to createRandomRowIndexList, so that we can pre-determine what shuffle order of rows is without clutering the CV split logic			
				
				
				#get the shuffled indices of the samples that will be used by the cross validation splits to create folds
				#
				outerShuffledIndices = createRowIndexOrdering(configDF[myio.OUTER_CV_SPLIT_TYPE_KEY][eix],trimmedNormDF,configDF[myio.TEMPORAL_OUTER_CV_CLUSTER_SIZE_KEY][eix],configDF[myio.NUMBER_OUTER_FOLDS_KEY][eix],tensorNumberTimeSteps)
				
				finalShuffleOrderList = populateCVOrderingList(configDF[myio.OUTER_CV_SPLIT_TYPE_KEY][eix],outerShuffledIndices)
				
				
				mycommon.listAppend(iterationOutputDataMap[CV_SAMPLE_ORDERING_COLUMN_NAME],finalShuffleOrderList)
				
				outerNSamples=len(finalShuffleOrderList)
				#print("number of samples before  index order: "+str(len(normDF.index)))
				#print("number of samples after index order: "+str(len(flattenedOuterShuffledIndices)))
				
				if TRAIN_TEST_MIX_INTEGRITY_CHECK:
					#map that tracks frequency a sample was part of testing
					outerSampleTestCountMap={}
					outerSampleTrainCountMap={}
					
				
				#for each outer fold outerTrain-test fold pairs
				for outerTrainIxs,outerTestIxs in crossValSplit(configDF[myio.OUTER_CV_SPLIT_TYPE_KEY][eix],trimmedNormDF,configDF[myio.NUMBER_OUTER_FOLDS_KEY][eix],outerShuffledIndices): 
					if TRAIN_TEST_MIX_INTEGRITY_CHECK:
						#make sure test and train data aren't mixed and that samples aren't missed/forgotten in one of fold splits
						for testIx in outerTestIxs:
							if testIx in outerSampleTestCountMap:
								myio.logWrite("sample was part of outter test set more than once ",myio.LOG_LEVEL_ERROR)
							else:
								outerSampleTestCountMap[testIx]=1
							for trainIx in outerTrainIxs:					
								if testIx==trainIx:
									myio.logWrite("Outter CV train-test data contamination.",myio.LOG_LEVEL_ERROR)
						for trainIx in outerTrainIxs:
							if trainIx in outerSampleTrainCountMap:
								outerSampleTrainCountMap[trainIx] = outerSampleTrainCountMap[trainIx] +1
							else:
								outerSampleTrainCountMap[trainIx]=1
							
						
					inFoldIx=0
										
					#extract outer-train/test datasets
					outerTrainDF = trimmedNormDF.iloc[outerTrainIxs]
					outerTestDF = trimmedNormDF.iloc[outerTestIxs]
					
					
					#model has tensors?
					if model.isModelWithInputTensors(algName):
						#normDF given twice since we havne't take subset yet
						outerTrain_X,outerTrain_y,outerTrainTimestamps=dataset.createTensorSets(normDF,trimmedNormDF,outerTrainIxs,tensorNumberTimeSteps,tempResolution) 
						outerTest_X,outerTest_y,outerTestTimestamps =dataset.createTensorSets(normDF,trimmedNormDF,outerTestIxs,tensorNumberTimeSteps,tempResolution)
						outerTrain_X = outerTrain_X.reshape((outerTrain_X.shape[0], tensorNumberTimeSteps, nFeatures))
						outerTest_X = outerTest_X.reshape((outerTest_X.shape[0], tensorNumberTimeSteps, nFeatures))
							
						if len(outerTrain_X)==0:
							raise Exception("A fold (iteration-outerfold-innerfold): "+str(itIx)+"-"+str(outFoldIx)+"-"+str(inFoldIx)+"- had an empty number of samples")							
							
						if len(outerTest_X)==0:
							raise Exception("A fold (iteration-outerfold-innerfold): "+str(itIx)+"-"+str(outFoldIx)+"-"+str(inFoldIx)+"- had an empty number of samples")							
								
								
						myio.logWrite("First sample of outer train X fold: "+str(outerTrain_X[0]),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("First sample of outer test X fold: "+str(outerTest_X[0]),myio.LOG_LEVEL_DEBUG)
						
						myio.logWrite("First sample of outer train Y fold: "+str(outerTrain_y[0]),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("First sample of outer test Y fold: "+str(outerTest_y[0]),myio.LOG_LEVEL_DEBUG)
						
						if outerTrain_X.shape[2] != nFeatures and outerTrain_X.shape[1] != tensorNumberTimeSteps:
							raise Exception("Failed to extract the outter training features as numpy matrix (some features missing")
						if outerTest_X.shape[2] != nFeatures and outerTest_X.shape[1] != tensorNumberTimeSteps:
							raise Exception("Failed to extract the outter testing features as numpy matrix (some features missing")
						
						
					else:
						
						outerTrain_X,outerTrain_y = dataset.extractTrainTestValues(outerTrainDF)					 
						outerTest_X,outerTest_y = dataset.extractTrainTestValues(outerTestDF)
						 
						myio.logWrite("First sample of outer train X fold: "+str(outerTrain_X[0,:]),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("First sample of outer test X fold: "+str(outerTest_X[0,:]),myio.LOG_LEVEL_DEBUG)
						
						
						if outerTrain_X.shape[1] != nFeatures:
							raise Exception("Failed to extract the outter training features as numpy matrix (some features missing")
						if outerTest_X.shape[1] != nFeatures:
							raise Exception("Failed to extract the outter testing features as numpy matrix (some features missing")
						
					
					myio.logWrite("Shape of outter train X: "+str(outerTrain_X.shape),myio.LOG_LEVEL_DEBUG)
					myio.logWrite("Shape of outer test X: "+str(outerTest_X.shape),myio.LOG_LEVEL_DEBUG)
					myio.logWrite("Shape of outter train y: "+str(outerTrain_y.shape),myio.LOG_LEVEL_DEBUG)
					myio.logWrite("Shape of outter test y: "+str(outerTest_y.shape),myio.LOG_LEVEL_DEBUG)
					
					if normalizeDataFlag:
						deNormedOuterTest_y=dataset.denormalize(minN2O,maxN2O,outerTest_y)
					else:
						deNormedOuterTest_y=outerTest_y
					
					#keep track of predictions, expected value, timestamp, sample id, and folds for iteration prediction file
					#there will be duplicate predictions for each timestamp based on number inner folds, so we distinguish duplicate 
					#timestamp and samplid predictions via fold ids 
											
					
					#for each row in the predictions, we have the id of folds
					for i in range(len(outerTest_y)):												
						iterationOutputDataMap["outer-fold"].append(outFoldIx)
					
											
					mycommon.listAppend(iterationOutputDataMap[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME],outerTestDF[dataset.SAMPLE_ID_EXTRA_COLUMN_NAME])
					
					if model.isModelWithInputTensors(algName):
						mycommon.listAppend(iterationOutputDataMap[dataset.TIMESTAMP_COLUMN_NAME],outerTestTimestamps) #timestamps offset by tensor length
					else:
						mycommon.listAppend(iterationOutputDataMap[dataset.TIMESTAMP_COLUMN_NAME],outerTestDF[dataset.TIMESTAMP_COLUMN_NAME])
					mycommon.listAppend(iterationOutputDataMap["actual_N2O"],deNormedOuterTest_y)
					
					#get the shuffled indices of the samples that will be used by the cross validation splits to create folds
					innerShuffledIndices = createRowIndexOrdering(configDF[myio.INNER_CV_SPLIT_TYPE_KEY][eix],outerTrainDF,configDF[myio.TEMPORAL_INNER_CV_CLUSTER_SIZE_KEY][eix],configDF[myio.NUMBER_INNER_FOLDS_KEY][eix],0)#0 for timesteps since already delt with the last samples of incomplete timesteps
					
					if TRAIN_TEST_MIX_INTEGRITY_CHECK:
						#map that tracks frequency a sample was part of testing					
						innerSampleTestCountMap={}
						innerSampleTrainCountMap={}
										
										
					#for each inner fold innerTrain-valiation fold pairs
					for innerTrainIxs,innerValIxs in crossValSplit(configDF[myio.INNER_CV_SPLIT_TYPE_KEY][eix],outerTrainDF,configDF[myio.NUMBER_INNER_FOLDS_KEY][eix],innerShuffledIndices):
						
								
						if TRAIN_TEST_MIX_INTEGRITY_CHECK:
							#make sure test and train data aren't mixed
							for testIx in innerValIxs:
								if testIx in innerSampleTestCountMap:
									myio.logWrite("sample was part of inner test set more than once ",myio.LOG_LEVEL_ERROR)
								else:
									innerSampleTestCountMap[testIx]=1
								for trainIx in innerTrainIxs:					
									if testIx==trainIx:
										myio.logWrite("inner CV train-test data contamination.",myio.LOG_LEVEL_ERROR)
							for trainIx in innerTrainIxs:
								if trainIx in innerSampleTrainCountMap:
									innerSampleTrainCountMap[trainIx] = innerSampleTrainCountMap[trainIx] +1
								else:
									innerSampleTrainCountMap[trainIx]=1			
						
						
						#extract inner-train/validate datasets
						innerTrainDF = outerTrainDF.iloc[innerTrainIxs]#CRASH OCCURING HERE FOR SOME REASON
						
						innerValDF = outerTrainDF.iloc[innerValIxs]
						
						#model has tensors?
						if model.isModelWithInputTensors(algName):
							innerTrain_X,innerTrain_y,innerTrainTimestamps=dataset.createTensorSets(normDF,outerTrainDF,innerTrainIxs,tensorNumberTimeSteps,tempResolution) 
							innerVal_X,innerVal_y,innerTestTimestamps=dataset.createTensorSets(normDF,outerTrainDF,innerValIxs,tensorNumberTimeSteps,tempResolution) 
							#make sure its [sample,timestep,feature]
							innerTrain_X = innerTrain_X.reshape((innerTrain_X.shape[0], tensorNumberTimeSteps, nFeatures))
							innerVal_X = innerVal_X.reshape((innerVal_X.shape[0], tensorNumberTimeSteps, nFeatures))
							if len(innerTrain_X)==0:
								raise Exception("A fold (iteration-outerfold-innerfold): "+str(itIx)+"-"+str(outFoldIx)+"-"+str(inFoldIx)+"- had an empty number of samples")
								
							if len(innerVal_X)==0:
								raise Exception("A fold (iteration-outerfold-innerfold): "+str(itIx)+"-"+str(outFoldIx)+"-"+str(inFoldIx)+"- had an empty number of samples")
								
							myio.logWrite("First sample of inner train X fold: "+str(innerTrain_X[0]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner val X fold: "+str(innerVal_X[0]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner train y fold: "+str(innerTrain_y[0]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner val y fold: "+str(innerVal_y[0]),myio.LOG_LEVEL_DEBUG)
															
							if innerTrain_X.shape[2] != nFeatures and innerTrain_X.shape[1] != tensorNumberTimeSteps:
								raise Exception("Failed to extract the inner training features as numpy matrix (some features missing")
							if innerVal_X.shape[2] != nFeatures and innerVal_X.shape[1] != tensorNumberTimeSteps:
								raise Exception("Failed to extract the inner val features as numpy matrix (some features missing")
							

						else:							
							
							#extract the features X from target  variable y									
							innerTrain_X,innerTrain_y = dataset.extractTrainTestValues(innerTrainDF)						 
							innerVal_X,innerVal_y = dataset.extractTrainTestValues(innerValDF)
							myio.logWrite("First sample of inner train X fold: "+str(innerTrain_X[0,:]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner val X fold: "+str(innerVal_X[0,:]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner train y fold: "+str(innerTrain_y[0]),myio.LOG_LEVEL_DEBUG)
							myio.logWrite("First sample of inner val y fold: "+str(innerVal_y[0]),myio.LOG_LEVEL_DEBUG)
								
							if innerTrain_X.shape[1] != nFeatures:
								raise Exception("Failed to extract the inner training features as numpy matrix (some features missing")
							if innerVal_X.shape[1] != nFeatures:
								raise Exception("Failed to extract the inner val features as numpy matrix (some features missing")
						
						myio.logWrite("Shape of inner train X: "+str(innerTrain_X.shape),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("Shape of inner test X: "+str(innerVal_X.shape),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("Shape of inner train y: "+str(innerTrain_y.shape),myio.LOG_LEVEL_DEBUG)
						myio.logWrite("Shape of inner test y: "+str(innerVal_y.shape),myio.LOG_LEVEL_DEBUG)
					
					
						bestHyperParamSet = tuneBasicModelHyperParameters(algName,tempResolution,tensorNumberTimeSteps,trials,numExecutionsPerTrial,inputShape,innerTrain_X,innerTrain_y,innerVal_X,innerVal_y)
						
						#saving hyperparameters to file?
						if outputHyperParamsFlag:							
							outputHypeParams(bestHyperParamSet,firstHyperParamOutFlag,hyperParamFile,itIx,outFoldIx,inFoldIx)
							#first time writing to file?
							if firstHyperParamOutFlag:
								firstHyperParamOutFlag=False
								
							
						#create the model to compute final performance for this inner-outer fold pair using the entire training data with best hyperparameters to maximize performance
						#(were assuming that more training data means better performance)						
						mlModel = model.Model(algName,bestHyperParamSet,inputShape)
						#train model
						mlModel.fit(outerTrain_X,outerTrain_y,outerTest_X,outerTest_y)
						
						preds = mlModel.predict(outerTest_X)						
							
						r2 = mycommon.computeR2(outerTest_y,preds)
						
						if normalizeDataFlag:
							mse = mycommon.computeMSE(dataset.denormalize(minN2O,maxN2O,outerTest_y),dataset.denormalize(minN2O,maxN2O,preds))
							mape = mycommon.computeMAPE(dataset.denormalize(minN2O,maxN2O,outerTest_y),dataset.denormalize(minN2O,maxN2O,preds),MAPE_SMALL_VALUE_OVERRIDE_THRESHOLD)
						else:
							mse = mycommon.computeMSE(outerTest_y,preds)
							mape = mycommon.computeMAPE(outerTest_y,preds,MAPE_SMALL_VALUE_OVERRIDE_THRESHOLD)
						
												
						rmse = math.sqrt(mse)
					
						
						finalR2=finalR2+r2
						finalMSE=finalMSE+mse
						finalRMSE=finalRMSE+rmse
						finalMAPE=finalMAPE+mape
						metricCounter = metricCounter+1
						
						if normalizeDataFlag:
							deNormedPreds=dataset.denormalize(minN2O,maxN2O,preds)
						else:
							deNormedPreds=preds
							
						
						
						#append predictions for inner fold to iteration prediction output result map
						mycommon.listAppend(iterationOutputDataMap["predicted_inner_fold"+str(inFoldIx)],deNormedPreds)
											
						foldPairEndTimeMS=time.time()
						foldPairExecTime=foldPairEndTimeMS-foldPairStartTimeMS
						foldPairStartTimeMS=time.time()
						myio.logWrite("Experiment "+str(eix)+", algorithm "+algName+", (iteration)-(outer fold)-(inner fold): ("+str(itIx)+")-("+str(outFoldIx)+")-("+str(inFoldIx)+")",myio.LOG_LEVEL_DEBUG)						
						
						resFile.write(str(eix)+","+actualProcessingDevice+","+algName+","+str(year)+","+str(chamber)+","+str(tempResolution)+","+str(tensorNumberTimeSteps)+","+str(featSelectionScheme)+","+str(nFeatures)+","+str(nSamples)+","+ str(itIx)+","+str(rngSeed)+","+str(outFoldIx)+","+str(inFoldIx)+","+str(mse)+","+str(rmse)+","+str(r2)+","+","+str(mape)+","+str(foldPairExecTime)+"\n")
						resFile.flush()
						
						inFoldIx = inFoldIx+1
						#append to results file outFile and flush
					outFoldIx = outFoldIx +1
					
					if TRAIN_TEST_MIX_INTEGRITY_CHECK:
						if len(outerTrainDF.index) != len(innerSampleTestCountMap):
							myio.logWrite("One of the samples in the inner CV was not part of a test set (expected count "+str(len(outerTrainDF.index))+" but was "+str(len(innerSampleTestCountMap))+"). Did you set an inner CV cluster size that was too small?",myio.LOG_LEVEL_ERROR)
						if len(outerTrainDF.index) != len(innerSampleTrainCountMap):
							myio.logWrite("One of the samples in the inner CV was not part of a train set (expected count "+str(len(outerTrainDF.index))+" but was "+str(len(innerSampleTrainCountMap))+") Did you set an inner CV cluster size that was too small?",myio.LOG_LEVEL_ERROR)
						
						expectedTrainParticipationCount = configDF[myio.NUMBER_INNER_FOLDS_KEY][eix]-1
						for k in innerSampleTrainCountMap:
							#make sure all samples were part of trainin equal 1 minus number of folds times
							if innerSampleTrainCountMap[k] !=  expectedTrainParticipationCount:
								myio.logWrite("One of the samples in the inner CV was not part of the training set every other train-test pair",myio.LOG_LEVEL_ERROR)
									#samples part of test set exactly once
						#samples part of test set exactly once
						for testIx in innerSampleTestCountMap:				
							if innerSampleTestCountMap[testIx]!=1:
								myio.logWrite("One of the samples in the inner CV was part of the test set more than once ("+str(innerSampleTestCountMap[testIx])+" times)",myio.LOG_LEVEL_ERROR)
						#samples part of train set nubmber of folds -1
						for trainIx in innerSampleTrainCountMap:				
							if innerSampleTrainCountMap[trainIx]!=inFoldIx - 1:
								myio.logWrite("One of the samples in the inner CV was not part of the train set number of folds ("+str(inFoldIx)+") -1 times",myio.LOG_LEVEL_ERROR)
				
				if TRAIN_TEST_MIX_INTEGRITY_CHECK:
					if outerNSamples != len(outerSampleTestCountMap):
						myio.logWrite("One of the samples in the outer CV was not part of a test set (expected count "+str(len(outerNSamples))+" but was "+str(len(outerSampleTestCountMap))+")",myio.LOG_LEVEL_ERROR)				
					if outerNSamples != len(outerSampleTrainCountMap):
						myio.logWrite("One of the samples in the outer CV was not part of a training set (expected count "+str(len(outerNSamples))+" but was "+str(len(outerSampleTrainCountMap))+")",myio.LOG_LEVEL_ERROR)				
					expectedTrainParticipationCount = outFoldIx-1
					for k in outerSampleTrainCountMap:
						#make sure all samples were part of trainin equal 1 minus number of folds times
						if outerSampleTrainCountMap[k] !=  expectedTrainParticipationCount:
							myio.logWrite("One of the samples in the outer CV was not part of the training set every other train-test pair",myio.LOG_LEVEL_ERROR)
				
				predFilePath = myio.parsePredictionFilePath(expDir,eix,itIx)
				
				
				
				#TAKE THE AVERAGE of every inner fold predictions for a outer fold
				avgInnerFoldPredictions=None
				nInnerFolds=configDF[myio.NUMBER_INNER_FOLDS_KEY][eix]
				for i in range (nInnerFolds):
					predInFoldColi="predicted_inner_fold"+str(i)
					if avgInnerFoldPredictions is None:
						avgInnerFoldPredictions=[]
						for j in range(len(iterationOutputDataMap[predInFoldColi])):
							avgInnerFoldPredictions.append(0)
						avgInnerFoldPredictions = np.array(avgInnerFoldPredictions)
					avgInnerFoldPredictions = avgInnerFoldPredictions+iterationOutputDataMap[predInFoldColi]
				avgInnerFoldPredictions = avgInnerFoldPredictions/nInnerFolds
					
				
				myio.logWrite("Average inner fold predictions length: "+str(len(avgInnerFoldPredictions)),myio.LOG_LEVEL_DEBUG)
				
				mycommon.listAppend(iterationOutputDataMap["predicted_inner_fold_average"],avgInnerFoldPredictions)
								
				
				#make a check that all columns in prediction file CSV output are same length (bug free means they all same length)
				lastColLen=-1
				for k in iterationOutputDataMap:
					
					if lastColLen != -1:
						#mis match of column lengths?
						if lastColLen !=len(iterationOutputDataMap[k]):
							myio.logWrite("Issue outputing prediction file of an iteration due to differign length of column. Column "+str(k)+" has length "+str(len(iterationOutputDataMap[k]))+" but expected "+str(lastColLen),myio.LOG_LEVEL_ERROR)
					else:
						lastColLen=len(iterationOutputDataMap[k])
						
				iterationPredictionDF = pd.DataFrame(data=iterationOutputDataMap)
				
				iterationPredictionDF = iterationPredictionDF.sort_values(by=[dataset.TIMESTAMP_COLUMN_NAME])#sort by timestamp (from early to later readings)	
	


				iterationPredictionDF.to_csv(predFilePath,sep=",",index=False,encoding='utf-8')#note that there will be a prediction for a sample id for each inner fold, so to chart the predictions, could take average prediction over a sample id
			
			if outputHyperParamsFlag:
				myio.logWrite("Finished writing hyperparameters for experiment "+str(eix)+", model "+algName+", to :"+ hyperParamOutFilePath,myio.LOG_LEVEL_DEBUG)
				hyperParamFile.close()
				hyperParamFile=None
			
			#final performance is avearage over each fold and iteration
			finalR2=finalR2/metricCounter
			finalMSE=finalMSE/metricCounter
			finalRMSE=finalRMSE/metricCounter
			finalMAPE=finalMAPE/metricCounter
			experimentEndTime = time.time()
			totalExperimentTime = experimentEndTime-experimentStartTime
			myio.logWrite("Experiment "+str(eix)+" finished after "+str(round(totalExperimentTime,2))+" seconds, algorithm "+algName+" performance R2 = "+str(round(finalR2,4))+", MSE = "+str(round(finalMSE,2))+", RMSE = "+str(round(finalRMSE,2))+", and MAPE = "+str(round(finalMAPE,2))+".",myio.LOG_LEVEL_INFO)
		#end of 1 experiment	
		except Exception as e:
			myio.logWrite("Aborting experiment "+str(eix)+" due to an exception: "+str(e)+"\n"+str(traceback.format_exc()),myio.LOG_LEVEL_ERROR)
			continue
	myio.logWrite("Finished ML experiments from experiment file "+inConfigFile+". Output files written to "+outDirPath+".",myio.LOG_LEVEL_INFO)
	
	myio.closeLogFile()
	resFile.close()
	#read in 
	pass
	

	
#returns a list of indices that are shuffled randomly or and shuffled via clutered sampling
def createRowIndexOrdering(splitType,df,subsetSize,nFolds,tensorNumberTimeSteps):

	if splitType == BLOCKED_CROSS_VAL_TYPE:		
		
		return temporalClusteredSampling(df,subsetSize,nFolds)
		
	elif splitType == RANDOM_CROSS_VAL_TYPE:
		
		
		return mycommon.randomRowIndexShuffle(df)		
	else:
		raise Exception("Unknown cross validation split type '"+str(splitType)+"'")

def populateCVOrderingList(cvSplitType,shuffledIndices):
	flattenedOuterShuffledIndices=None
	
	if cvSplitType == BLOCKED_CROSS_VAL_TYPE:
		
		#we now iterate over the folds, flatten them 
		#to a single list
			
		indexList = []
		
		for fold in shuffledIndices:
			mycommon.listAppend(indexList,fold)					
		
		
		flattenedOuterShuffledIndices=indexList
		#we will record the index/ordering of a sample after CV shuffling is applied in iteration output file
		
		
	elif cvSplitType == RANDOM_CROSS_VAL_TYPE:
		
		
		flattenedOuterShuffledIndices=shuffledIndices
		#we will record the index/ordering of a sample after CV shuffling is applied in iteration output file
		
		
	else:
		raise Exception("Unknown cross validation split type '"+str(cvSplitType)+"'")
	
	
	finalShuffleOrderList=[]
	#now we update the record of each sample with its position after shuffling
	#so its simply 0, 1, 2,... because the order of the results is relative to the shuffle, and then is sorted by timestamp
	for i in range(len(flattenedOuterShuffledIndices)):
		finalShuffleOrderList.append(i)
	
	return finalShuffleOrderList
				
				

#a generator function that returns the indices of test and train rows at each iteration of given dataframe
#splitType: type of split (random KF-CV or blocked CV using clusters as samples to populate folds)
#normDF: the dataset to split
#nFolds: number of folds in corss validation
#shuffledIndices: the list of indices the represent how we want to shuffle the dataset
def crossValSplit(splitType,normDF,nFolds,shuffledIndices):

	#dataframe was passed?
	if isinstance(normDF,pd.DataFrame):
	
		nRows=len(normDF.index)
	#number of rows was passed?
	elif isinstance(normDF,int):
		nRows = normDF
	else:
		raise Exception ("Cannot perform cross validation split. Invalid type for norm DF. Expected pandas dataframe or number of rows")
	
	myio.logWrite("Splitting input dataset of "+str(len(shuffledIndices))+" samples using: "+splitType,myio.LOG_LEVEL_DEBUG)
	
	if  len(shuffledIndices)==0:
		raise Exception("Cannot perform cross-validation split. Empty dataset")
	
	if nFolds <= 0:	
		raise Exception("Cannot perform cross-validation split. Illegal number of folds 0")
		
		
	
	if  len(shuffledIndices) < nFolds:
		raise Exception("Cannot have fewer samples ("+str(len(shuffledIndices))+") than number of folds ("+str(nFolds)+") in cross-validation split.")
	
	
	
	#samples per fold
	foldSize = math.ceil(len(shuffledIndices)/nFolds)
	
	#special case where the last fold will be empty given the rounding up of fold size?
	if foldSize * (nFolds-1) == len(shuffledIndices):
		#make last fold get overflow instead of being underflow as 0 samples
		foldSize = math.floor(len(shuffledIndices)/nFolds) 
	else:
		#make last fold have fewer samples (minimum one) than other folds
		pass
		
				
	#populate lists of indices for test and training samples		
	trainIxs=[]
	testIxs=[]
	
	
	if splitType == BLOCKED_CROSS_VAL_TYPE:
		
		sampleCount = 0
		
		for fold in shuffledIndices:
			sampleCount = sampleCount + len(fold)
						
		if len(shuffledIndices) > nRows:
			raise Exception("Cannot create train-test folds from dataset because more incides ("+str(len(shuffledIndices))+") are given than the dataset's size ("+nRows+"), and were assuming unique indices")
	
	
		#the shuffled indices in blocked cross val type are actually folds
		folds = shuffledIndices	
		
		#create each train-test fold pair
		for  testFoldIx in range(nFolds):
		
			trainIxs.clear()
			testIxs.clear()
						
				
			#put all indices in appropriate set
			for i in range(len(folds)):
			
				if i == testFoldIx: #test fold?
					mycommon.listAppend(testIxs,folds[i]) 
				else: #training fold
					mycommon.listAppend(trainIxs,folds[i]) 
						
			myio.logWrite("Resulting test indices sizes (train,test): ("+str(len(trainIxs))+","+str(len(testIxs))+").",myio.LOG_LEVEL_DEBUG)
			
			if len(testIxs)==0 or len(trainIxs)==0:
				myio.logWrite("Warning, a fold was empty. Skipping fold "+str(testFoldIx),myio.LOG_LEVEL_WARNING)
				continue
				
			yield trainIxs,testIxs
			
	elif splitType == RANDOM_CROSS_VAL_TYPE:
	
		#in this case shuffledIndices is just list of indices
				
		if len(shuffledIndices) > nRows:
			raise Exception("Cannot create train-test folds from dataset because more incides ("+str(len(shuffledIndices))+") are given than the dataset's size ("+nRows+"), and were assuming unique indices")
	
	
		testIxLowBound=0
		testIxUpperBound=foldSize
		#create each train-test fold pair
		for  foldIx in range(nFolds):		
			trainIxs.clear()
			testIxs.clear()
			
			#last fold?
			if foldIx == (nFolds-1):
				lastFoldFlag=True
			else:	
				lastFoldFlag=False
			
			#iterate over each sample to put index in appropriate list
			for j in range(len(shuffledIndices)):
				if j >=testIxLowBound and (lastFoldFlag or j <testIxUpperBound): #make sure to include all remaining samples for last fold
					testIxs.append(shuffledIndices[j])
				else:
					trainIxs.append(shuffledIndices[j])
			
			myio.logWrite("Resulting test indices sizes (train,test): ("+str(len(trainIxs))+","+str(len(testIxs))+").",myio.LOG_LEVEL_DEBUG)
			
			if len(testIxs)==0 or len(trainIxs)==0:
				myio.logWrite("Warning, a fold was empty. Skipping fold "+str(foldIx),myio.LOG_LEVEL_WARNING)
				#move the test indices window over 1 fold
				testIxLowBound=testIxUpperBound
				testIxUpperBound=testIxUpperBound+foldSize #note that last fold may be smaller if dataset isn't perfectly divisible by # folds
				
				continue
				
			yield trainIxs,testIxs
			
			#move the test indices window over 1 fold
			testIxLowBound=testIxUpperBound
			testIxUpperBound=testIxUpperBound+foldSize #note that last fold may be smaller if dataset isn't perfectly divisible by # folds

			
	else:
		raise Exception("Unknown cross validation split type '"+str(splitType)+"'")
	

def tuneBasicModelHyperParameters(algName,temporalResolution,tensorNumberTimeSteps,trials,numExecutionsPerTrial,inputShape,innerTrain_X,innerTrain_y,innerVal_X,innerVal_y):
	
	
	#randomly generate the set of random hyperparameters
	hyperParamSet = model.generateHyperparameterSets(algName,temporalResolution,tensorNumberTimeSteps,trials,inputShape,innerTrain_X.shape[0],innerVal_X.shape[0])
	
	#now do hyperparameter tuning
	bestR2=None
	bestMSE=None						
	bestHyperParamSet = None
	
	#train and evaluate model performance for many different sets of hyperparameters
	#keeping tracl of set of hyperparameters that led to best performance (higher R2 and lowest MSE)
	for hyperParams in hyperParamSet:
		r2 = 0
		mse=0
		#create a model usign same set of hypermeters multiple times
		#to average out effect of randomness from model creation (e.g., model weight assignments)
		#on model perfromance
		for i in range(numExecutionsPerTrial):
			#create the model
			mlModel = model.Model(algName,hyperParams,inputShape)
			
			#this would be another potential area where tensor length defined. Would need the timestamps column and dataset, but otherwise
			#could change size of tensors here before feeding to model
			
			#train model
			mlModel.fit(innerTrain_X,innerTrain_y,innerVal_X,innerVal_y)
			
			preds = mlModel.predict(innerVal_X)
			r2 = r2 + mycommon.computeR2(innerVal_y,preds)
			
			#don't unnormalize here because the scale of value isn't important, its the relative size of mse compared to other mse results
			mse = mse + mycommon.computeMSE(innerVal_y,preds)  
		
		r2 = r2/numExecutionsPerTrial
		mse = mse/numExecutionsPerTrial
		#first hyperparameter set?
		if bestR2 is None:
			bestR2=r2
			bestMSE=mse
			bestHyperParamSet=hyperParams
		elif r2>bestR2 and mse <bestMSE: # a trial is only considered with better performance if both R2 is increased and MSE is decreased
			bestR2=r2
			bestMSE=mse
			bestHyperParamSet=hyperParams
			
	return bestHyperParamSet



#given a set of hyperparameters, will append a row to an output file in the form of 
#and will ccreate a header when createHeaderFlag = True:
#Header: iteration, outer fold, inner fold, <hyper param 1 name>, <hyper param 2 name>, ..., <hyper param n name>
#row values:<ITERATION>,<OUTER FOLD>, <INNER FOLD>, <HYPER PARAM 1 VALUE>, <HYPER PARAM 2 VALUE>, ..., <HYPER PARAM N VALUE>
def outputHypeParams(bestHyperParamSet,createHeaderFlag,hyperParamFile,itix,outFoldIx,inFoldIx):
	#first time writing to file?
	if createHeaderFlag:		
		#create the CSV header							
		hyperParamFileHeader= "iteration,outer fold, inner fold"
		
		#add all hyperparameter names to header
		for paramName in bestHyperParamSet:
			hyperParamFileHeader = hyperParamFileHeader+","+paramName
		
		hyperParamFileHeader=hyperParamFileHeader+"\n"
			
		hyperParamFile.write(hyperParamFileHeader)
	
	#add the hyperparam values entry to output file
	rowOut = str(itix)+","+str(outFoldIx)+","+str(inFoldIx)
	for paramName in bestHyperParamSet:
		hyperParamValue=bestHyperParamSet[paramName]
		rowOut = rowOut+","+str(hyperParamValue)
	
	rowOut=rowOut+"\n"
	hyperParamFile.write(rowOut)
	hyperParamFile.flush()
		
		
#clusters the samples by those nearest in time, shuffles the clusters, and the create folds by sampling clustesrs and returns the folds (set of index lists representing the folds)
#df dataframe to split into multiple dataframes each with samples withint the temporal resolution  subsetSize
#subsetSize (in hours) the size of the window used to split datasets (note the window is sliding. The datasubsets don't overlap)
def temporalClusteredSampling(df,subsetSize,nFolds):

	#create a dataframe tmpDF that has the timestamps of the original dataframe and the row index of the timestamp
	sampleIxs = []
	for i in range(len(df.index)):
		sampleIxs.append(i)
	
	internalIndexColName="index"
	tmpMap = {internalIndexColName:sampleIxs,dataset.TIMESTAMP_COLUMN_NAME:df[dataset.TIMESTAMP_COLUMN_NAME]}
	tmpDF = pd.DataFrame(data=tmpMap)
	
	
	#list of datasets clustered by subset size
	dfClusters = []
	
	#sort by timestamp (from early to later readings) and put and index over the timestamp for searching
	tmpDF = tmpDF.sort_values(by=[dataset.TIMESTAMP_COLUMN_NAME])
	tmpDF[dataset.TIMESTAMP_COLUMN_NAME] = pd.to_datetime(tmpDF[dataset.TIMESTAMP_COLUMN_NAME])
	tmpDF.set_index(dataset.TIMESTAMP_COLUMN_NAME,inplace=True)

	finishedFlag=False
	wStart = tmpDF.index[0] #starting of the window is first timestamp of dataset
	wEnd = wStart+ timedelta(hours=int(subsetSize))#ending of the window
	while not finishedFlag:

		#get all samples within time window [tStart,tEnd) in sensor data				
		#get an index of all samples withint lag window
		windowFilter = (tmpDF.index>= wStart) & (tmpDF.index< wEnd)
		
		#get the samples in window
		subset = tmpDF.loc[windowFilter]
		
		dfClusters.append(subset)
		
		
		#increment window by 1
		wStart = wEnd #starting of the window
		wEnd = wEnd + timedelta(hours=int(subsetSize))#ending of the window
		
		#the window is outside the timestamp range of the dataset?
		if wStart >tmpDF.index[-1]:
			finishedFlag=True
		

	#the cluster at the moment are dataframes, convert to list of indexes
	clusters=[]
	for clusterDF in dfClusters:
		clusters.append(clusterDF[internalIndexColName])
		
	#shuffle the list of clusters
	random.shuffle(clusters)
	
	
			
	#the number of clusters per fold
	foldNClusters = int(math.floor(len(clusters)/nFolds))
	
	
	myio.logWrite("number of clusters per fold: "+str(foldNClusters),myio.LOG_LEVEL_DEBUG)
	myio.logWrite("number clusters: "+str(len(clusters)),myio.LOG_LEVEL_DEBUG)
	
	cix = 0
	fold = []
	folds = []
	#create the folds by iterating over each cluster
	while(cix < len(clusters)):
		
		
		c=clusters[cix]
		
		#put the samples of the cluster in the fold
		mycommon.listAppend(fold,c)
		
		
		#increment number of clusters added to the curent fold
		cix = cix+1
		
		#finised populating a fold?
		if (cix % foldNClusters) == 0:
			#we only create a new fold and append resulting fold to list  
			#for folds before last fold. The last fold may get some overflow
			#if number of clusters can't be perfectly divided by number of folds
			if len(folds) < (nFolds-1):		
				folds.append(fold)
				fold=[]
							
	if len(fold)>0:
		#print("finished creating fold dataframe (out loop)")					
		folds.append(fold)
		
	return folds
	
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
	
	runSingleDatsetCVExperiments(inputConfigFile,outDir)
		