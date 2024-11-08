import shutil

import numpy as np
import os
from datetime import datetime, timedelta
import time

SINGLE_DATASET_CV_CONFIG_FILE_TYPE=0
TRAIN_TEST_DATASET_CONFIG_FILE_TYPE=1
FEDERATED_LEARNING_CONFIG_FILE_TYPE=2

configTypeVerboseMap=["single dataset cross-validation experiments config type","supplied train and test dataset experiments config type","single dataset federated learning cross-validation experiment config file"]

EXP_OUT_SUB_DIR_NAME_DATE_PREFIX="%Y-%m-%d_%H_%M_%S"
LOG_FILE_NAME="log"
RESULTS_FILE_NAME="results.csv"
EXPERIMENT_SUB_DIR_PREFIX="e"#
CLIENT_SUB_DIR_PREFIX="c"
EXPERIMENT_SUB_DIR_NAME="experiments"#
CLIENTS_SUB_DIR_NAME="clients"
ITERATION_PREDICTIONS_FILE_NAME_PREFIX="i"
HYPERPARAM_CHOICE_FILE_NAME="hyperparameter-choices.csv"

INPUT_DATASET_PATH_KEY="input dataset path"
INPUT_TRAIN_DATASET_PATH_KEY="input train dataset path"
INPUT_TEST_DATASET_PATH_KEY="input test dataset path"
YEAR_KEY="year"
CHAMBER_KEY="chamber"

TRAIN_DATASET_YEAR_KEY="train dataset year"
TRAIN_DATASET_CHAMBER_KEY="train dataset chamber"
TEST_DATASET_YEAR_KEY="test dataset year"
TEST_DATASET_CHAMBER_KEY="test dataset chamber"

TEMPORAL_RESOLUTION_KEY="temporal resolution (min)"
FEATURE_SELECTION_SCHEME_KEY="feature selection scheme"
INPUT_TENSOR_NUMBER_TIME_STEPS_KEY="input tensor number of time steps"#1 for basic ML (one sample per timestamp), and DNN, and > 1 for LSTM and CNN that have time dimension in input (many samples for many timestamps in a single isntance)
RNG_SEED_KEY="seed"
SELECTED_SENSOR_PATH_KEY="selected sensors path"
APPLY_MIN_MAX_SCALING_FLAG_KEY="apply min-max scaling"
ALGORITHM_KEY="algorithm"
ITERATIONS_KEY="iterations"
OUTER_CV_SPLIT_TYPE_KEY="outer CV split type"
NUMBER_OUTER_FOLDS_KEY="number of outer folds"
INNER_CV_SPLIT_TYPE_KEY="inner CV split type"
NUMBER_INNER_FOLDS_KEY="number of inner folds"
NUMBER_OF_TRIALS_KEY="number of trials"
TEMPORAL_OUTER_CV_CLUSTER_SIZE_KEY="clustered outer-CV cluster size (hours)"
TEMPORAL_INNER_CV_CLUSTER_SIZE_KEY="clustered inner-CV cluster size (hours)"
NUMBER_OF_EXECUTIONS_PER_TRIAL_KEY = "number of executions per trial"
OUTPUT_HYPERPARAMETER_CHOICE_FLAG_KEY= "output hyperparameter choice"


FEDERATED_LEARNING_EXPERIMENT_TYPE_KEY="FL experiment type"
FEDERATED_LEARNING_CLIENT_HETEROGENEITY_KEY="client heterogeneity"
FEDERATED_LEARNING_NUMBER_OF_CLIENTS_KEY = "number of clients"
FEDERATED_LEARNING_EPOCH_OVERRIDE_KEY="epoch override"
FEDERATED_LEARNING_BATCH_SIZE_OVERRIDE_KEY="batch size override"
FEDERATED_LEARNING_LEARNING_RATE_OVERRIDE_KEY="learning rate override"
FEDERATED_LEARNING_CROSS_VALIDATION_TYPE_KEY="cross-validation type"
FEDERATED_LEARNING_BLOCK_CV_CLUSTER_SIZE_KEY="block CV cluster size (hours)"
FEDERATED_LEARNING_NUMBER_OF_FOLDS_KEY="number of folds"
FEDERATED_LEARNING_NUM_SEL_CLIENTS_PER_ROUND_KEY="number of selected clients per round"
FEDERATED_LEARNING_NUMBER_OF_ROUNDS_KEY="number of rounds"
FEDERATED_LEARNING_AGGREGATOR_KEY="federated aggregation algorithm"
#map that both types of experiments share 
configCommonFieldTypeMap={}

#array for looking up map using ID of configuration experiment type (SINGLE_DATASET_CV_CONFIG_FILE_TYPE vs. TRAIN_TEST_DATASET_CONFIG_FILE_TYPE)
configFieldTypeMaps=[{},{},{}]

optionalConfigEntryMap={}


#the entries that both types of experiments share
#populate the expected type for each field that should be in config file

configCommonFieldTypeMap[RNG_SEED_KEY]=np.int64
configCommonFieldTypeMap[SELECTED_SENSOR_PATH_KEY]=str
configCommonFieldTypeMap[ALGORITHM_KEY]=str
configCommonFieldTypeMap[ITERATIONS_KEY]=np.int64
configCommonFieldTypeMap[INNER_CV_SPLIT_TYPE_KEY]=str
configCommonFieldTypeMap[NUMBER_INNER_FOLDS_KEY]=np.int64
configCommonFieldTypeMap[NUMBER_OF_TRIALS_KEY]=np.int64
configCommonFieldTypeMap[TEMPORAL_INNER_CV_CLUSTER_SIZE_KEY]=np.int64
configCommonFieldTypeMap[APPLY_MIN_MAX_SCALING_FLAG_KEY]=np.bool_


configCommonFieldTypeMap[TEMPORAL_RESOLUTION_KEY]=np.int64
configCommonFieldTypeMap[FEATURE_SELECTION_SCHEME_KEY]=str
configCommonFieldTypeMap[INPUT_TENSOR_NUMBER_TIME_STEPS_KEY]=np.int64
configCommonFieldTypeMap[NUMBER_OF_EXECUTIONS_PER_TRIAL_KEY]=np.int64
configCommonFieldTypeMap[OUTPUT_HYPERPARAMETER_CHOICE_FLAG_KEY]=np.bool_

#the optional config entries
optionalConfigEntryMap[OUTPUT_HYPERPARAMETER_CHOICE_FLAG_KEY]=None

#make sure both types of maps get populated with common entreis
for k in configCommonFieldTypeMap:
	
	configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][k]=configCommonFieldTypeMap[k]
	configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][k]=configCommonFieldTypeMap[k]
	
	#don't include federated learning here since it has different input fields comapred to centralized leanring
	#configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][k]=configCommonFieldTypeMap[k]
	
	

#the entries unique to cross-validation and train/test file experiments
#populate the expected type for each field that should be in config file

#only CV experiments have outter fold entries
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][NUMBER_OUTER_FOLDS_KEY]=np.int64
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][TEMPORAL_OUTER_CV_CLUSTER_SIZE_KEY]=np.int64
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][INPUT_DATASET_PATH_KEY]=str
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][OUTER_CV_SPLIT_TYPE_KEY]=str
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][YEAR_KEY]=str
configFieldTypeMaps[SINGLE_DATASET_CV_CONFIG_FILE_TYPE][CHAMBER_KEY]=str

#only train/test datset file experiments have a file for train and file for test
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][INPUT_TRAIN_DATASET_PATH_KEY]=str
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][INPUT_TEST_DATASET_PATH_KEY]=str
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][TRAIN_DATASET_YEAR_KEY]=str
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][TRAIN_DATASET_CHAMBER_KEY]=str
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][TEST_DATASET_YEAR_KEY]=str
configFieldTypeMaps[TRAIN_TEST_DATASET_CONFIG_FILE_TYPE][TEST_DATASET_CHAMBER_KEY]=str

#federated earning fields
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][INPUT_DATASET_PATH_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][SELECTED_SENSOR_PATH_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][YEAR_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][CHAMBER_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][TEMPORAL_RESOLUTION_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][ALGORITHM_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][INPUT_TENSOR_NUMBER_TIME_STEPS_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_AGGREGATOR_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_EXPERIMENT_TYPE_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][APPLY_MIN_MAX_SCALING_FLAG_KEY]=np.bool_
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_CLIENT_HETEROGENEITY_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_LEARNING_RATE_OVERRIDE_KEY]=np.float64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_EPOCH_OVERRIDE_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_BATCH_SIZE_OVERRIDE_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][RNG_SEED_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_CROSS_VALIDATION_TYPE_KEY]=str
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_BLOCK_CV_CLUSTER_SIZE_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][ITERATIONS_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_NUMBER_OF_FOLDS_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_NUMBER_OF_CLIENTS_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_NUM_SEL_CLIENTS_PER_ROUND_KEY]=np.int64
configFieldTypeMaps[FEDERATED_LEARNING_CONFIG_FILE_TYPE][FEDERATED_LEARNING_NUMBER_OF_ROUNDS_KEY]=np.int64








#LOG STUFF

LOG_LEVEL_DEBUG=0
LOG_LEVEL_INFO=1
LOG_LEVEL_WARNING=2
LOG_LEVEL_ERROR=3
LOG_LEVEL_STRINGS=["DEBUG","INFO","WARNING","ERROR"]
GLOBAL_LOG_LEVEL=LOG_LEVEL_INFO




globalLogFile = None


#makes sure all expected columns  are present in configuration file dataset
def configFileIntegrityCheck(df,configType):
	configFieldTypeMap=configFieldTypeMaps[configType]
	for k in configFieldTypeMap:
		fieldType = configFieldTypeMap[k]
		
		#optional entry?
		if k in optionalConfigEntryMap:
			continue
			
		#there is a column missing from config file?
		if not k in df:
			raise Exception("Malformed configuration file ("+configTypeVerboseMap[configType]+"). Missing column. Expected '"+k+"' field, but column was not present.")
	



#makes sure all field values are correct in the configuration file dataset
def configEntryIntegrityCheck(eix,df,configType):
	configFieldTypeMap=configFieldTypeMaps[configType]
	for k in configFieldTypeMap:
		fieldType = configFieldTypeMap[k]
		
		#optional entry?
		if k in optionalConfigEntryMap:
			continue
		
		#for str type, any value is acceptable, so only check for non-str
		if fieldType!= str:
			#value of field is wrong type?
			if not isinstance(df[k][eix],fieldType):
				raise Exception("Malformed configuration file ("+configTypeVerboseMap[configType]+"). Incorrect variable format. Expected field '"+k+"' of type '"+str(fieldType)+"' but was of type '"+str(type(df[k][eix]))+"'.")
		
		

#create a direcotyr outDir
#in the directory outDir, create a direction with name a date (yyyy-mm-dd_hh_mm_ss) with a log file underneat that will log basic stuff like "runing experiment i for ML model RF at time yyyy-mm-dd hh:mm:ss"
#will create an output file  named results.csv in the directory named by date to append all results.
#Create subfolder under the date folder called experiments
#for each experiment id create a sub folder in experiments called e<expereimntid>, and for each iteration create predictin output file called i<iteration>.csv for appropriate experiment id sub folder
# example:
#>/home/user/n2o/2024-ml/2024-03-11_12_37_42/
#											> log
#											>results.csv
#											>experiments/
#												>e0/
#													>i0.csv
#													>i1.csv
#													>...
#												>e1/
#													>i0.csv
#													>i1.csv
#													>...
#												>...
#parentDir: parent directory to store all output files of experiments
#numExperiments: number of experiments to create sub directories for
def setupOutputDirectory(inConfigFile,parentDir, numExperiments):
	#get current time
	now = datetime.now()
	#convert time to appropriate format (year-month-day_hour_minute_second)
	strTime=now.strftime(EXP_OUT_SUB_DIR_NAME_DATE_PREFIX)
	
	
	#create subdirectory named after time of creation
	outPath = os.path.join(parentDir,strTime)
			
	#output directory name already exists?
	if os.path.exists(outPath):
		#keep trying a new directory name by counting
		#until find a free name		
		dirNameCounter=1
		tmpOutPath=outPath+"("+str(dirNameCounter)+")"		
		
		while os.path.exists(tmpOutPath):
			dirNameCounter = dirNameCounter +1
			tmpOutPath=outPath+"("+str(dirNameCounter)+")"		
			
		#found available name
		outPath=tmpOutPath
			
	#create output directory
	os.mkdir(outPath)	
	
	shutil.copy(inConfigFile,outPath) #make a copy of input configuration file that ran this experiment
	
	logFilePath =os.path.join(outPath,LOG_FILE_NAME)
	
	
	resFilePath =os.path.join(outPath,RESULTS_FILE_NAME)
	
	expDir=os.path.join(outPath,EXPERIMENT_SUB_DIR_NAME)
	
	os.mkdir(expDir)
	#create all sub directories for each experiment
	
	for i in range(numExperiments):
		expDirName =EXPERIMENT_SUB_DIR_PREFIX+str(i) 
		expSubDirPath = os.path.join(expDir,expDirName)
		
		os.mkdir(expSubDirPath)				
	

	return logFilePath,resFilePath,expDir,outPath



#create a direcotyr outDir
#in the directory outDir, create a direction with name a date (yyyy-mm-dd_hh_mm_ss) with a log file underneat that will log basic stuff like "runing experiment i for ML model RF at time yyyy-mm-dd hh:mm:ss"
#will create an output file  named results.csv in the directory named by date to append all results.
#Create subfolder under the date folder called experiments
#for each experiment id create a sub folder in experiments called e<expereimntid>, and for each iteration create predictin output file called i<iteration>.csv for appropriate experiment id sub folder
# example:
#>/home/user/n2o/2024-ml/2024-03-11_12_37_42/
#											> log
#											>results.csv
#											>experiments/
#												>e0/ #experiment 0
#													>clients
#														>c0/   #the client 0
#															>i0.csv
#															>i1.csv
#															>...
#														>c1/   # client 1
#															>i0.csv
#															>i1.csv
#															>...
#														>...
#												>e1/ #experiment 1
#													>clients
#														>c0/   #the client 0
#															>i0.csv
#															>i1.csv
#															>...
#														>c1/   # client 1
#															>i0.csv
#															>i1.csv
#															>...
#														>...
#												>...
#given an experiment directory: e.g., experiments/e0/clients
#setup the directory as
#														>c0/   #the client 0
#															>i0.csv
#															>i1.csv
#															>...
#														>c1/   # client 1
#															>i0.csv
#															>i1.csv
#															>...
#														>...
#													>i0.csv
#													>i1.csv
#													>...	
#for each client
#eix experiment id
def setupFedLearnClientOutputDirectory(expDir,eix,numClients):
	#create client subdirectories for each client
	expSubDir= parsePredictionFileDir(expDir,eix)
	
	clientsSubDirPath = os.path.join(expSubDir,CLIENTS_SUB_DIR_NAME)
	os.mkdir(clientsSubDirPath)		
	
	for cix in range(numClients):
		clientDirName =CLIENT_SUB_DIR_PREFIX+str(cix) 
		clientSubDirPath = os.path.join(clientsSubDirPath,clientDirName)
		
		os.mkdir(clientSubDirPath)		
	
#expDir: experiments output directory path
#expNum: experiment id
#itNum: iteration
def parsePredictionFilePath(expDir,expNum,itNum):	
	expDirPath = parsePredictionFileDir(expDir,expNum)
	predFileName = ITERATION_PREDICTIONS_FILE_NAME_PREFIX+str(itNum) 
	predFilePath = os.path.join(expDirPath,predFileName)
	
	return predFilePath+".csv"

#expDir: experiments output directory path
#expNum: experiment id
#itNum: iteration
#clientid: client id
def parseClientPredictionFilePath(expDir,expNum,clientid,itNum):	
	#.../experiments/e<expNum>
	expSubDir= parsePredictionFileDir(expDir,expNum)

	#.../experiments/e<expNum>/clients
	clientsSubDirPath = os.path.join(expSubDir,CLIENTS_SUB_DIR_NAME)
	
	#c<clientid>
	clientDirName =CLIENT_SUB_DIR_PREFIX+str(clientid) 
	
	#.../experiments/e<expNum>/clients/c<clientid>
	clientSubDirPath = os.path.join(clientsSubDirPath,clientDirName)
	
	#i<itNum>
	predFileName = ITERATION_PREDICTIONS_FILE_NAME_PREFIX+str(itNum) 
	
	#.../experiments/e<expNum>/clients/c<clientid>/i<itNum>
	predFilePath = os.path.join(clientSubDirPath,predFileName)
	
	return predFilePath+".csv"
	
#expDir: experiments output directory path
#expNum: experiment id
def parseHyperParamOutFilePath(expDir,expNum):	
	expDirPath = parsePredictionFileDir(expDir,expNum)	
	hyperParamOutPath = os.path.join(expDirPath,HYPERPARAM_CHOICE_FILE_NAME)
	
	return hyperParamOutPath

	
#expDir: experiments output directory path
#expNum: experiment id
def parsePredictionFileDir(expDir,expNum):
	expDirName =EXPERIMENT_SUB_DIR_PREFIX+str(expNum) 
	expDirPath = os.path.join(expDir,expDirName)
	return expDirPath
	
#expDir: experiments output directory path
#expNum: experiment id
def parseDeepLearningHyperParamTuneOutDir(expDir,expNum):
	expDirPath = parsePredictionFileDir(expDir,expNum)
	
	logDir2 = f"{int(time.time())}"
	logDir2 = "deep-learn-hyperparam-tune-out-"+logDir2
	return os.path.join(expDirPath,logDir2)

	
def openLogFile(logFilePath):
	#here were telling python than were not defining a local variable, instead were assigning a value to the global log variable so that other functions can 
	#log things without needing to pass the log reference everytwhere (for clarity reasons)
	global globalLogFile 
	
	globalLogFile = open(logFilePath,"a")
	

def logWrite(msg,level):
	if globalLogFile is None:
		print("error, cannot write to null log file. Maybe you forget to initiate 'globalLogFile'?")
		return
		
	#only print messages of ateleast priority of global logging level
	if level < GLOBAL_LOG_LEVEL:
		return
	
	
	now = datetime.now()
	#convert time to appropriate format (year-month-day_hour_minute_second)
	strTime=now.strftime("%Y-%m-%d %H:%M:%S")
	outputMsg="["+LOG_LEVEL_STRINGS[level]+"]["+strTime+"]: "+msg
	print(outputMsg)
	globalLogFile.write(outputMsg+"\n")
	
def flushLogFile():
	globalLogFile.flush()
	
def closeLogFile():
	globalLogFile.close()
	