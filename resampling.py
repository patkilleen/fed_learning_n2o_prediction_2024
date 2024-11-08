
import glob
import os
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np
import math
import argparse

#indexFilePath: the input csv file path
#outputFilePath
#newTemporalRes: number of minutes for new samples size (e.g., 30 for 30 minutes)
#aggOp: mean, max, min,sum,  last, first
def resampleCSVTimeSeries(indexFilePath,outputFilePath,newTemporalRes,aggOp="mean"):
	#read the CSV file
	df = pd.read_csv(indexFilePath, sep=",")

	#sort by timestamp (from early to later readings)
	df = df.sort_values(by=['timestamp'])

	#convert the timestamp column to a datetime format
	df['timestamp'] = pd.to_datetime(df['timestamp'])

	#set the index of dataframe to the timestamp
	df.set_index('timestamp',inplace=True)

	resampler=df.resample(str(newTemporalRes)+'min')
	if aggOp=="mean":
		df = resampler.mean()
	elif aggOp=="max":
		df = resampler.max()
	elif aggOp=="min":
		df = resampler.min()
	elif aggOp=="sum":
		df = resampler.sum()
	elif aggOp=="last":
		df = resampler.last()
	elif aggOp=="first":
		df = resampler.first()
	else:
		print("unknown aggregation operation: "+aggOp)
		return	

	df.to_csv(outputFilePath,sep=",",index=True,encoding='utf-8')
	
	
def mergeCSVFilesOnTimeStamp(searchDirPath,outputCSVPath):
	os.chdir(searchDirPath)
	csvFiles = glob.glob('*.csv')
	
	
	aggDF = None
	for csvPath in csvFiles:
		#read the CSV file
		df = pd.read_csv(csvPath, sep=",")


		#convert the timestamp column to a datetime format
		df['timestamp'] = pd.to_datetime(df['timestamp'])

		#set the index of dataframe to the timestamp
		df.set_index('timestamp',inplace=True)
		
		#first dataset?
		if aggDF is None:
			aggDF=df
		else:
			aggDF =pd.concat([aggDF,df],axis=1)
	
	#save the merged csv file
	aggDF.to_csv(outputCSVPath,sep=",",index=True,encoding='utf-8')


#merges dataframes on a shared column, where merge column values that
#are not identical will instead use the next most similar value to merge
#searchDirPath: dirrectoyr containing all .csv files to merge 
#outputCSVPath: output fil path of dataset resulting from merge
#mergeColName: column name to merge files on
def mergeDFOnColumn(inputCSV1Path,inputCSV2Path, mergeColName,outCSVPath):
	df1 = pd.read_csv(inputCSV1Path, sep=",")
	df2 = pd.read_csv(inputCSV2Path, sep=",")
	
	#integrity checks
	for df in [df1,df2]:
	
		#missing merge key column?
		if not (mergeColName in df):
			
			raise Exception("Cannot merge dataframes on a column, since one of the dataframes doesn't have the column: "+str(mergeColName))
	
	
	resMap = {}
	for col in df2.columns:
		resMap[col]=[]
				
		
	#sort dataframes by merge Col name
	df2 = df2.sort_values(by=[mergeColName])
	#iterate over every sample of df1
	for i in range(len(df1.index)):
		val = df1[mergeColName][i]
		
		#find row index with most similar key value
		ix = _findRowIndex(mergeColName,val,df2)
		
		#add all values of row of df2 to result mpa
		for col in df2.columns:
			resMap[col].append(df2[col][ix])
	
	#append all the columns to df1
	for col in df2.columns:
		df1.insert(len(df1.columns),col,resMap[col],True)
	
	df1.to_csv(outCSVPath,sep=",",index=True,encoding='utf-8')



	
#given a column name as key, and a lookup value, finds the first row index
#in a dataframe where the key shares the same value, 
#or finds the row with the most similar value
def _findRowIndex(key,val,df):	
	for i in range(len(df.index)):
		
		#havne't found the value yet?
		if df[key][i]<val:
			continue
		elif df[key][i]==val: #found the value
			return i
		else:
			#we got to the poitn where the preevious value was smaller
			#but current value is bigger, so choose the most similar value
			
			#special case where its first value?
			if i ==0:
				return i
			else:
				diff1=val-df[key][i-1]
				diff2=df[key][i]-val
				
				#previous value closer to lookup value?
				if diff1<diff2:
					return i-1
				else:
					return i
	
	
	return len(df.index)-1
#window sizes are in hours, so [3,6,9] would mean 3 lagged feature created for each sensor feature (3h mean, 6h mean, 9h mean)
#offsetSizes are the offset of each window before applying aggregation over window, so [24,48,72] would mean a window  of size 3 hours applied 24 hours earlier, 6 h window applied 2 days
# earlier, and 9 h window applied 3 days earlier for window size [3,6,9]
#newTemporalRes: in minutes, the n2o dataset resampled resolution (e.g., 60 = 1 hour)
def extractLaggedFeatures(n2oDatasetInPath,sensorDatasetInPath,outputPath, newTemporalRes,windowSizes,offsetSizes=None):

	#keep in mind the window sizes should be defined based on target resolution. a target res of daily, with (3h mean, 6h mean, 9h mean) windows would miss more than
	#half the start of the day

	#read the n2o data CSV file
	originalN2ODF = pd.read_csv(n2oDatasetInPath, sep=",")
	
	originalN2ODF = originalN2ODF.sort_values(by=['timestamp'])#sort by timestamp (from early to later readings)	
	originalN2ODF['timestamp'] = pd.to_datetime(originalN2ODF['timestamp'])#convert the timestamp column to a datetime format	
	originalN2ODF.set_index('timestamp',inplace=True)#set the index of dataframe to the timestamp

	#resample to target resolution (e.g., 30 mins, hourly, daily) via mean, where have multiple n2o columsn for each chamber
	#we resample to be able to iterate over timestamps of desired resolution
	#to later manually take average over originalN2ODF
	n2o_df = originalN2ODF.resample(str(newTemporalRes)+'min').mean()#func call options: mean,sum,max,min  #
	
	#sort again after resampling to make sure the mean of older values taken to form later values
	#n2o_df.sort_index(inplace=True,ascending=True)#sort by timestamp/index (from early to later readings)	 

	#read in sensord dataset
	
	#read the sensor data CSV file
	sensor_df = pd.read_csv(sensorDatasetInPath, sep=",")
	
	sensor_df = sensor_df.sort_values(by=['timestamp'])#sort by timestamp (from early to later readings)	
	sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])#convert the timestamp column to a datetime format	
	sensor_df.set_index('timestamp',inplace=True)#set the index of dataframe to the timestamp

	
	#create ag op dictionary, where part column name indicates
	#what aggregatio noperator to use  when creating lagged features
	aggs = {}
	#for each column name in sensor dataset
	for cName in sensor_df.columns:
		if "_avg" in cName:
			aggs[cName]='mean'
		elif "_max" in cName:
			aggs[cName]='max'
		elif "_min" in cName:
			aggs[cName]='min'
		elif "_sum" in cName:
			aggs[cName]='sum'
		elif "_time" in cName:
			aggs[cName]='sum'
		else:
			aggs[cName]='mean'
			print("unknown operation for column '"+cName+"', using mean as default")
			
	#create pandas dataframe for each window
	#laggedDataFrames={} #TODO: implement
	
	#create our resulting dataframe and make sure name all columns appropriatly based on lag/window size
	resSensorDF = None
	resN2ODF = None
	
	offsetIx=0
	windowOffset=0
	#########################
	#			Sensor data
	#########################
	#for each window size w in windowSizes:
	for w in windowSizes:
	
		if not offsetSizes is None:
			windowOffset = offsetSizes[offsetIx]
			offsetIx = offsetIx + 1
			
		lagColNames=[] #column names for lagged feature of window size w will be of format: <colName>_Lag<w>H
		lagFeatureValueLists={}
		n2oColsValueLists={}
		colNameMap = {}
		#lagFeatureValueLists['timestamp']=[]
		for cName in sensor_df.columns:
		
			if offsetSizes is None:
				newCName = cName+"_Lag"+str(w)+"H"
			else:
				newCName = cName+"_Lag"+str(w)+"H_Offset_"+str(windowOffset)+"H"
				
			lagColNames.append(newCName)
			colNameMap[cName]=newCName
			lagFeatureValueLists[newCName]= []
		
		for cName in n2o_df.columns:
			n2oColsValueLists[cName]= []
			
		#timestamps = []
		
		#for every timestamp  t in n2o dataset :
		for t in n2o_df.index:
			#get all samples within time window [t-w,t] in sensor data
			
			minTimeBound =t - timedelta(hours=(w+windowOffset)) #remove w hours from timestamp t
			maxTimeBound = t- timedelta(hours=windowOffset)
			#get an index of all samples withint lag window
			lagFilter = (sensor_df.index>= minTimeBound) & (sensor_df.index<= maxTimeBound)
			#get the samples in window
			lag = sensor_df.loc[lagFilter]
			
			#take the average/max/min/sum (based on ag op dictionary) of each column, and add to new dataset along with n2o data for t
			tmpDF =lag.agg(aggs)
			#tmpDF['timestamp']=t
			
			#append the aggreagtew window values for timestamp t to resulting lists
			for cName in sensor_df.columns:	

				#must make sure for summation aggregations, for empty columns with only NaN (missing values), 
				#pandas summ will returns all 0s, but we must replace it with Nan
				if aggs[cName] == "sum":
					#create an array that has True for NaN indices and False for non-NaN
					nanList = pd.isna(lag[cName])
					
					#detect if all values missing in lagged feature
					allNaN=True
					for flag in nanList:
						allNaN = allNaN and flag
						if not allNaN:
							break
					if allNaN:
						#replace all the 0s with Nan
						tmpDF[cName]=float("NaN")
				lagFeatureValueLists[colNameMap[cName]].append(tmpDF[cName])
		
						
		sensorDataDF=pd.DataFrame(data =lagFeatureValueLists,index=n2o_df.index)
		
		#concatenate all the lagged features and n2o samples
		#first dataset?
		if resSensorDF is None:
			resSensorDF=sensorDataDF			
		else:
			resSensorDF =pd.concat([resSensorDF,sensorDataDF],axis=1)
	
	
	
	#########################
	#			N2O data
	#########################

	n2oColsValueLists={}
	colNameMap = {}
	
	for cName in n2o_df.columns:
		n2oColsValueLists[cName]= []
		
	#timestamps = []
	
	#for every timestamp  t in n2o dataset :
	for t in n2o_df.index:
		
		
		minTimeBound =t - timedelta(minutes=newTemporalRes) #remove w minutes from timestamp t
		
		#compute the mean of n2o readings of all past readings from t in window
		#get an index of all samples withint lag window
		lagFilter = (originalN2ODF.index> minTimeBound) & (originalN2ODF.index<= t) #here we use > minbound instead of >= to avoid including extra readings
		#for example, a 30 minute window would containing 2 readings (t and t-newTemporalRes), so forcing > will mean only t included. Same for 1h windows, 2 readings (30
		#min intervals) will be included instead of 3 (1.5 h) readings
		
		#get the samples in window
		lag = originalN2ODF.loc[lagFilter]
		
		#take the average/max/min/sum (based on ag op dictionary) of each column, and add to new dataset along with n2o data for t
		tmpDF =lag.mean()
		#tmpDF['timestamp']=t
		
		#append the aggreagtew window values for timestamp t to resulting lists
		for cName in originalN2ODF.columns:											
			n2oColsValueLists[cName].append(tmpDF[cName])
		
		
		
	
	tmpn2oDF=pd.DataFrame(data =n2oColsValueLists,index=n2o_df.index)
	
	
	#concatenate all the lagged features and n2o samples
	#first dataset?
	if resN2ODF is None:
		resN2ODF=tmpn2oDF			
	else:
		resN2ODF =pd.concat([resN2ODF,tmpn2oDF],axis=1)

	#add the n2o readings to resuling dataset	
	
	resDF =pd.concat([resSensorDF,resN2ODF],axis=1)
	
	
	#save the merged csv file
	resDF.to_csv(outputPath,sep=",",index=True,encoding='utf-8')


	
	
#linearly interpolate missing values in columns, where a max block length can be 
#specified to only interpolate block up to that length, leaving longer blocks missing values
#inPath : csv input file path to read
#outPath : csv output path where interpolation results written to
#blkSizeThresh : the length of the block to interpolate up to, leaving blocks empty if length above this threshold
#exponentialLerp: False means linear interpolation is done, True means exponential interpolation done via log curve fitting
def lerpCSV(inPath,outPath,blkSizeThresh=-1,ignoreFirstCol=True,exponentialLerp=False):
	inDF = pd.read_csv(inPath, sep=",")
	resMap = {}
	
	ignoredFirstCol=False
	#iterate each column
	for cName in inDF.columns:			
		if ignoreFirstCol and not ignoredFirstCol:
			ignoredFirstCol=True
			resMap[cName] = inDF[cName]
			continue
		#interpolate missing values
		interpCol = lerp_array(inDF[cName],blkSizeThresh,exponentialLerp)
		
		#store results in tmp map
		resMap[cName] = interpCol
		#print(interpCol)
	outDF=pd.DataFrame(data =resMap)
	outDF.to_csv(outPath,sep=",",index=False,encoding='utf-8')
	
		
def lerp_array(arr,blkSizeThresh=-1,exponentialLerp=False):
	#the resulting array
	res = []
	
	missBlockLen = 0
	inMissValBlock=False	
	startIx=-1
	endIx=-1
	#iterate each value in the original array
	for i in range(len(arr)):
		val = arr[i]
		res.append(val) #create copy of array
		
		#misisng value?
		if val is None or np.isnan(val):
		
			if not inMissValBlock:
				#starting missing value block
				
				startIx=i
				
			inMissValBlock=True
			missBlockLen = missBlockLen+1
		else:
			if inMissValBlock:
				#ending missing value block
				
				endIx = i-1
			
				#determine if we should begin the interpolation
				
				#is the block small enough to interpolate
				interpolateFlag = False
				#special case: interpolate blocks of all sizes?
				if blkSizeThresh <= -1:
					interpolateFlag = True
				else:
					#only interpolate  blocks of size that respect threshold
					interpolateFlag= missBlockLen<=blkSizeThresh
					
				#can't interpolate if list starts with nulls/empty values
				if startIx==0:
					interpolateFlag=False
				
				if interpolateFlag:
					
					#linearly interpolate from startIx to endIx of column c using values at startIx-1 and endIx+1
					
					#2-value array of values before/after missing block
					#see https://numpy.org/doc/stable/reference/generated/numpy.interp.html
					#-1 and +1 cause want index to non-elements immediatly adjacent to empty block
					xp = [startIx-1,endIx+1]
					fp = [res[startIx-1],res[endIx+1]]
					
					#exponential interpolation?
					if exponentialLerp:
						
						#fit exponential cruve
						coefPair = np.polyfit(xp,np.log(fp),1)
						c1=coefPair[0]
						c2=coefPair[1]
						for j in range(startIx,endIx+1):
							res[j]=math.exp(c2)*math.exp(c1*j)
						#y ~= math.exp(c2)*math.exp(c1*x)
					else: #linear interpolation
					
						for j in range(startIx,endIx+1):
							res[j]=np.interp(j,xp,fp) 
				missBlockLen=0
				inMissValBlock=False
				startIx=-1 
				endIx=-1 
	return res
	
#df dataframe to split into multiple dataframes each with samples withint the temporal resolution  subsetSize
#subsetSize (in hours) the size of the window used to split datasets (note the window is sliding. The datasubsets don't overlap)
def temporalSplit(df,subsetSize):

	dfList = []
	#sort by timestamp (from early to later readings)
	df = df.sort_values(by=['timestamp'])

	#convert the timestamp column to a datetime format
	df['timestamp'] = pd.to_datetime(df['timestamp'])

	#set the index of dataframe to the timestamp
	df.set_index('timestamp',inplace=True)

	finishedFlag=False
	wStart = df.index[0] #starting of the window is first timestamp of dataset
	wEnd = wStart+ timedelta(hours=subsetSize)#ending of the window
	while not finishedFlag:

		#get all samples within time window [tStart,tEnd) in sensor data				
		#get an index of all samples withint lag window
		windowFilter = (df.index>= wStart) & (df.index< wEnd)
		
		#get the samples in window
		subset = df.loc[windowFilter]
		
		#only append non-empty blocks/clusters  to list
		if len(subset) != 0:
			dfList.append(subset)			

		#increment window by 1
		wStart = wEnd #starting of the window
		wEnd = wEnd + timedelta(hours=subsetSize)#ending of the window
		
		#the window is outside the timestamp range of the dataset?
		if wStart >df.index[-1]:
			finishedFlag=True
		
	return dfList
	
def temporalClusteredSampling(df,clusterSize,nFolds):
	#require random
	#require math
	
	#split Xc into days (size will be n matrices of data where n = number of days in datset)
	#clusters = Xc split into days (issue Xc doesn't have time stamps)
	#cluster samples by non-overlapping time windows of size clusterSize(hours)
	clusters = temporalSplit(df,clusterSize)

	#shuffle the list of clusters
	random.shuffle(clusters)
	
	
	#define the folds, where multiple clusters form a fold
	folds = []
			
	#the number of clusters per fold
	foldNClusters = int(math.floor(len(clusters)/nFolds))
	
	#print("number of clusters per fold: "+str(foldNClusters))
	#print("number clusters: "+str(len(clusters)))
	cix = 0
	foldDF = None
	while(cix < len(clusters)):
		
		#print("cix: "+str(cix))
		c=clusters[cix]
		
		if foldDF is None:
			#print("created deep copy of cluster")
			foldDF=c.copy(deep=True) # the fold starts off as first dataset cluster (deep copy)
		else:
			#print("appending the cluster to fold's dataframe")
			foldDF = pd.concat([foldDF,c],ignore_index=True)
			#foldDF.append(c) #append the clusters into a single dataset (the fold)
			
		cix = cix+1
		
		#finised populating a fold?
		if (cix % foldNClusters) == 0:
			#we only create a new fold and append resulting fold to list  
			#for folds before last fold. The last fold may get some overflow
			#if number of clusters can't be perfectly divided by number of folds
			if len(folds) < (nFolds-1):		
				#print("finished creating fold dataframe (in loop)")				
				folds.append(foldDF)
				foldDF=None
		
	
	if not foldDF is None:
		#print("finished creating fold dataframe (out loop)")					
		folds.append(foldDF)
	
	return folds	
	
def createListTmp(r1, r2):
 
	# Testing if range r1 and r2
	# are equal
	if (r1 == r2):
		return r1
 
	else:
		# Create empty list
		res = []
 
	# loop to append successors to
	# list until r2 is reached.
	while(r1 < r2+1 ):
	 
		res.append(r1)
		r1 += 1
	return res
  
  
  
#only run the below code if ran as script
#fuses 2 tempoeral datasets by taking lagged features (1 hour, 3 h, 6 h, 9 h, and 12 h)
if __name__ == '__main__':
		
	ap = argparse.ArgumentParser()
	

	ap.add_argument("-n", "--n2oDatasetInputPath", type=str, required=True,
		help="path to n2o emission dataset")
		
	ap.add_argument("-s", "--sensorDatasetInputPath", type=str, required=True,
		help="path to sensor data to dervied lagged features and fuse to the n2o dataset")
	
	ap.add_argument("-o", "--outputDatasetPath", type=str, required=True,
		help="path where to create output file of fused data")
	
	ap.add_argument("-t", "--temporalResolution", type=int, required=True,
		help="the target temporal resolution in minutes to save the dataset as (should be lower or equal than the n2o dataset)")
		
	args = vars(ap.parse_args())
	
	extractLaggedFeatures(args["n2oDatasetInputPath"],args["sensorDatasetInputPath"],args["outputDatasetPath"], args["temporalResolution"],[1,3,6,9,12],offsetSizes=None)