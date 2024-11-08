#code from following  tutorial https://nirpyresearch.com/variable-selection-method-pls-python/
from sys import stdout
 
import pandas as pd
import numpy as np

import resampling 
import sys
from datetime import datetime
 
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import random
import argparse

def runFeatureSelection(inPath,outResPath,outCoeffPath,year,n2oChamberName,resolution,inDatasetName,trials,temporalCVFlag,nFolds,blockClusterSize):
	
	start_time = datetime.now()

	start_timeStr = start_time.strftime("%Y-%m-%d %H:%M:%S")
	print("starting feature selection of "+inPath+" at "+start_timeStr)
	
	
	data = pd.read_csv(inPath)
	t,X,y =readDataset(inPath)
	#dataSubset,numFeatures,mse,r2,sorted_ind,trials= pls_variable_selection(X, y, trials)
	bestNumberFeatures,bestMSE,bestR2, sorted_ind,coeffs,trials=pls_variable_selection(t,X, y, trials,temporalCVFlag,nFolds,blockClusterSize)
	
	predVarCols = data.columns[1:-2] #predictor variable col names (ignore timestamps as first column, and n2o target variable is last column)
	
	#create map that indicates if a column is selected
	selectedColNameMap={}
	for i in range(len(sorted_ind)):
		
		colIx = sorted_ind[i]
		colName = predVarCols[colIx]
		#column selected?
		if i <bestNumberFeatures:
		
			selectedColNameMap[colName]=True
		else:
			#not selected
			selectedColNameMap[colName]=False
			

	end_time = datetime.now()

	end_timeStr = end_time.strftime("%Y-%m-%d %H:%M:%S")	
	
	secondsPerHour=60.0*60.0
	
	timeDiff=end_time-start_time
	
	ellapsedTimeHours = float(timeDiff.seconds)/secondsPerHour
	
	#RESCALE the MSE back to orignal scale
	n2oCol = data[data.columns[-1]] #n2o is last column
	std = n2oCol.std()
	m = n2oCol.mean()	
	if std== 0:
		std= 0.000001 #avoid divisions by 0
	bestMSE = (bestMSE*std)+m
	#write following to outPath
	#year, input dataset name, start time,end time, total duration (hours),trials,numberSelectedFeatures,MSE,R2,feature1,feature2,feature3,...featuren
	#,where year is growth season year, input dataset name is a friendly name of input dataset (e.g., no wind sensor) featurei is flag if feature i is selected, and 
	#trials is number of different random hyperparameters (number of components) we try
	
	
	header="year,input dataset name,n2oChamberName,resolution,start time,end time,total duration (hours),# trials,number selected features,MSE,R2"
	#append all non-timestamp and non-N2O column names to header
	for cName in data.columns[1:-2]:
		header = header +","+cName
	
	header = header + "\n"
	
	resultRow= str(year)+","+inDatasetName+","+str(n2oChamberName)+","+str(resolution)+","+start_timeStr+","+end_timeStr+","+str(ellapsedTimeHours)+","+str(trials)+","+str(bestNumberFeatures)+","+str(bestMSE)+","+str(bestR2)
	
	#add indication of 1 when column selected, 0 when not selected
	for cName in predVarCols:
		#column selected?
		if selectedColNameMap[cName]:
			resultRow = resultRow +",1"
		else:
			resultRow = resultRow +",0"
	
	
	
	#write results to file
	with open(outResPath,"w") as file:
		file.write(header)
		file.write(resultRow)
		
	#we also write a 2nd file that has coefficients values for each column
	coeffHeader = ""
	for i  in range(len(predVarCols)):
		cName = predVarCols[i]
		if i ==0:
			coeffHeader =cName
		else:
			coeffHeader = coeffHeader + "," + cName
			
	coeffHeader = coeffHeader+"\n"
	coeffRow = ""
	#write coefficients of each column
	for i  in range(len(predVarCols)):		
		coeff = coeffs[i]
		coeff=coeff[0]
		if i ==0:
			coeffRow = str(coeff)
		else:
			coeffRow = coeffRow + "," + str(coeff)
		
	
	#write coeef results to file
	with open(outCoeffPath,"w") as file:
		file.write(coeffHeader)
		file.write(coeffRow)
		
	print("finished feature selection on "+inPath+" at "+end_timeStr+", writing results to "+outResPath+" and "+outCoeffPath)
	
def readDataset(inPath):
	data = pd.read_csv(inPath)
	data = scaleData(data)
	t = data["timestamp"]
	X=data.values[:,1:-2]#don't include timestamp (index 0) and the n2o (last column, index -1)
	y=data.values[:,-1] #only the n2o data, column -1 (last scolumn)	
	return t,X,y
def scaleData(X):
	for c in X.columns:	
		if c != 'timestamp':
			#iterate over each column
			std = X[c].std()
			m = X[c].mean()
			
			if std== 0:
				std= 0.000001 #avoid divisions by 0
				
			#scale the column values by the standard deviation
			X[c]= (X[c]-m)/std
	return X
def pls_variable_selection(t,X, y, trials,temporalCVFlag,nFolds=5,blockClusterSize=24):

	#chosen max number coponents based on number of features
	numberSamples, numberFeatures = X.shape
	max_comp=numberFeatures-2

	if trials <0:
		trials =1
		print("number of trials provided to small: setting it to 1 trial")
	if trials > (max_comp):
		trials=max_comp-1
		print("number of trials provided to big: setting it to "+str(trials)+" trial")
		
	# Define MSE matrix to be populated where each row represents a trial and and columns are for features
	mse = np.zeros((trials,numberFeatures))
	r2 = np.zeros((trials,numberFeatures))
	
	random.seed(123) #always search for hyperparameters the same way, so datasets with same number features are processed identically

	#random search to tune hyperparameter (number of components (up to max_comp))
	cList = []
	number_of_components_list = random.sample(range(1, max_comp), trials)
	
	totalNumIterations =0
	for i in range(trials):
		numComp= number_of_components_list[i]
		for j in range(numberFeatures-numComp):
			totalNumIterations=totalNumIterations+1
			
	iteration=0
	# Loop over the number of PLS components via random search
	for i in range(trials):
	
		numComp= number_of_components_list[i]
		
		# Regression with specified number of components, using full number of features
		pls1 = PLSRegression(n_components=numComp)
		pls1.fit(X, y)

		coeffs = np.transpose(pls1.coef_,(1,0))
		# Indices of sort features according to ascending absolute value of PLS coefficients
		#(so sorted_ind[0] is the index of smallest coefficient, so coeffs[sorted_ind] would be sorted coeffs small to big)
		sorted_ind = np.argsort(np.abs(coeffs[:,0]))

		# Sort features accordingly 
		Xc = X[:,sorted_ind]
		#print(Xc.shape)
		# Discard one feature at a time of the sorted features (features with smallest coefficient (least influence) discarded first), where 
		#first run no features discarded, second run one feature discarded
		# regress, and calculate the MSE cross-validation
		for j in range(numberFeatures-numComp):
			
			pls2 = PLSRegression(n_components=numComp)
			pls2.fit(Xc[:, j:], y) #i don't think this is necessary, since cross-validation will fit the data and override it
						
			y_cv = makePredictions(pls2,Xc[:, j:],t,y,temporalCVFlag,nFolds,blockClusterSize)
									
			mse[i,j] = mean_squared_error(y, y_cv)
			r2[i,j] = r2_score(y,y_cv)
			
			iteration = iteration+1
			comp = 100*iteration/(totalNumIterations)
			stdout.write("\r%d%% completed" % comp)
			stdout.flush()
	stdout.write("\n")

	# # Calculate and the (i,j) position  of minimum in MSE matrix
	#INDICES of non zero MSE ENTRIES
	mseNonZeroIx = np.nonzero(mse)
	#the array of non-zero mse
	mseNonZero = mse[mseNonZeroIx]
	
	#array of flags indicating if the element is equal to minimum MSE 
	minMSEFlags = mse==np.min(mseNonZero)
	
	#find index pairs of minimum MSE element
	min_mse_rowIndices,min_mse_colIndices = np.where(minMSEFlags)
	#a few mse may be equals, so many indices may be returned. Just pick first one, since I think its safe to assume this won't happen (a more correct solution would be to pick identiacal MSE wiehre R2 is highest, but we just want basic info, not intersted in really maximizing performance)
	
	#sometimes may have more than 1  run with best MSE
	#if so, find the run among them with highest R2
	min_mse_rowIx=min_mse_rowIndices[0]
	min_mse_colIx = min_mse_colIndices[0]
	bestR2=r2[min_mse_rowIx,min_mse_colIx]
	for i in min_mse_rowIndices:
		for j in min_mse_colIndices:
			if bestR2 < r2[i,j]:
				min_mse_rowIx=i
				min_mse_colIx = j
	
	bestMSE = mse[min_mse_rowIx,min_mse_colIx]
	bestR2 = r2[min_mse_rowIx,min_mse_colIx] #may not be best R2, but is r2 of the best mse run
	bestNumberFeatures = numberFeatures-min_mse_colIx
	
	bestNumberComponents =number_of_components_list[min_mse_rowIx]
	
	stdout.write("\n")	

	# Calculate PLS with optimal components and export values
	pls = PLSRegression(n_components=bestNumberComponents)
	pls.fit(X, y)
	coeffs2=np.transpose(pls.coef_,(1,0))
	sorted_ind = np.argsort(np.abs(coeffs2[:,0]))
	
	
	#numFeatures,mse,r2,sorted_ind,trials
	return (bestNumberFeatures,bestMSE,bestR2, np.flip(sorted_ind),coeffs2,trials) #flip the sorted indices, so first index is best feature's ix

  	

#pls2: partial leasat squares model 
#Xc: predictor variable matrix (features are columns)
#t: timestamp column of same length as Xc's columns
#y: target variable column
#temporalCVFlag: True when temporal splits (clustered sampling where clusters are days), False for random sampling-based splits
#blockCVClusterSize: size of cluster when using block CV (in hours)
def makePredictions(pls2,Xc,t,y,temporalCVFlag,nFolds,blockCVClusterSize=24):
	#requires resampling
	#requires random
	#requires math
	#doing corvss vaildation where folds are defined by temporal splits?
	if temporalCVFlag:
	
		#is numpyt array?
		if isinstance(Xc, np.ndarray):
			#convert to pandas
			Xc = pd.DataFrame(Xc)
		
		#is numpyt array?
		if isinstance(y, np.ndarray):
			#convert to pandas
			y = pd.DataFrame(y,columns=["timestamp"])
			
		#insert t as first column into Xc
		Xc.insert(0, "timestamp", t, True)
		
		#create row id column and append to Xc as first column
		ids = []
		for i in range(len(t)):
			ids.append(i)
		
		Xc.insert(0, "rowID", ids, True)
		
		
		#make sure append the target variable to the matrix so the shuffling keeps track of 
		#target var too
		Xc.insert(len(Xc.columns), "targetVar", y, True)
		folds = resampling.temporalClusteredSampling(Xc,blockCVClusterSize,nFolds) # 24 hours (1 day) clusters
		#folds = temporalClusteredSampling(Xc,24,nFolds) # 24 hours (1 day) clusters
		
		predResultsMap={}
		predResultsMap["rowID"]=[]
		predResultsMap["pred"]=[]
		for i in range(len(folds)):
			testDF= folds[i]
			
			trainDF = None
			#create train datafarame
			for j in range(len(folds)):
			
				if j==i:
					continue
				if trainDF is None:
					trainDF=folds[j]
				else:
					trainDF = pd.concat([trainDF,folds[j]],ignore_index=True)

			trainX=trainDF.values[:,1:-2]#don't include rowid (index 0) and the target variable (last column, index -1)
			trainY = trainDF.values[:,-1]#only the traget variable column last column)
			testX=testDF.values[:,1:-2]#don't include rowid (index 0) and the target variable (last column, index -1)
			pls2.fit(trainX,trainY)			
			predY = pls2.predict(testX)
			
			#store results
			for rowIx in range(len(testDF["rowID"])):
				rowID = testDF["rowID"][rowIx]
				pred = predY[rowIx]
				predResultsMap["rowID"].append(rowID)
				predResultsMap["pred"].append(pred)		
		
		predDF = pd.DataFrame(data=predResultsMap)
				
		#sorte the predictions by row id	
		predDF = predDF.sort_values(by=['rowID'])
		return predDF["pred"]
		
	else:
		return cross_val_predict(pls2,Xc, y, cv=nFolds)

			
class ZeroR:

	
	def __init__(self):
		self.meanTargetVar=0
		pass
		
	def fit(self, X,Y):
		for val in Y:
			self.meanTargetVar=self.meanTargetVar+val
		self.meanTargetVar = self.meanTargetVar/len(Y)
	def predict(self,X):
		res = []
		for i in range(len(X)):
			res.append(self.meanTargetVar)
		return res
	
	
#only run the below code if ran as script
if __name__ == '__main__':
		
	ap = argparse.ArgumentParser()
	

	ap.add_argument("-y", "--year", type=int, required=True,
		help="year of the dataset")
		
	ap.add_argument("-c", "--chamber", type=str, required=True,
		help="gas chamber id in 'C<NUMBER>' format")
	
	ap.add_argument("-r", "--resolution", type=int, required=True,
		help="temporal resolution (in minutes) of the dataset")
		
	args = vars(ap.parse_args())
	
	year=str(args["year"])
	n2oChamberName=args["chamber"]
	resolution=str(args["resolution"])
	
	inPath = "input/datasets/"+year+"/"+n2oChamberName+"/"+resolution+"min.csv"

	outResPath = "output/feature-selection/feature-selection-summary.csv"
	outCoeffPath = "output/feature-selection/plsr-coefficients.csv"
	inDatasetName="optional user-defined dataset information"
	trials=100 #100 different hyperparameter choices are used to search for best hyperparameter choice
	temporalCVFlag=True#True means block cross-validation applied, False means random cross-validation applied
	nFolds=5 #number of folds
	blockClusterSize=24 #size of the block cross-validation cluster/block
	runFeatureSelection(inPath,outResPath,outCoeffPath,year,n2oChamberName,resolution,inDatasetName,trials,temporalCVFlag,nFolds,blockClusterSize)
