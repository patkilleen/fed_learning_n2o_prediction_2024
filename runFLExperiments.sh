#create directory for output files
mkdir output

mkdir output/fed-learn
python FLExperimenter.py --inFile  input/configs/2021/in2021C4Config30-LSTM.csv  --outDirectory output/fed-learn
