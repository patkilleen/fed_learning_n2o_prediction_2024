#create directory for output files
mkdir output

mkdir output/hyper-param-select
python experimenter.py --inFile  input/configs/hyper-param-sel/2021C4Config180-LSTM.csv  --outDirectory output/hyper-param-select
