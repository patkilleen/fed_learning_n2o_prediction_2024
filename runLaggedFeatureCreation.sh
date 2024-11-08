#create directory for output files
mkdir output

python resampling.py --n2oDatasetInputPath raw-data/licor-n2o-data/2021.csv  --sensorDatasetInputPath raw-data/pessl-soil-weather-data/2021/sensor-node-00209FC8.csv --outputDatasetPath output/2021-n2o-00209FC8-fused-30min.csv --temporalResolution 30
python resampling.py --n2oDatasetInputPath raw-data/licor-n2o-data/2021.csv  --sensorDatasetInputPath raw-data/pessl-soil-weather-data/2021/sensor-node-00209FC8.csv --outputDatasetPath output/2021-n2o-00209FC8-fused-180min.csv --temporalResolution 180
