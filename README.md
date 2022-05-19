# DeepIOStability

### Google colaboratory/ Jupyter notebook
https://colab.research.google.com/drive/1WQ-ICfVrrd_yJ_0LbRFw67Pypj3NhbMu?usp=sharing

### Requirements
* python3 (> 3.3)
  * Pytorch
  * joblib
  * scikit-learn
  * scipy
  * optuna
```
conda install pytorch
conda install -c conda-forge optuna
```
### Anaconda install
First, please install anaconda by the official anaconda instruction [https://conda.io/docs/user-guide/install/linux.html].

### Installation of DeepIOStability
```
pip install git+https://github.com/clinfo/DeepIOStability.git
```

## Command
The `dios` command for training
```
dios train --config <config file>
```

The `dios` command for plotting, where the plot signals are selected from validation data at random by default. 
```
dios-plot --config <config file>
```


## Demo

### Making a sample dataset
```
$ cd sample02
$ python make.py
```
### Execution
```
$ dios train --config config.json
```

#### linear model comparative method
```
$ dios-linear train --config config.json --method <method>
```
`<method>` is selected from
 - `ORT`
 - `MOESP`
 - `ORT_auto`
 - `MOESP_auto`
 
