# Data generated with glucose insulin model

## Step 1: data generation
```
python make.py
```

## Step 2: checking generated data

```
python check.py
```

## Step 3: converting generated data
- Normalization
- Splitting train/test dataset

```
python convert.py
```

## Step 4: generate config files
For Deep IO stability cost function:
```
python make_config.py --config config.json
```

For standard neural network cost function (square error):

```
python make_config.py --config config_base.json
```


## Step 5: training and test


```
sh run.sh
```

For config_base.json:
```
sh run_base.sh
```

For linear model:
```
sh run_linear.sh
```


## Step 6: summarizing all results
```
python eval.py ./result*/log*test*.txt
```
This script outputs a `eval.tsv` as a summarized result. 


