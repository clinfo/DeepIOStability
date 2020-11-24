# Dataset generated with linear system
## Step 1: data generation
```
python make.py
```

## Step 2: training and test


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


## Step 3: summarizing all results
```
python eval.py ./result*/log*test*.txt
```
This script outputs a `eval.tsv` as a summarized result. 


