# DeepIOStability

### Google colaboratory/ Jupyter notebook
https://colab.research.google.com/drive/1WQ-ICfVrrd_yJ_0LbRFw67Pypj3NhbMu?usp=sharing

### Requirements
* python3 (> 3.3)
  * Pytorch
  * joblib


### Anaconda install
First, please install anaconda by the official anaconda instruction [https://conda.io/docs/user-guide/install/linux.html].

### DeepIOStabilityインストール
```
pip install git+https://github.com/clinfo/DeepIOStability.git
```

## コマンド
インストールするとdiosコマンドが使えるようになる
```
dios train --config <config file>
```

validationデータからランダムにサンプルを選んでプロットする場合は、学習後にdios-plotを用いる
```
dios-plot --config <config file>
```


## サンプルの動かし方

以下のコマンドで動かすことが可能です．

### データの作成
```
$ cd sample02
$ python make.py
```
学習実行
```
$ dios train --config config.json
```
