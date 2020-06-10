# DeepIOStability

### Requirements
* python3 (> 3.3)
  * Pytorch (>2.0)
  * joblib

### Anaconda install
First, please install anaconda by the official anaconda instruction [https://conda.io/docs/user-guide/install/linux.html].

### DeepIOStabilityインストール
```
pip install 
```

## コマンド
```
dios train --config <config file>
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
