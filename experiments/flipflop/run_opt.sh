## clean
rm -rf study
rm study.db

##
nohup dios-opt --config ./config.json --gpu 1 &

