## clean
rm -rf study_base
rm study_base.db

##
nohup dios-opt --config ./config_base.json  --study_name study_base --db ./study_base.db  --output ./study_base.csv --gpu 1 &
nohup dios-opt --config ./config_base.json  --study_name study_base --db ./study_base.db  --output ./study_base.csv --gpu 2 &

