opt=`dios-get-opt`
opt_d=`printf "%04d" ${opt}`
echo $opt_d
dios test --config study/trial${opt_d}/config.json
dios-plot --config study/trial${opt_d}/config.json

