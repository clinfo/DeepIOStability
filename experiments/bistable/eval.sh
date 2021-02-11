opt=`dios-get-opt`
opt_d=`printf "%04d" ${opt}`
echo $opt_d
#dios test --config study/trial${opt_d}/config.json
dios-eval ./result/log*test*.txt ./result_base/*test*.txt study/trial${opt_d}/log_test.txt
