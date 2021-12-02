do_experiment () {
        ex_name=$1
        num=$2

        pre=`printf "%06d" ${num}`
        echo "...start ${ex_name} ${pre}"

        cd ${ex_name}
        train_num=`python -c "print(int(0.9*${num}))"`

        config=config/${pre}config_linear.json
        ####### eval
        dios-eval ./*result_*/*test*.txt
        cd ..

}

do_experiment bistable 100
do_experiment glucose 100
do_experiment glucose_insulin 100
do_experiment limit_cycle 100 
do_experiment linear 100
#do_experiment nagumo 100

do_experiment bistable 1000
do_experiment glucose 1000
do_experiment glucose_insulin 1000
do_experiment limit_cycle 1000
do_experiment linear 1000
#do_experiment nagumo 1000

do_experiment bistable 3000
do_experiment glucose 3000
do_experiment glucose_insulin 3000
do_experiment limit_cycle 3000
do_experiment linear 3000
#do_experiment nagumo 3000

python all_eval.py
