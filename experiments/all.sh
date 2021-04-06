do_experiment () {
        ex_name=$1
        echo "...start ${ex_name}"

        cd ${ex_name}

        #dios-dataset ${ex_name} --prefix 001 --num 100 --train_num 90 &
        #dios-dataset ${ex_name} --prefix 010 --num 1000 --train_num 900 &
        dios-dataset ${ex_name} --prefix 100 --num 10000 --train_num 9000 &
        wait
        mkdir -p config

        ###### 100
        pre=100
        dios-config --config config.json --save_config config/${pre}config.json \
                --result_path ${pre}result/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test

        dios-config --config config/${pre}config.json --save_config config/${pre}config_stable.json \
                --result_path ${pre}result_stable/ \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0
        
        dios-config --config config/${pre}config.json --save_config config/${pre}config_base.json \
                --result_path ${pre}result_base/ \
                --stable_f false \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0


        dios train,test --config config/${pre}config_base.json --gpu 0 &
        dios train,test --config config/${pre}config_stable.json --gpu 1 &
        dios train,test --config config/${pre}config.json      --gpu 2 &
        wait
        dios-plot --config config/${pre}config.json
        dios-plot --config config/${pre}config_base.json
        dios-plot --config config/${pre}config_stable.json

        config=config/${pre}config.json
        methods="MOESP MOESP_auto ORT ORT_auto ARX ARX_auto PWARX PWARX_auto"
        for m in $methods; do
        dios-linear train,test --config ${config} --method ${m} &
        done
        wait

        ####### eval
        dios-eval ./*result/log*test*.txt ./*result_base/*test*.txt
        cd ..

}


do_experiment bistable
do_experiment glucose
do_experiment glucose_insulin
do_experiment limit_cycle
do_experiment linear
do_experiment nagumo

