do_experiment () {
        ex_name=$1
        num=$2

        pre=`printf "%06d" ${num}`
        echo "...start ${ex_name} ${pre}"

        cd ${ex_name}
        train_num=`python -c "print(int(0.9*${num}))"`

        dios-dataset ${ex_name} --prefix ${pre} --num ${num} --train_num ${train_num} &
        #wait
        mkdir -p config

        ###### 100
        
        dios-config --config config.json \
                --save_config config/${pre}config_fgh_loss.json \
                --result_path ${pre}result_fgh_loss/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type fgh \
                --pretrain_epoch 0 \
                --alpha_state 0.0
         
        dios-config --config config.json \
                --save_config config/${pre}config_fg_loss.json \
                --result_path ${pre}result_fg_loss/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type fg \
                --pretrain_epoch 0 \
                --alpha_state 0.0
        

        dios-config --config config.json \
                --save_config config/${pre}config_loss.json \
                --result_path ${pre}result_loss/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type none \
                --pretrain_epoch 0 \
                --alpha_state 0.0

        dios-config --config config.json \
                --save_config config/${pre}config_fgh.json \
                --result_path ${pre}result_fgh/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type fgh \
                --pretrain_epoch 0 \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0
        
        dios-config --config config.json \
                --save_config config/${pre}config_fg.json \
                --result_path ${pre}result_fg/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type fg \
                --pretrain_epoch 0 \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0
       
        dios-config --config config.json \
                --save_config config/${pre}config_f.json \
                --result_path ${pre}result_f/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type f \
                --pretrain_epoch 0 \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0
        
        dios-config --config config.json \
                --save_config config/${pre}config_vanilla.json \
                --result_path ${pre}result_vanilla/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test \
                --stable_type none \
                --pretrain_epoch 0 \
                --alpha_HJ 0.0 \
                --alpha_gamma 0.0 \
                --alpha_state 0.0
        
        dios-config --config config.json \
                --save_config config/${pre}config_linear.json \
                --result_path ${pre}result_linear/ \
                --data_train dataset/${pre}${ex_name}.train \
                --data_test  dataset/${pre}${ex_name}.test

        mkdir -p ${pre}result_fgh_loss
        mkdir -p ${pre}result_fg_loss
        mkdir -p ${pre}result_loss
        mkdir -p ${pre}result_fgh
        mkdir -p ${pre}result_fg
        mkdir -p ${pre}result_f
        mkdir -p ${pre}result_vanilla
        echo "cd ${ex_name} && dios train,test --config config/${pre}config_fgh_loss.json   >${pre}result_fgh_loss/error.txt 2>&1" >> ../all4parallel_gpu.list
        #echo "cd ${ex_name} && dios train,test --config config/${pre}config_fg_loss.json    >${pre}result_fg_loss/error.txt 2>&1"  >> ../all4parallel_gpu.list
        #echo "cd ${ex_name} && dios train,test --config config/${pre}config_loss.json       >${pre}result_loss/error.txt 2>&1" >> ../all4parallel_gpu.list
        echo "cd ${ex_name} && dios train,test --config config/${pre}config_fgh.json        >${pre}result_fgh/error.txt 2>&1" >> ../all4parallel_gpu.list
        #echo "cd ${ex_name} && dios train,test --config config/${pre}config_fg.json         >${pre}result_fg/error.txt 2>&1" >> ../all4parallel_gpu.list
        #echo "cd ${ex_name} && dios train,test --config config/${pre}config_f.json          >${pre}result_f/error.txt 2>&1" >> ../all4parallel_gpu.list
        echo "cd ${ex_name} && dios train,test --config config/${pre}config_vanilla.json    >${pre}result_vanilla/error.txt 2>&1" >> ../all4parallel_gpu.list
        
        #dios-plot --config config/${pre}config_fgh_loss.json
        #dios-plot --config config/${pre}config_fg_loss.json
        #dios-plot --config config/${pre}config_loss.json
        #dios-plot --config config/${pre}config_fgh.json
        #dios-plot --config config/${pre}config_fg.json
        #dios-plot --config config/${pre}config_f.json
        #dios-plot --config config/${pre}config_vanilla.json

        config=config/${pre}config_linear.json
        methods="MOESP MOESP_auto ORT ORT_auto ARX ARX_auto PWARX PWARX_auto"
        for m in $methods; do
          echo "cd ${ex_name} && dios-linear train,test --config ${config} --method ${m} >${pre}linear_error.txt 2>&1" >> ../all4parallel_cpu.list
        done

        ####### eval
        #dios-eval ./*result_*/*test*.txt
        cd ..

}

echo -n "" > all4parallel_gpu.list
echo -n "" > all4parallel_cpu.list

#do_experiment bistable 100
#do_experiment glucose 100
#do_experiment glucose_insulin 100
#do_experiment limit_cycle 100 
do_experiment linear 100
#do_experiment nagumo 100

#do_experiment bistable 500
#do_experiment glucose 500
#do_experiment glucose_insulin 500
#do_experiment limit_cycle 500
do_experiment linear 500
#do_experiment nagumo 500


#do_experiment bistable 1000
#do_experiment glucose 1000
#do_experiment glucose_insulin 1000
#do_experiment limit_cycle 1000
do_experiment linear 1000
#do_experiment nagumo 1000

#do_experiment bistable 3000
#do_experiment glucose 3000
#do_experiment glucose_insulin 3000
#do_experiment limit_cycle 3000
#do_experiment linear 3000
#do_experiment nagumo 3000

