for d in `seq 0 8`
do
i=`printf "%02d" ${d}`

dios-linear train --config config/config${i}.json --method MOESP
dios-linear test  --config config/config${i}.json --method MOESP
dios-linear train --config config/config${i}.json --method MOESP_auto
dios-linear test  --config config/config${i}.json --method MOESP_auto

dios-linear train --config config/config${i}.json --method ORT
dios-linear test  --config config/config${i}.json --method ORT
dios-linear train --config config/config${i}.json --method ORT_auto
dios-linear test  --config config/config${i}.json --method ORT_auto



done
