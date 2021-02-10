for d in `seq 0 8`
do
i=`printf "%02d" ${d}`

#dios-linear train --config config/config${i}.json --method MOESP
dios-linear test  --config config/config${i}.json --method MOESP
#dios-linear train --config config/config${i}.json --method MOESP_auto
dios-linear test  --config config/config${i}.json --method MOESP_auto

#dios-linear train --config config/config${i}.json --method ORT
dios-linear test  --config config/config${i}.json --method ORT
#dios-linear train --config config/config${i}.json --method ORT_auto
dios-linear test  --config config/config${i}.json --method ORT_auto

#dios-linear train --config config/config${i}.json --method ARX
dios-linear test  --config config/config${i}.json --method ARX
#dios-linear train --config config/config${i}.json --method ARX_auto
dios-linear test  --config config/config${i}.json --method ARX_auto


#dios-linear train --config config/config${i}.json --method PWARX
dios-linear test  --config config/config${i}.json --method PWARX
#dios-linear train --config config/config${i}.json --method PWARX_auto
dios-linear test  --config config/config${i}.json --method PWARX_auto

done
