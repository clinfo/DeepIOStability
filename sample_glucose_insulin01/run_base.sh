for d in `seq 0 8`
do
i=`printf "%02d" ${d}`
dios train --config config/config_base${i}.json $@
dios-plot --config config/config_base${i}.json
done

