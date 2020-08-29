for d in `seq 0 8`
do
i=`printf "%02d" ${d}`
mkdir -p result${i}
dios train --config config/config${i}.json 
dios-plot --config config/config${i}.json
done

