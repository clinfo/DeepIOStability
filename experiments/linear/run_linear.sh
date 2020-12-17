dios-linear train --config config.json --method MOESP
dios-linear test  --config config.json --method MOESP
dios-linear train --config config.json --method MOESP_auto
dios-linear test  --config config.json --method MOESP_auto

dios-linear train --config config.json --method ORT
dios-linear test  --config config.json --method ORT
dios-linear train --config config.json --method ORT_auto
dios-linear test  --config config.json --method ORT_auto

dios-linear train --config config.json --method ARX
dios-linear test  --config config.json --method ARX
dios-linear train --config config.json --method ARX_auto
dios-linear test  --config config.json --method ARX_auto

dios-linear train --config config.json --method PWARX
dios-linear test  --config config.json --method PWARX
dios-linear train --config config.json --method PWARX_auto
dios-linear test  --config config.json --method PWARX_auto



