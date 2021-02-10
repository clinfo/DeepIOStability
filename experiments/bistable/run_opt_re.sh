dios train --config config_opt.json $@
dios test  --config config_opt.json $@
dios-plot  --config config_opt.json

dios train --config config_opt_base.json $@
dios test  --config config_opt_base.json $@
dios-plot  --config config_opt_base.json
