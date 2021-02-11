import numpy as np
import joblib
import json
import argparse

from dios.dios import get_default_config, NumPyArangeEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--save_config", type=str, default=None, nargs="?", help="config json file"
    )
    ## config
    for key, val in get_default_config().items():
        if type(val) is int:
            parser.add_argument("--"+key, type=int, default=val, help="[config integer]")
        elif type(val) is float:
            parser.add_argument("--"+key, type=float, default=val, help="[config float]")
        elif type(val) is bool:
            parser.add_argument("--"+key, type=bool, default=val, help="[config float]")
            #parser.add_argument("--"+key, action="store_true", help="[config bool]")
        else:
            parser.add_argument("--"+key, type=str, default=val, help="[config string]")
    args = parser.parse_args()
    #
    config={}
    if args.config is None:
        if not args.no_config:
            parser.print_help()
            quit()
    else:
        print("[LOAD]",args.config)
        fp = open(args.config, "r")
        config.update(json.load(fp))
    for key, val in get_default_config().items():
        config[key]=getattr(args,key)
    #
    if args.save_config is not None:
        print("[SAVE] config: ", args.save_config)
        fp = open(args.save_config, "w")
        json.dump(
            config,
            fp,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            cls=NumPyArangeEncoder,
        )


if __name__ == "__main__":
    main()
