#import dios.data.glucose
#import dios.data.glucose_insulin
#import dios.data.bistable
#import dios.data.limit_cycle
#import dios.data.linear
#import dios.data.nagumo
import argparse
from importlib import import_module

modes=["glucose","glucose_insulin","bistable","limit_cycle","linear","nagumo"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", type=str, default=None, nargs="?", help="/".join(modes)
    )
    parser.add_argument(
        "--path", type=str, default="dataset", help="output path"
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="output"
    )
    parser.add_argument(
        "--num", type=int, default=10000, help="#data"
    )
    parser.add_argument(
        "--test_num", type=int, default=9000, help="#test_data"
    )
    args = parser.parse_args()
    if args.mode in modes:
        mod = import_module("dios.data."+args.mode+"")
        mod.generate_dataset(N = args.num, M = args.test_num, name = args.prefix+args.mode, path=args.path)
    else:
        print("unknown mode:"+args.mode)

if __name__ == "__main__":
    main()

