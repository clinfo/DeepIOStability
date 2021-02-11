import re
import sys
import argparse
import os
import copy
import json

def parse_log(filenames, output="eval.tsv"):
    data={}
    items=["mean error", "data io gain", "test io gain", "stable error", "gamma"]
    for filename in sys.argv[1:]:
        obj={}
        for line in open(filename):
            m = re.match(r'([^:]+): (.*)', line.strip())
            if m:
                key=m.group(1)
                val=m.group(2)
                if key in items:
                    obj[key]= float(val)
        data[filename]=obj

    fp=open(output,"w")
    s="\t".join(["filename"]+items)
    fp.write(s)
    fp.write("\n")
    for filename,obj in data.items():
        line=[filename]
        for item in items:
            if item in obj:
                line.append(str(obj[item]))
            else:
                line.append("")
        s="\t".join(line)
        fp.write(s)
        fp.write("\n")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames", type=str, default=None, nargs="+", help="config json file"
    )
    parser.add_argument(
        "--output", type=str, default="eval.tsv", help="output"
    )
    parser.add_argument(
        "--db", type=str, default="./study.db", help="config json file"
    )
    args = parser.parse_args()

    data=parse_log(args.filenames, output=args.output)
    print(data)

if __name__ == "__main__":
    main()

