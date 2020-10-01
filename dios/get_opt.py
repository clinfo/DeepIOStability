import sys
import argparse

def get_opt(filename):
    fp=open(filename)
    h=next(fp)
    data=[]
    for line in fp:
        arr=line.strip().split(",")
        trial=arr[1]
        f=float(arr[2])
        #print((f,trial))
        data.append((f,trial))
    data=sorted(data)
    return data[0][1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="study.csv", help="output csv file"
    )
    args = parser.parse_args()
    infile = args.input
    opt_id=get_opt(infile)
    print(opt_id)

if __name__ == "__main__":
    main()

