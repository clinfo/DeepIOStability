import re
import sys

data={}
items=["mean error", "stable error", "data io gain", "test io gain"]
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
print(data)


fp=open("eval.tsv","w")
s="\t".join(["filename"]+items)
fp.write(s)
fp.write("\n")
for filename,obj in data.items():
    line=[filename]
    for item in items:
        if item in obj:
            line.append("{:2.2e}".format(obj[item]))
        else:
            line.append("")
    s="\t".join(line)
    fp.write(s)
    fp.write("\n")
