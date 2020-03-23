"""
    Author: Stephen Pauwels
"""
def convert2ints(file_in, file_out, header = True, dict = None):
    cnt = 0

    if dict is None:
        dict = []

    with open(file_in, "r") as fin:
        with open(file_out, "w") as fout:
            if header:
                fout.write(fin.readline())
            for line in fin:
                cnt += 1
                input = line.replace("\n", "").split(",")
                if len(dict) == 0:
                    for t in range(len(input)):
                        dict.append({})
                output = []
                attr = 0
                for i in input:
                    if i not in dict[attr]:
                        dict[attr][i] = len(dict[attr]) + 1
                    output.append(str(dict[attr][i]))
                    attr += 1
                fout.write(",".join(output))
                fout.write("\n")
    return cnt