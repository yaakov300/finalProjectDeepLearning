import re

shakes = open("../data/vca_shape_learning.txt", "r")

for line in shakes:
    if re.match("(.*)(L|l)ove(.*)", line):
        print line