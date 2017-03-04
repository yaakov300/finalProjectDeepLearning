import re

uniprotFile=open("vca_shape_learning.txt") #read original alignment file
uniprotFileContent=uniprotFile.read()

myre = re.compile("(?<!:)ImageFileName")
uniprotFileList = myre.split(uniprotFileContent)


print len(uniprotFileList)
print uniprotFileList[751]