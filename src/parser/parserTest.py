import re

file = open("vca_shape_learning.txt") #read original alignment file
fileContent = file.read()

imageReg = re.compile("(?<!:)ImageFileName")
imageList = imageReg.split(fileContent)[1:]

for image in imageList:
    templetReg = re.compile("(?<!:)Normal Template results")
    templetList = templetReg.split(image)
    for templet in templetList[1:]:
        pass

print len(imageList)
print imageList[0]