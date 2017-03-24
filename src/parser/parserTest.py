import re
import numpy as np

file = open("../../data/vca_shape_learning.txt") #read original alignment file
fileContent = file.read()

imageReg = re.compile("(?<!:)ImageFileName")
imageList = imageReg.split(fileContent)[1:]

imageArray =[]
typeArray = []

for image in imageList:

    templetReg = re.compile("(?<!:)Normal Template results")
    templetList = templetReg.split(image)
    templetArray = []
    for templet in templetList[1:]:
        tmp = re.search(',0, \[(.*)]', templet)
        tmp = tmp.group(1)

        for num in tmp.split(';'):
            templetArray.append(num)

    typeArray.append(re.search('LearningClassId=(.*)',templetList[0]).group(1)[0]);
    imageArray.append(np.array(templetArray))

training_data_size = len(imageArray)* 5 / 7
print training_data_size
# Randomly shuffle data
np.random.seed(training_data_size)
shuffle_indices = np.random.permutation(np.arange(len(imageArray)))
imageArray = np.array(imageArray)
typeArray = np.array(typeArray)


training_image_data = imageArray[shuffle_indices]
training_type_data = typeArray[shuffle_indices]





