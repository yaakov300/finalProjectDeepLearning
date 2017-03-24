import re

import cPickle
import numpy as np
import os

pkl_file_name = "vca_datasetSY.pkl";

def write_pkl_file(list, file_name):
    # open the file for writing
    fileObject = open(file_name, 'wb')

    # this writes the object a to the
    cPickle.dump(list, fileObject)

    # here we close the fileObject
    fileObject.close()

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
validation_data_size = (len(imageArray)-training_data_size)/2
print training_data_size
# Randomly shuffle data
np.random.seed(training_data_size)
shuffle_indices = np.random.permutation(np.arange(len(imageArray)))
imageArray = np.array(imageArray,dtype='float32')
typeArray = np.array(typeArray)


training_image_data = imageArray[shuffle_indices[:training_data_size+1]]
training_type_data = typeArray[shuffle_indices[:training_data_size+1]]
validation_image_data = imageArray[shuffle_indices[training_data_size+1:training_data_size+validation_data_size+1]]
validation_type_data = typeArray[shuffle_indices[training_data_size+1:training_data_size+validation_data_size+1]]
test_image_data = imageArray[shuffle_indices[training_data_size+validation_data_size+1:]]
test_type_data = typeArray[shuffle_indices[training_data_size+validation_data_size+1:]]

# convert to nappy array
training_data = (training_image_data,training_type_data)
validation_data = (validation_image_data,validation_type_data)
test_data = (test_image_data,test_type_data)

write_pkl_file((training_data,validation_data,test_data), os.path.join("../../data", pkl_file_name))



