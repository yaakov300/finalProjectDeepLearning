import PIL.Image as Image
import glob
import os
import numpy as np
import cPickle

#image size = size * size
size = 28

#source images dir
src_dir = "img"

pkl_file_name = "datasetSY.pkl";



def write_pkl_file(list, file_name):
    # open the file for writing
    fileObject = open(file_name, 'wb')

    # this writes the object a to the
    # file named 'testfile'
    cPickle.dump(list, fileObject)

    # here we close the fileObject
    fileObject.close()

def load_pkl_data(file_name):
    try:
        fileObject = open(file_name, 'rb')
        training_data, validation_data, test_data = cPickle.load(fileObject)
        return (training_data, validation_data, test_data)
    except IOError:
        userAnswer = raw_input("pklList.pkl not exist press yes to start training other to exit:")
        if (userAnswer != 'yes'):
            exit()
    return None


def createPklFile():
    image_list = []
    type_list = []
    categories = get_immediate_subdirectories(src_dir)
    for category in range(len(categories)):
        for jpgfile in glob.iglob(os.path.join(src_dir+"/"+categories[category], "*.jpg")):
            # open image and convert to Gray level
            img = Image.open(jpgfile).convert('L')
            # resize image
            img = img.resize((size, size), Image.ANTIALIAS)

            # normalize image
            pix_val = np.array(list(img.getdata())).astype('float')
            pix_val = pix_val/np.linalg.norm(pix_val)

            image_list.append(pix_val)
            type_list.append(category)

    # convert to nappy array
    image_list = np.array(image_list,dtype='float32')
    type_list = np.array(type_list)

    training_data = (image_list,type_list)
    alidation_data = (image_list,type_list)
    test_data = (image_list,type_list)
    print training_data

    write_pkl_file((training_data,alidation_data,test_data),os.path.join("../data",pkl_file_name))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

createPklFile()
def init():

    print('-----------\n')

    createPklFile(["sss","bbb"])

    tr_d, va_d, te_d = load_pkl_data(pkl_file_name)


