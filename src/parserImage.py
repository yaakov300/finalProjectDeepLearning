import PIL.Image as Image
import glob
import os
import numpy as np
import cPickle

#image size = size * size
size = 28

#source images dir
src_dir = "img"

pkl_file_name = "source.pkl";



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

    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        # open image and convert to Gray level
        img = Image.open(jpgfile).convert('L')
        # resize image
        img = img.resize((size, size), Image.ANTIALIAS)

        # normalize image
        pix_val = np.array(list(img.getdata())).astype('float')
        pix_val = pix_val/np.linalg.norm(pix_val)

        image_list.append(pix_val)
        type_list.append(1)

    image_list = np.array(image_list,dtype='float32')
    type_list = np.array(type_list)


    training_data = (image_list,type_list)
    alidation_data = (image_list,np.array([2,2]))
    test_data = (image_list,np.array([3,3]))
    print training_data

    write_bucket_pkl((training_data,alidation_data,test_data),pkl_file_name)



print('-----------\n')


tr_d, va_d, te_d = load_pkl_data(pkl_file_name)


