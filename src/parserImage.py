import PIL.Image as Image
import glob
import os
import numpy as np
import cPickle


#image size = size * size
size = 60

#
training_size = 5/6
testing_size = 1/6

#source images dir
src_dir = "img"

pkl_file_name = "datasetSY.pkl";



def write_pkl_file(list, file_name):
    # open the file for writing
    fileObject = open(file_name, 'wb')

    # this writes the object a to the
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
    test_image_list = []
    test_type_list = []
    training_image_list = []
    training_type_list = []
    validation_image_list = []
    validation_type_list = []

    # each folder in src_dir is a category
    categories = get_immediate_subdirectories(src_dir)
    for category in range(len(categories)):
        images =glob.glob(os.path.join(src_dir + "/" + categories[category], "*.jpg"))
        number_of_file = len(images)
        # set training size 83% of the image category
        training_size = np.math.ceil(0.83 * number_of_file)
        print training_size
        testing_size = number_of_file - training_size
        print testing_size
        for jpgfile in range(number_of_file):
            # open image and convert to Gray level
            img = Image.open(images[jpgfile]).convert('L')
            # resize image
            img = img.resize((size, size), Image.ANTIALIAS)

            # normalize image
            pix_val = np.array(list(img.getdata())).astype('float')
            pix_val = pix_val/np.linalg.norm(pix_val)

            if jpgfile+1 <= training_size :
                training_image_list.append(pix_val)
                training_type_list.append(category)
                print "training"

                if jpgfile+1 <= testing_size :
                    validation_image_list.append(pix_val)
                    validation_type_list.append(category)
                    print "validation"

            else:
                test_image_list.append(pix_val)
                test_type_list.append(category)
                print "test"


    # convert to nappy array
    training_data = (np.array(training_image_list,dtype='float32'),np.array(training_type_list))
    validation_data = (np.array(validation_image_list,dtype='float32'),np.array(validation_type_list))
    test_data = (np.array(test_image_list,dtype='float32'),np.array(test_type_list))


    write_pkl_file((training_data,validation_data,test_data),os.path.join("../data",pkl_file_name))

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

createPklFile()
def init():

    print('-----------\n')

    createPklFile(["sss","bbb"])

    tr_d, va_d, te_d = load_pkl_data(pkl_file_name)


