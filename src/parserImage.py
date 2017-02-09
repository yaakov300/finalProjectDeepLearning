import numpy as np
import numpy
import PIL.Image as Image
import glob
import os
from numpy.linalg import norm

#image size = size * size
size = 5

#source images dir
src_dir = "img"
image_list = []
type_list = []

for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    # open image
    img = Image.open(jpgfile)
    # resize image
    img = img.resize((size, size), Image.ANTIALIAS)
    # convert to Gray level
    img = img.convert('L')
    pix_val = list(img.getdata())
    print pix_val[:, None]
    print ("===========\n")
    matrix = numpy.asarray(img)
    # print matrix


    # normalize matrix
    linfnorm = norm(matrix, axis=1, ord=np.inf)
    print linfnorm[:, None]


    matrix = matrix.astype(np.float) / linfnorm[:, None]

    image_list.append(matrix)
    type_list.append(0)
# print image_list
training_data = [image_list,numpy.asarray(type_list)]
# print training_data
# matrix1 = np.reshape(matrix, (size*size, 1))

# vectorRezulte = x = np.array([1,0,0,0])

# print matrix1

# imshow(matrix, cmap=get_cmap('gray'))
# show()