import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image, numpy
from numpy.linalg import norm
from resizeimage import resizeimage
#save matrix as image
import scipy.misc

#image size = size * size
size = 5
#open image
img = Image.open('testimage.jpg')
#resize image
img = img.resize((size, size), Image.ANTIALIAS)
#convert to Gray level
matrix = numpy.asarray(img.convert('L'))
#normalize matrix
linfnorm = norm(matrix, axis=1, ord=np.inf)
matrix = matrix.astype(np.float) / linfnorm[:,None]

matrix1 = np.reshape(matrix, (size*size, 1))

vectorRezulte = x = np.array([1,0,0,0])

print matrix1

plt.imshow(matrix, cmap=plt.get_cmap('gray'))
plt.show()