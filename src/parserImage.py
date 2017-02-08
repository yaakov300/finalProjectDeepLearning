
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Image, numpy
from numpy.linalg import norm


size = 10, 10
img = Image.open('testimage.jpg',mode= 'r')
img.thumbnail(size, Image.ANTIALIAS)
matrix = numpy.asarray(img.convert('L'))
#print matrix

linfnorm = norm(matrix, axis=1, ord=np.inf)
matrix.astype(np.float) / linfnorm[:,None]
print matrix.astype(np.float) / linfnorm[:,None]

print np.reshape(matrix.astype(np.float) / linfnorm[:,None], (matrix.size, 1))

plt.imshow(matrix, cmap = plt.get_cmap('gray'))
plt.show()