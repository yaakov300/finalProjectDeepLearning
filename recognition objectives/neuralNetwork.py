import numpy as np

class neuralNetwork(object):

    def __init__(self, sizesNetwork):
        self.numberOfLayer = len(sizesNetwork)
        self.sizes = sizesNetwork
        self.biases = [np.random.randn(y, 1) for y in sizesNetwork[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizesNetwork[:-1], sizesNetwork[1:])]




