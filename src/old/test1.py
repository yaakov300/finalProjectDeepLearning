import mnist_loader

from src.old import network

#add data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "finish load"
#init neural networks
net = network.Network([3600, 60, 10])
#start tarnining
net.SGD(training_data, 100, 1, 3.0, test_data=test_data)




