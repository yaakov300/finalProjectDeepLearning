import loader
import network

#add data
training_data, validation_data, test_data = loader.load_data_wrapper()
#init neural networks
net = network.Network([784, 30, 10])
#start tarnining
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)




