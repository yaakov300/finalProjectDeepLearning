import loader
import network

#add data
training_data, validation_data, test_data = loader.load_data_wrapper()
#init neural networks
net = network.Network([405, 10, 10])
#start tarnining
net.SGD(training_data, 200, 107, 0.25, test_data=training_data)




