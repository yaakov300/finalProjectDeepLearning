import json
import cv2
import tensorflow as tf
import dataset
import numpy as np
import matplotlib.pyplot as plt
import math
import yaml



def visual_network(dict_network):
    # init

    network = dict_network
    # load config
    config = yaml.safe_load(open("config.yml"))
    modelConfig = yaml.safe_load(open(config['training']['output_training_path']+network['model']['configPath']+"config.yml"))

    # create graph
    sess = tf.Session()
    saver = tf.train.import_meta_graph(config['training']['output_training_path']+network['model']['path']+network['model']['name'])
    saver.restore(sess, tf.train.latest_checkpoint(config['training']['output_training_path']+network['model']['path']+'./'))
    graph = tf.get_default_graph()

    # read images
    test_path = config['visualizing']['path']#constant value
    folder = config['visualizing']['folder'] #constant value
    num_channels = config['training']['num_channels'] #constant value
    img_size = modelConfig['network']['img_size']

    img_size_flat = img_size * img_size * num_channels
    test_images, test_ids, test_label = dataset.read_test_set(test_path, img_size,folder)

    # get tensor from graph
    x= graph.get_tensor_by_name(config['tensor_name']['input_x']+":0")
    x_batch = test_images.reshape(len(test_images), img_size_flat)
    y_true = graph.get_tensor_by_name(config['tensor_name']['input_y_true']+":0")
    y_test_images = test_label

    # create dict to runing in graph
    y_test_images = np.zeros((len(test_images), network['data']['numClasses']))
    feed_dict_testing = {x: x_batch, y_true: y_test_images}


    def visualitzing():
        return getActivations(graph.get_tensor_by_name(network['layers']['Names']+":0"), feed_dict_testing)


    def getActivations(layer, dict):
        units = sess.run(layer,feed_dict=dict)
        # return plotNNFilter(units)
        return imagesNNFilter(units)


    def plotNNFilter(units):
        images = []
        filters = units.shape[3]
        plt.figure(3, figsize=(20,20))
        n_columns = 10
        n_rows = math.ceil(filters / n_columns) + 1
        for j in xrange(0, filters, network['return']['steps']):
            plt.subplot(n_rows, n_columns, j+1)
            plt.title('Filter ' + str(j))
            plt.imshow(units[0, :, :, j], interpolation="nearest", cmap="gray")
        plt.show()
        return []

    def imagesNNFilter(units):
        images = []
        filters = units.shape[3]
        for j in xrange(0, filters, network['return']['steps']):
            images.append(units[0, :, :, j])
        return images;

    return visualitzing()

# with open('visualising.json') as data_file:
#     network = json.load(data_file)
#     data_file.close()
#     dict_network = network
#     print len(visual_network(dict_network))