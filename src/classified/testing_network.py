import json
import tensorflow as tf
import dataset
import numpy as np
import yaml
import os
from django.conf import settings


def testing_network(dict_network):
    # int

    # load config
    network = dict_network
    config = yaml.safe_load(open(os.path.join(base_dir, "config.yml")))
    modelConfig = yaml.safe_load(open(
        os.path.join(root_dir, config['training']['output_training_path'], network['model']['configPath'],
                     "config.yml")))

    # create graph
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(root_dir,
                                                    config['training']['output_training_path'],
                                                    network['model']['path'], network['model']['name']))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(root_dir,
                                                                config['training']['output_training_path'],
                                                                network['model']['path'], './')))
    graph = tf.get_default_graph()

    # read image for test
    num_channels = config['training']['num_channels']
    img_size = modelConfig['network']['img_size']

    def test():
        # test_path = config['testing']['path_testing']
        test_path = config['testing']['path_testing']
        classes = modelConfig['network']['classes']
        num_classes = len(classes)
        img_size_flat = img_size * img_size * num_channels
        test_images, test_ids, test_label = dataset.read_test_set(test_path, img_size, classes)
        # print "len = {}".format(len(test_images))
        # get y_pred and x pred layer
        accuracy = graph.get_tensor_by_name(config["tensor_name"]["accuracy"] + ":0")
        x = graph.get_tensor_by_name(config["tensor_name"]["input_x"] + ":0")
        y_true = graph.get_tensor_by_name(config["tensor_name"]["input_y_true"] + ":0")

        # create dicts for training
        x_batch = test_images.reshape(len(test_images), img_size_flat)
        y_test_images = test_label
        # print "y_test_images = {}".format(y_test_images)

        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        # print "accuracy = {:>6.1%}".format(sess.run(accuracy, feed_dict=feed_dict_testing))
        accuracy = sess.run(accuracy, feed_dict=feed_dict_testing)
        msg = "{:6.3%}".format(accuracy)
        return msg

    def use():
        test_path = config['testing']['path_using']
        classes = modelConfig['network']['classes']
        num_classes = len(classes)
        folder = config['visualizing']['folder']

        img_size_flat = img_size * img_size * num_channels
        test_images, test_ids, test_label = dataset.read_test_set(test_path, img_size, folder)

        # get y_pred and x pred layer
        y_pred = graph.get_tensor_by_name(config['tensor_name']["y_pred"] + ":0")
        x = graph.get_tensor_by_name(config["tensor_name"]["input_x"] + ":0")
        y_true = graph.get_tensor_by_name(config["tensor_name"]["input_y_true"] + ":0")

        # create dicts for training
        x_batch = test_images.reshape(len(test_images), img_size_flat)
        y_test_images = test_label
        # print "y_test_images = {}".format(y_test_images)
        y_test_images = np.zeros((len(test_images), 3))

        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        pred = sess.run(y_pred, feed_dict=feed_dict_testing)

        # msg = ""
        # for i in range(num_classes):
        #     msg += classes[i] + " = {:6.3%}, ".format(pred[0][i])
        # # print msg+"\n"
        # return msg

        dict_to_return = {}
        for i in range(num_classes):
            dict_to_return.update({classes[i]: "{:6.3%}".format(pred[0][i])})
        return dict_to_return


        # num_images = len(test_images)
        # print "len = {}".format(len(test_images))
        # for j in range(num_images):
        #     msg = ""
        #     for i in range(num_classes):
        #         # msg = "animals = {0:6.5%}, cars = {1:6.5%}, people = {2:6.5%}"
        #         msg += classes[i] + "= {:6.3%}, ".format(pred[j][i])
        #     print test_ids[j] +": " + msg+"\n"

    def run():
        if network['testing']:
            return test()
        else:
            return use()

    return run()


base_dir = os.path.dirname(__file__)
root_dir = settings.CLASSIFIED_SETTING['app']['root']
with open(os.path.join(base_dir, 'testing.json')) as data_file:
    network = json.load(data_file)
    data_file.close()
    print testing_network(network)
